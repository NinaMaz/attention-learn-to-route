import torch
from typing import NamedTuple
from utils.boolmask import mask_long2bool, mask_long_scatter


class StateCVRP(NamedTuple):
    # Fixed input
    coords: torch.Tensor  # Depot + loc
    demand: torch.Tensor

    # If this state contains multiple copies (i.e. beam search) for the same instance,
    #  then for memory efficiency the coords and demands tensors are not kept multiple
    #  times, so we need to use the ids to index the correct rows.
    ids: torch.Tensor  # Keeps track of original fixed data index of rows

    # State
    prev_a: torch.Tensor
    used_capacity: torch.Tensor
    visited_: torch.Tensor  # Keeps track of nodes that have been visited
    lengths: torch.Tensor
    last_length: torch.Tensor
    cur_coord: torch.Tensor
    i: torch.Tensor  # Keeps track of step
    done: torch.Tensor

    VEHICLE_CAPACITY = 1.0  # Hardcoded

    @property
    def visited(self):
        if self.visited_.dtype == torch.uint8:
            return self.visited_.squeeze(1)
        else:
            return mask_long2bool(self.visited_, n=self.demand.size(-1)).squeeze(1)

    @property
    def dist(self):
        return (self.coords[:, :, None, :] - self.coords[:, None, :, :]).norm(
            p=2, dim=-1
        )

    def __getitem__(self, key):
        assert torch.is_tensor(key) or isinstance(
            key, slice
        )  # If tensor, idx all tensors by this tensor:
        return self._replace(
            ids=self.ids[key],
            prev_a=self.prev_a[key],
            used_capacity=self.used_capacity[key],
            visited_=self.visited_[key],
            lengths=self.lengths[key],
            cur_coord=self.cur_coord[key],
        )

    # Warning: cannot override len of NamedTuple, len should be number of fields,
    #  not batch size
    # def __len__(self):
    #     return len(self.used_capacity)

    @staticmethod
    def initialize(input, visited_dtype=torch.uint8):

        depot = input["depot"]
        loc = input["loc"]
        demand = input["demand"]

        batch_size, n_loc, _ = loc.size()
        return StateCVRP(
            coords=torch.cat((depot[:, None, :], loc), -2),
            demand=demand,
            ids=torch.arange(batch_size, dtype=torch.int64, device=loc.device)[
                :, None
            ],  # Add steps dimension
            prev_a=torch.zeros(batch_size, 1, dtype=torch.long, device=loc.device),
            used_capacity=demand.new_zeros(batch_size, 1),
            # Visited as mask is easier to understand, as long more memory efficient
            visited_=(
                # Keep visited_ with depot so we can scatter efficiently
                torch.zeros(
                    batch_size, 1, n_loc + 1, dtype=torch.uint8, device=loc.device
                )
                if visited_dtype == torch.uint8
                else torch.zeros(
                    batch_size,
                    1,
                    (n_loc + 63) // 64,
                    dtype=torch.int64,
                    device=loc.device,
                )  # Ceil
            ),
            lengths=torch.zeros(batch_size, device=loc.device),
            last_length=torch.zeros(batch_size, device=loc.device),
            done=torch.zeros(batch_size, dtype=torch.bool, device=loc.device),
            cur_coord=input["depot"][:, None, :],  # Add step dimension
            i=torch.zeros(
                1, dtype=torch.int64, device=loc.device
            ),  # Vector with length num_steps
        )

    def get_final_cost(self):

        assert self.all_finished()

        return self.lengths

    def get_cost(self):
        return self.last_length

    def update(self, selected):

        assert self.i.size(0) == 1, "Can only update if state represents single step"

        # Update the state
        selected = selected[:, None]  # Add dimension for step
        prev_a = selected
        n_loc = self.demand.size(-1)  # Excludes depot

        # Add the length
        cur_coord = self.coords[self.ids, selected]
        # cur_coord = self.coords.gather(
        #     1,
        #     selected[:, None].expand(selected.size(0), 1, self.coords.size(-1))
        # )[:, 0, :]
        last_length = (cur_coord - self.cur_coord).norm(p=2, dim=-1).squeeze(1)  # (batch_size)
        lengths = self.lengths + last_length

        # Not selected_demand is demand of first node (by clamp) so incorrect for nodes
        #  that visit depot!
        # selected_demand = self.demand.gather(
        #     -1, torch.clamp(prev_a - 1, 0, n_loc - 1))
        selected_demand = self.demand[self.ids, torch.clamp(prev_a - 1, 0, n_loc - 1)]

        # Increase capacity if depot is not visited, otherwise set to 0
        # used_capacity = torch.where(
        #     selected == 0, 0, self.used_capacity + selected_demand)
        used_capacity = (self.used_capacity + selected_demand) * (prev_a != 0).float()

        if self.visited_.dtype == torch.uint8:
            # Note: here we do not subtract one as we have to scatter so the first
            #  column allows scattering depot
            # Add one dimension since we write a single value
            visited_ = self.visited_.scatter(-1, prev_a[:, :, None], 1)
        else:
            # This works, will not set anything if prev_a -1 == -1 (depot)
            visited_ = mask_long_scatter(self.visited_, prev_a - 1)

        done = self.done | ((visited_.sum(-1) == visited_.size(-1)).squeeze(1) & (prev_a == 0).squeeze(1))

        return self._replace(
            prev_a=prev_a,
            used_capacity=used_capacity,
            visited_=visited_,
            lengths=lengths,
            last_length=last_length,
            cur_coord=cur_coord,
            done=done,
            i=self.i + 1,
        )

    def all_finished(self):
        return self.i.item() >= self.demand.size(-1) and self.done.all()

    def get_finished(self):
        return self.done

    def get_current_node(self):
        return self.prev_a.squeeze(1)

    def get_used_capacity(self):
        return self.used_capacity.squeeze(1)

    def get_mask(self):
        """
        Gets a (batch_size, n_loc + 1) mask with the feasible actions (0 = depot),
        depends on already visited and remaining capacity. 0 = feasible, 1 = infeasible
        Forbids to visit depot twice in a row, unless all nodes have been visited
        :return:
        """

        if self.visited_.dtype == torch.uint8:
            visited_loc = self.visited_[:, :, 1:]
        else:
            visited_loc = mask_long2bool(self.visited_, n=self.demand.size(-1))

        # For demand steps_dim is inserted by indexing with id, for used_capacity insert
        #  node dim for broadcasting
        exceeds_cap = (
            self.demand[self.ids, :] + self.used_capacity[:, :, None]
            > self.VEHICLE_CAPACITY
        )
        # Nodes that cannot be visited are already visited or too much demand to be
        #  served now
        mask_loc = visited_loc.to(exceeds_cap.dtype) | exceeds_cap

        # Cannot visit the depot if just visited
        mask_depot = (self.prev_a == 0)
        mask = torch.cat((mask_depot[:, :, None], mask_loc), -1).squeeze(1)
        return mask | self.done.view(-1, 1)   # mask out all actions if done

    def construct_solutions(self, actions):
        return actions
