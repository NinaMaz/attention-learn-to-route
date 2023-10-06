import torch
from torch import nn
import torch.nn.functional as F
from torch import Tensor, LongTensor
from math import sqrt, log, log2


class TimeEmbedding(nn.Module):
    # https://github.com/w86763777/pytorch-ddpm/blob/master/model.py
    def __init__(self, T, d_model, dim):
        assert d_model % 2 == 0
        super().__init__()
        emb = torch.arange(0, d_model, step=2) / d_model * log(10000)
        emb = torch.exp(-emb)
        pos = torch.arange(T).float()
        emb = pos[:, None] * emb[None, :]
        assert list(emb.shape) == [T, d_model // 2]
        emb = torch.stack([torch.sin(emb), torch.cos(emb)], dim=-1)
        assert list(emb.shape) == [T, d_model // 2, 2]
        emb = emb.view(T, d_model)

        self.timembedding = nn.Sequential(
            nn.Embedding.from_pretrained(emb),
            nn.Linear(d_model, dim),
            nn.LeakyReLU(),
            nn.Linear(dim, dim),
        )
        self.initialize()

    def initialize(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, t):
        emb = self.timembedding(t)
        return emb


class AgentGNN(nn.Module):
    def __init__(self,
                 input_dim,
                 hidden_dim,
                 out_dim,
                 dropout,
                 num_steps,
                 num_agents,
                 node_readout,
                 visited_decay = 0.9,
                 num_pos_attention_heads = 1,
                 bptt_steps = None,
                 gate = True,
                 temp = 2.0/3.0
                 ):
        super().__init__()

        self.input_dim = input_dim
        self.dim = hidden_dim
        self.out_dim = out_dim
        self.dropout = dropout
        self.num_steps = num_steps
        self.num_agents = num_agents
        self.node_readout = node_readout
        self.visited_decay = visited_decay
        self.num_pos_attention_heads = num_pos_attention_heads
        self.bptt_steps = bptt_steps if bptt_steps is not None else num_steps
        self.gate = gate
        self.temp = temp

        self.activation = nn.LeakyReLU(negative_slope=0.01)
        extra_global_dim = self.dim
        mlp_width_mult = 2
        attn_width_mult = 1

        if isinstance(self.num_steps, (tuple, list)):
            self.min_steps, self.max_steps = self.num_steps
            self.num_steps = self.min_steps
        else:
            self.min_steps, self.max_steps  = self.num_steps, self.num_steps

        if isinstance(self.temp, (tuple, list)):
            self.min_temp, self.max_temp = self.temp
        else:
            self.min_temp, self.max_temp  = self.temp, self.temp

        MAX_POSSIBLE_STEPS = 200
        assert MAX_POSSIBLE_STEPS >= self.max_steps
        self.time_emb = TimeEmbedding(MAX_POSSIBLE_STEPS + 1, self.dim, self.dim * mlp_width_mult)
        if input_dim != hidden_dim:
            self.input_proj = nn.Sequential(nn.Linear(input_dim, self.dim * 2), self.activation,
                                            nn.Linear(self.dim * 2, self.dim))
        else:
            self.input_proj = nn.Identity()
        self.out_proj = nn.Linear(self.dim, self.out_dim)
        # Node and agents states
        self.node_mem_init = torch.nn.Parameter(torch.zeros(self.dim, requires_grad=True))
        self.agent_emb = nn.Embedding(self.num_agents, self.dim)
        # self.agent_emb = nn.Linear(self.input_dim, self.dim)

        # Node and agent update
        self.message_val = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, self.dim),
            nn.ReLU()
        )
        self.conv_mlp = nn.Sequential(
            nn.LayerNorm(self.dim * 2 + input_dim),
            nn.Linear(self.dim * 2 + input_dim, self.dim * 2 * mlp_width_mult),
            self.activation,
            nn.Dropout(self.dropout),
            nn.Linear(self.dim * 2 * mlp_width_mult, self.dim * 2) if self.gate else nn.Linear(self.dim * 2 * mlp_width_mult, self.dim),
            nn.Dropout(self.dropout)
        )
        self.node_mlp = nn.Sequential(
            nn.LayerNorm(self.dim * 2 + extra_global_dim + input_dim),
            nn.Linear(self.dim * 2 + extra_global_dim + input_dim, self.dim * 2 * mlp_width_mult),
            self.activation,
            nn.Dropout(self.dropout),
            nn.Linear(self.dim * 2 * mlp_width_mult, self.dim * 2) if self.gate else nn.Linear(self.dim * 2 * mlp_width_mult, self.dim),
            nn.Dropout(self.dropout)
        )
        self.agent_mlp = nn.Sequential(
            nn.LayerNorm(self.dim * 2 + extra_global_dim),
            nn.Linear(self.dim * 2 + extra_global_dim, self.dim * 2 * mlp_width_mult),
            self.activation,
            nn.Dropout(self.dropout),
            nn.Linear(self.dim * 2 * mlp_width_mult, self.dim * 2) if self.gate else nn.Linear(self.dim * 2 * mlp_width_mult, self.dim),
            nn.Dropout(self.dropout)
        )
        self.global_agent_pool_mlp = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, self.dim * mlp_width_mult), self.activation,
            nn.Dropout(self.dropout),
            nn.Linear(self.dim * mlp_width_mult, self.dim),
            nn.Dropout(self.dropout)
        )
        self.step_readout_mlp = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, self.dim), self.activation,
            nn.Dropout(self.dropout)
        )
        self.graph_readout = nn.Linear(2 * self.dim, self.out_dim)

        # Have learnable global BSEU [back, stay, explored, unexplored] params
        self.back_param = nn.Parameter(torch.tensor([0.0], requires_grad=True))
        self.stay_param = nn.Parameter(torch.tensor([-1.0], requires_grad=True))
        self.explored_param = nn.Parameter(torch.tensor([0.0], requires_grad=True))
        self.unexplored_param = nn.Parameter(torch.tensor([5.0], requires_grad=True))

        # Time emb projections
        self.node_mlp_time = nn.Sequential(self.activation,
                                           nn.Linear(self.dim * mlp_width_mult, self.dim * 2 + extra_global_dim + input_dim))
        self.agent_mlp_time = nn.Sequential(self.activation,
                                            nn.Linear(self.dim * mlp_width_mult, self.dim * 2 + extra_global_dim))
        self.step_readout_mlp_time = nn.Sequential(self.activation,
                                                   nn.Linear(self.dim * mlp_width_mult, self.dim))
        self.conv_mlp_time = nn.Sequential(self.activation,
                                           nn.Linear(self.dim * mlp_width_mult, self.dim * 2 + input_dim))
        self.global_agent_pool_mlp_time = nn.Sequential(self.activation,
                                                        nn.Linear(self.dim * mlp_width_mult, self.dim))

        # Agent jump
        # self.key = nn.Sequential(nn.LayerNorm(self.dim * 2 + input_dim * 2),
        #                          nn.Linear(self.dim * 2 + input_dim * 2,
        #                                    self.dim * attn_width_mult * self.num_pos_attention_heads), nn.Identity())
        self.key1 = nn.Sequential(nn.LayerNorm(self.dim + input_dim),
                                 nn.Linear(self.dim + input_dim,
                                           self.dim * attn_width_mult * self.num_pos_attention_heads), nn.Identity())
        self.key2 = nn.Sequential(nn.LayerNorm(self.dim + input_dim),
                                 nn.Linear(self.dim + input_dim,
                                           self.dim * attn_width_mult * self.num_pos_attention_heads), nn.Identity())
        self.query = nn.Sequential(nn.LayerNorm(self.dim),
                                   nn.Linear(self.dim, self.dim * attn_width_mult * self.num_pos_attention_heads))
        self.attn_lin = nn.Sequential(nn.Linear(self.num_pos_attention_heads, 1))

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if isinstance(self.activation, nn.LeakyReLU):
                    nn.init.kaiming_uniform_(m.weight, a=self.activation.negative_slope, nonlinearity='leaky_relu')
                else:
                    raise NotImplementedError
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                m.reset_parameters()
            elif isinstance(m, nn.Embedding):
                m.reset_parameters()

    def forward(self, node_emb: torch.Tensor, mask: torch.Tensor = None,  # mask indicates valid nodes
                num_steps: int = None, start_pos: torch.Tensor = None, return_agent_paths=False):
        # x.shape = [batch_size, num_nodes, num_features], mask.shape = [batch_size, num_nodes]
        batch_size, num_nodes, input_dim = node_emb.size()
        num_steps = num_steps if num_steps is not None else self.num_steps

        # time_emb = self.time_emb(torch.zeros(1, device=node_emb.device, dtype=torch.long))
        init_node_emb = node_emb.clone()
        node_emb = self.input_proj(node_emb)  # [batch_size, num_nodes, dim]
        # agent_emb = self.agent_emb(torch.arange(self.num_agents, device=node_emb.device)).unsqueeze(0).expand(batch_size, -1, -1)  # [batch_size, n_agents, dim]
        if mask is not None:
            node_emb = node_emb * mask.unsqueeze(-1)

        # set initial positions randomly or in start_pos
        if start_pos is not None:
            agent_pos = start_pos.view(-1, 1, 1).expand(-1, self.num_agents, 1)  # [batch_size, n_agents, 1]
        else:
            if mask is not None:
                agent_pos = (mask.to(dtype=torch.float) + 1e-16).multinomial(self.num_agents, replacement=True).unsqueeze(-1)  # [batch_size, n_agents, 1]
            else:
                agent_pos = torch.randint(num_nodes, (batch_size, self.num_agents, 1), device=node_emb.device)  # [batch_size, n_agents, 1]
        agent_node_attn_value = torch.ones(agent_pos.size(), dtype=torch.float, device=agent_pos.device)
        # agent_emb = self.agent_emb(init_node_emb.gather(1, agent_pos.expand(-1, -1, input_dim)))  # [batch_size, n_agents, dim]
        agent_emb = self.agent_emb(torch.arange(self.num_agents, device=node_emb.device)).expand(batch_size, -1, -1) # [batch_size, n_agents, dim]

        # Track visited nodes
        prev_agent_pos_onehot = torch.zeros(batch_size, self.num_agents, num_nodes, dtype=torch.float, device=node_emb.device) # [batch_size, n_agents, n_nodes]
        visited_nodes = prev_agent_pos_onehot.clone()  # [batch_size, n_agents, num_nodes]
        visited_nodes.scatter_(-1, agent_pos, 1.0)  # [batch_size, n_agents, n_nodes]

        # Outs
        agent_paths = []
        out = []
        agent_embs = []

        if self.training:
            for i in range((num_steps + self.bptt_steps - 1) // self.bptt_steps):
                n_steps = min(num_steps - i * self.bptt_steps, self.bptt_steps)
                node_emb, agent_emb, agent_node_attn_value, agent_pos, prev_agent_pos_onehot, visited_nodes, final_node_emb, agent_path = self.walk(
                    init_node_emb, node_emb, agent_emb, agent_node_attn_value, agent_pos, prev_agent_pos_onehot, visited_nodes, mask, n_steps)
                out.append(final_node_emb)
                agent_embs.append(agent_emb)
                agent_paths.extend(agent_path)
                node_emb = node_emb.detach()
                agent_emb = agent_emb.detach()

            out_node_emb = out[-1]  # [batch_size, n_nodes, dim]
            out = torch.cat(out, dim=0)  # [n_iters * batch_size, n_agents, dim]
            agent_embs = torch.cat(agent_embs, dim=0)  # [n_iters * batch_size, n_agents, dim]

        else:
            node_emb, agent_emb, agent_node_attn_value, agent_pos, prev_agent_pos_onehot, visited_nodes, final_node_emb, agent_path = self.walk(
                init_node_emb, node_emb, agent_emb, agent_node_attn_value, agent_pos, prev_agent_pos_onehot, visited_nodes, mask, num_steps)

            out_node_emb = final_node_emb  # [batch_size, n_nodes, dim]
            out = final_node_emb  # [batch_size, n_nodes, dim]
            agent_embs = agent_emb  # [batch_size, n_agents, dim]

        if mask is not None:
            graph_emb = self.graph_readout(
                torch.cat([
                    agent_embs.mean(dim=-2),
                    out.mean(dim=-2) / (torch.log2(mask.count_nonzero(dim=1).view(-1, 1).repeat(out.shape[0]//batch_size, 1) + 1) + 1e-6)], dim = -1)
            )
        else:
            graph_emb = self.graph_readout(
                torch.cat([
                    agent_embs.mean(dim=-2),
                    out.mean(dim=-2) / (log2(num_nodes + 1) + 1e-6)], dim=-1)
            )
        out = self.out_proj(out)

        if return_agent_paths:
            return out_node_emb, out, graph_emb, agent_paths
        else:
            return out_node_emb, out, graph_emb

    # STACK VERSION
    #     if self.training:
    #         for i in range((num_steps + self.bptt_steps - 1) // self.bptt_steps):
    #             n_steps = min(num_steps - i * self.bptt_steps, self.bptt_steps)
    #             node_emb, agent_emb, agent_node_attn_value, agent_pos, prev_agent_pos_onehot, visited_nodes, final_node_emb, agent_path = self.walk(
    #                 init_node_emb, node_emb, agent_emb, agent_node_attn_value, agent_pos, prev_agent_pos_onehot, visited_nodes, mask, n_steps)
    #             out.append(final_node_emb)
    #             agent_embs.append(agent_emb)
    #             agent_paths.extend(agent_path)
    #             node_emb = node_emb.detach()
    #             agent_emb = agent_emb.detach()
    #
    #     else:
    #         node_emb, agent_emb, agent_node_attn_value, agent_pos, prev_agent_pos_onehot, visited_nodes, final_node_emb, agent_path = self.walk(
    #             init_node_emb, node_emb, agent_emb, agent_node_attn_value, agent_pos, prev_agent_pos_onehot, visited_nodes, mask, num_steps)
    #
    #         out.append(final_node_emb)
    #         agent_embs.append(agent_emb)
    #
    #     out_node_emb = out[-1]  # [batch_size, n_nodes, dim]
    #     out = torch.stack(out, dim=0)  # [n_iters, batch_size, n_agents, dim]
    #     agent_embs = torch.stack(agent_embs, dim=0)  # [n_iters, batch_size, n_agents, dim]
    #
    #     if mask is not None:
    #         graph_emb = self.graph_readout(
    #             torch.cat([
    #                 agent_embs.mean(dim=-2),
    #                 out.mean(dim=-2) / (torch.log2(mask.count_nonzero(dim=1).view(1, -1, 1) + 1) + 1e-6)], dim = -1)
    #         )
    #     else:
    #         graph_emb = self.graph_readout(
    #             torch.cat([
    #                 agent_embs.mean(dim=-2),
    #                 out.mean(dim=-2) / (log2(num_nodes + 1) + 1e-6)], dim=-1)
    #         )
    #     out = self.out_proj(out)
    #
    #     if not self.training:
    #         out = out.squeeze(0)
    #         graph_emb = graph_emb.squeeze(0)
    #
    #     if return_agent_paths:
    #         return out_node_emb, out, graph_emb, agent_paths
    #     else:
    #         return out_node_emb, out, graph_emb


    def walk(self, init_node_emb, node_emb, agent_emb, agent_node_attn_value,
             agent_pos, prev_agent_pos_onehot, visited_nodes, mask, num_steps):
        batch_size, num_nodes, _ = node_emb.size()
        agent_paths = []
        final_node_emb = torch.zeros(batch_size, num_nodes, self.dim, device=node_emb.device)  # [batch_size, n_nodes, dim]

        for i in range(num_steps):
            agent_paths.append(agent_pos.detach())
            # Get time for current step
            time_emb = self.time_emb(torch.tensor([i], device=node_emb.device, dtype=torch.long))  # [time_emb_dim]

            # Update node embeddings
            agent_count = torch.zeros(batch_size, num_nodes, 1, dtype=torch.float, device=node_emb.device)
            agent_count.scatter_add_(1, agent_pos, torch.ones_like(agent_count)) # [batch_size, n_nodes, 1]
            agent_emb_sum = torch.zeros_like(node_emb).scatter_add_(1, agent_pos.expand(-1, -1, self.dim), agent_emb)
            agent_emb_sum = agent_emb_sum / (torch.log2(agent_count + 1) + 1e-6)  # [batch_size, n_nodes, dim]
            node_mask = (agent_count > 0)
            node_update = torch.cat([
                init_node_emb,
                node_emb,
                agent_emb_sum,  # sum of agent embeddings on each node
                agent_emb.mean(dim=1, keepdim=True).expand(-1, num_nodes, -1)  # log mean of all agent embeddings
            ], dim=-1)   # [batch_size, n_nodes, dim * 3]
            node_update = node_update + self.node_mlp_time(time_emb)
            if self.gate:
                node_update, z = self.node_mlp(node_update).chunk(2, dim=-1)
                z = torch.sigmoid(z)
                node_emb = ~node_mask * node_emb + node_mask * ((1-z) * node_emb + z * node_update)
            else:
                node_emb = node_emb + node_mask * self.node_mlp(node_update)
            del node_update, agent_emb_sum, agent_count

            # Do a convolution to get neighborhood info
            if mask is not None:
                node_emb_sum = (node_emb * mask.unsqueeze(-1)).sum(dim=1, keepdim=True) - node_emb  # [batch_size, n_nodes, dim]
                node_emb_sum = node_emb_sum / (torch.log2(mask.count_nonzero(dim=1).view(-1, 1, 1) + 1) + 1e-6)  # [batch_size, n_nodes, dim]
            else:
                node_emb_sum = node_emb.sum(dim=1, keepdim=True) - node_emb  # [batch_size, n_nodes, dim]
                node_emb_sum = node_emb_sum / (log2(num_nodes + 1) + 1e-6)  # [batch_size, n_nodes, dim]
            node_update = torch.cat([
                init_node_emb,
                node_emb,
                node_emb_sum  # log mean of neighbor embeddings
            ], dim=-1)  # [batch_size, n_nodes, dim * 2]
            node_update = node_update + self.conv_mlp_time(time_emb)
            if self.gate:
                node_update, z = self.conv_mlp(node_update).chunk(2, dim=-1)
                z = torch.sigmoid(z)
                node_emb = ~node_mask * node_emb + node_mask * ((1-z) * node_emb + z * node_update)
            else:
                node_emb = node_emb + node_mask * self.conv_mlp(node_update)
            del node_update, node_emb_sum, node_mask

            # Update agent embeddings
            node_emb_sum = node_emb.gather(1, agent_pos.expand(-1, -1, self.dim))  # [batch_size, n_agents, dim]
            agent_update = torch.cat([
                agent_emb,
                node_emb_sum * agent_node_attn_value,
                agent_emb.mean(dim=1, keepdim=True).expand(-1, self.num_agents, -1)
            ], dim=-1)  # [batch_size, n_agents, dim * 3]
            agent_update = agent_update + self.agent_mlp_time(time_emb)
            if self.gate:
                agent_update, z = self.agent_mlp(agent_update).chunk(2, dim=-1)
                z = torch.sigmoid(z)
                agent_emb = (1-z) * agent_emb + z * agent_update
            else:
                agent_emb = agent_emb + self.agent_mlp(agent_update)

            # Readout
            if self.node_readout:
                layer_out = self.step_readout_mlp(node_emb + self.step_readout_mlp_time(time_emb))
                final_node_emb += layer_out / num_steps
            else:
                final_node_emb = node_emb

            # In the first iteration just update the starting node/agent embeddings
            if i < num_steps:
                # Move agents
                Q = self.query(agent_emb).view(agent_emb.size(0), agent_emb.size(1), 1, self.num_pos_attention_heads, 1, -1)  # [batch_size, n_agents, 1, n_heads, 1, d]
                ext_node_emb = torch.cat([init_node_emb, node_emb], dim=-1)  # [batch_size, n_nodes, init_dim + dim]
                K1 = ext_node_emb.gather(1, agent_pos.expand(-1, -1, ext_node_emb.size(-1)))  # [batch_size, n_agents, init_dim + dim]
                K2 = ext_node_emb  # [batch_size, n_nodes, init_dim + dim]
                K1 = self.key1(K1).reshape(agent_emb.size(0), agent_emb.size(1), 1, self.num_pos_attention_heads, -1, 1)  # [batch_size, n_agents, 1, n_heads, d, 1]
                K2 = self.key2(K2).reshape(ext_node_emb.size(0), 1, ext_node_emb.size(1), self.num_pos_attention_heads, -1, 1)  # [batch_size, 1, n_nodes, n_heads, d, 1]
                attn_score = (Q @ (K1 + K2)).squeeze((-1, -2)) / sqrt(Q.size(-1))   # [batch_size, n_agents, n_nodes, n_heads]
                # print(attn_score.shape)
                del K1, K2, Q
                if self.num_pos_attention_heads > 1:
                    attn_score = self.attn_lin(attn_score)  # [batch_size, n_agents, n_nodes, 1]
                attn_score = attn_score.squeeze(-1)  # [batch_size, n_agents, n_nodes]

                # Fill in neighbor attention scores using the learned logits for [back, stay, explored, unexplored]
                agent_pos_onehot = torch.zeros_like(prev_agent_pos_onehot).scatter_(-1, agent_pos, 1.0)
                if i > 0:  # No previous node at first step
                    attn_score += self.back_param * prev_agent_pos_onehot  # Previous node
                attn_score += self.stay_param * agent_pos_onehot  # Current node
                attn_score += self.explored_param * visited_nodes  # Explored nodes
                attn_score += self.unexplored_param * (1 - visited_nodes)  # Unexplored nodes

                if mask is not None:
                    attn_score = torch.where(mask.unsqueeze(1), attn_score, -1e18)  # [batch_size, n_agents, n_nodes]

                attn_score = F.gumbel_softmax(attn_score, hard=True, dim=2,
                    tau=(self.temp if self.training else 1e-6))  # [batch_size, n_agents, n_nodes]

                # Get updated agent positions
                agent_node_attn_value, agent_pos = torch.max(attn_score, dim=2, keepdim=True)  # [batch_size, n_agents, 1], [batch_size, n_agents, 1]
                # agent_node_attn_value - multiply node emb with this to attach gradients when getting node agent is on
                del attn_score
                # Update tracked positions
                prev_agent_pos_onehot = agent_pos_onehot  # back: [batch_size, n_agents, n_nodes]
                visited_nodes = visited_nodes * self.visited_decay
                visited_nodes = torch.scatter(visited_nodes, -1, agent_pos, 1.0)  #[batch_size, n_agents, n_nodes]

        return node_emb, agent_emb, agent_node_attn_value, agent_pos, prev_agent_pos_onehot, visited_nodes, final_node_emb, agent_paths


if __name__ == '__main__':
    import time
    batch_size, num_nodes, num_agents, f_dim = 20, 100, 50, 32
    agent_net = AgentGNN(
        input_dim=4,
        hidden_dim=f_dim,
        out_dim=5,
        dropout=0.0,
        num_steps=12,
        num_agents=num_agents,
        node_readout=False,
        visited_decay=0.9,
        num_pos_attention_heads=1,
        bptt_steps=5
    ).to("cuda:0")
    input = torch.randn(batch_size, num_nodes, 4, device="cuda:0")
    mask = torch.rand(batch_size, num_nodes, device="cuda:0") < 0.7

    out = agent_net(input)
    # print GPU memory usage in GB
    print(f"Allocated: {torch.cuda.memory_allocated() / 1024 ** 3:.1f} GB")
    print(f"Cached: {torch.cuda.memory_reserved() / 1024 ** 3:.1f} GB")
    print(out[1].mean(), out[0].shape, out[1].shape, out[2].shape)
    input[0, :50] = 1e15
    mask[0, :50] = False
    out = agent_net(input, mask)
    # print GPU memory usage in GB
    print(f"Allocated: {torch.cuda.memory_allocated() / 1024 ** 3:.1f} GB")
    print(f"Cached: {torch.cuda.memory_reserved() / 1024 ** 3:.1f} GB")
    print(out[1].mean(), out[0].shape, out[1].shape, out[2].shape)

    with torch.autograd.profiler.profile(use_cuda=True) as prof:
        out = agent_net(input)
    print("NO MASK:", prof.total_average())
    with torch.autograd.profiler.profile(use_cuda=True) as prof:
        out = agent_net(input, mask)
    print("MASK:", prof.total_average())

    # training
    input = torch.randn(batch_size, num_nodes, 4, device="cuda:0")
    mask = torch.ones(batch_size, num_nodes, device="cuda:0", dtype=torch.bool)
    pos = torch.zeros(batch_size, device="cuda:0", dtype=torch.long)
    agent_net.eval()
    torch.random.manual_seed(0)
    out1 = agent_net(input, start_pos=pos)
    torch.random.manual_seed(0)
    out2 = agent_net(input, mask, start_pos=pos)
    for o1, o2 in zip(out1, out2):
        print(torch.allclose(o1, o2, 1e-4))


