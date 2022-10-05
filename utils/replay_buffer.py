import torch
import numpy as np


class ReplayBuffer:
    def __init__(self, capacity, obs_shape, act_shape, device):
        self.capacity = capacity
        self.device = device

        self.obss = torch.empty((capacity, *obs_shape))
        # self.next_obss = torch.empty((capacity, *obs_shape))
        self.rewards = torch.empty((capacity, 1))
        self.actions = torch.empty((capacity, *act_shape))
        # self.dones = torch.empty((capacity, 1))

        self.idx = 0
        self.full = False

        self.bins = []

    def __len__(self):
        return self.capacity if self.full else self.idx

    def add(self, obs, action, reward):  # , next_obs, done):
        self.obss[self.idx] = obs
        # self.next_obss[self.idx] = next_obs
        self.rewards[self.idx] = reward
        self.actions[self.idx] = action
        # self.dones[self.idx] = done

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def sample(self, batch_size):

        idxs = np.random.randint(
            0,
            self.capacity - 1
            if self.full
            else self.idx
            - 1,  # in the case of not sampling the value-nets' (s,a) from the buffer
            size=batch_size,
        )
        obss = self.obss[idxs].to(self.device)
        actions = self.actions[idxs].to(self.device)
        rewards = self.rewards[idxs].to(self.device)
        # next_obss = self.next_obss[idxs].to(self.device)
        # dones = self.dones[idxs].to(self.device)

        return obss, actions, rewards  # , next_obss, dones

    # def get_bin(self, reward_1, reward_2):
    #     ### The segmentation may be further improved here
    #     return reward_1 == reward_2
