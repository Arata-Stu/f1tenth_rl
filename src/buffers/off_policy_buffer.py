import torch
import numpy as np
from collections import deque
from .base import ReplayBufferBase

class OffPolicyBuffer(ReplayBufferBase):
    def __init__(self, size: int, state_z_dim: int, state_vec_dim: int, action_dim: int, n_step: int=3, gamma: float=0.99):
        super().__init__(size, state_z_dim, state_vec_dim, action_dim)
        self.n_step = n_step
        self.gamma = gamma
        self.temp_buffer = deque(maxlen=n_step)
    
    def add(self, state_z: torch.Tensor, state_vec: torch.Tensor, action: torch.Tensor, reward: float, next_state_z: torch.Tensor, next_state_vec: torch.Tensor, done: bool):
        self.temp_buffer.append((state_z, state_vec, action, reward, next_state_z, next_state_vec, done))
        if len(self.temp_buffer) >= self.n_step:
            self._store_n_step_transition()
        if done:
            while self.temp_buffer:
                self._store_n_step_transition()
    
    def _store_n_step_transition(self):
        state_z, state_vec, action, _, _, _, _ = self.temp_buffer[0]
        reward = 0
        discount = 1
        _, _, _, _, next_state_z, next_state_vec, done = self.temp_buffer[-1]

        for _, _, _, r, _, _, d in self.temp_buffer:
            reward += discount * r
            discount *= self.gamma
            if d:
                break

        idx = self.position
        self.buffer["state_z"][idx] = self._to_numpy(state_z)
        self.buffer["state_vec"][idx] = self._to_numpy(state_vec)
        self.buffer["action"][idx] = self._to_numpy(action)
        self.buffer["reward"][idx] = reward
        self.buffer["next_state_z"][idx] = self._to_numpy(next_state_z)
        self.buffer["next_state_vec"][idx] = self._to_numpy(next_state_vec)
        self.buffer["done"][idx] = done

        self.position = (self.position + 1) % self.size
        if self.position == 0:
            self.full = True

        self.temp_buffer.popleft()
