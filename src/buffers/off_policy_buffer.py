from typing import Union
import torch
import numpy as np
from collections import deque
from .base import ReplayBufferBase

class OffPolicyBuffer(ReplayBufferBase):
    def __init__(self, size: int, scans_dim: int, vehicle_info_dim: int, action_dim: int, n_step: int=3, gamma: float=0.99):
        super().__init__(size, scans_dim, vehicle_info_dim, action_dim)
        self.n_step = n_step
        self.gamma = gamma
        self.temp_buffer = deque(maxlen=n_step)

    def add(self, scans: torch.Tensor, vehicle_info: Union[torch.Tensor, None], 
            action: torch.Tensor, reward: float, 
            next_scans: torch.Tensor, next_vehicle_info: Union[torch.Tensor, None], 
            done: bool):
        if vehicle_info is None:
            vehicle_info = torch.zeros(scans.shape[0], self.vehicle_info_dim, device=scans.device)
        if next_vehicle_info is None:
            next_vehicle_info = torch.zeros(next_scans.shape[0], self.vehicle_info_dim, device=next_scans.device)

        self.temp_buffer.append((scans, vehicle_info, action, reward, next_scans, next_vehicle_info, done))
        if len(self.temp_buffer) >= self.n_step:
            self._store_n_step_transition()
        if done:
            while self.temp_buffer:
                self._store_n_step_transition()

    def _store_n_step_transition(self):
        scans, vehicle_info, action, _, _, _, _ = self.temp_buffer[0]
        reward = 0
        discount = 1
        _, _, _, _, next_scans, next_vehicle_info, done = self.temp_buffer[-1]

        for _, _, _, r, _, _, d in self.temp_buffer:
            reward += discount * r
            discount *= self.gamma
            if d:
                break

        idx = self.position
        self.buffer["scans"][idx] = self._to_numpy(scans, self.scans_dim)
        self.buffer["action"][idx] = self._to_numpy(action, self.action_dim)
        self.buffer["reward"][idx] = reward
        self.buffer["next_scans"][idx] = self._to_numpy(next_scans, self.scans_dim)
        self.buffer["done"][idx] = done

        if self.vehicle_info_dim > 0:
            self.buffer["vehicle_info"][idx] = self._to_numpy(vehicle_info, self.vehicle_info_dim)
            self.buffer["next_vehicle_info"][idx] = self._to_numpy(next_vehicle_info, self.vehicle_info_dim)

        self.position = (self.position + 1) % self.size
        if self.position == 0:
            self.full = True

        self.temp_buffer.popleft()
