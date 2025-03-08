from typing import Union
import numpy as np
import torch
from abc import ABC, abstractmethod

class ReplayBufferBase(ABC):
    def __init__(self, size: int, scans_dim: int, vehicle_info_dim: int, action_dim: int):
        self.size = size
        self.scans_dim = scans_dim
        self.vehicle_info_dim = vehicle_info_dim
        self.action_dim = action_dim

        self.buffer = {
            "scans": np.zeros((size, scans_dim), dtype=np.float32),
            "action": np.zeros((size, action_dim), dtype=np.float32),
            "reward": np.zeros((size, 1), dtype=np.float32),
            "next_scans": np.zeros((size, scans_dim), dtype=np.float32),
            "done": np.zeros((size, 1), dtype=np.bool_)
        }

        if vehicle_info_dim > 0:
            self.buffer["vehicle_info"] = np.zeros((size, vehicle_info_dim), dtype=np.float32)
            self.buffer["next_vehicle_info"] = np.zeros((size, vehicle_info_dim), dtype=np.float32)

        self.position = 0
        self.full = False

    @abstractmethod
    def add(self, scans: torch.Tensor, vehicle_info: Union[torch.Tensor, None], 
            action: torch.Tensor, reward: float, 
            next_scans: torch.Tensor, next_vehicle_info: Union[torch.Tensor, None], 
            done: bool):
        pass

    def sample(self, batch_size: int) -> dict:
        max_idx = self.size if self.full else self.position
        indices = np.random.choice(max_idx, batch_size, replace=False)
        return {key: self.buffer[key][indices] for key in self.buffer}

    def __len__(self) -> int:
        return self.size if self.full else self.position

    def _to_numpy(self, data: Union[np.ndarray, torch.Tensor, None], dim: int) -> np.ndarray:
        if data is None:
            return np.zeros((dim,), dtype=np.float32)
        if isinstance(data, torch.Tensor):
            return data.detach().cpu().numpy()
        return data