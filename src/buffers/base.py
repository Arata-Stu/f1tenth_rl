from typing import Union
import numpy as np
import torch
from abc import ABC, abstractmethod

class ReplayBufferBase(ABC):
    def __init__(self, size: int, state_z_dim: int, state_vec_dim: int, action_dim: int):
        self.size = size
        self.state_img_dim = state_z_dim
        self.state_vec_dim = state_vec_dim
        self.action_dim = action_dim
        
        self.buffer = {
            "state_z": np.zeros((size, state_z_dim), dtype=np.float32),
            "state_vec": np.zeros((size, state_vec_dim), dtype=np.float32),
            "action": np.zeros((size, action_dim), dtype=np.float32),
            "reward": np.zeros((size, 1), dtype=np.float32),
            "next_state_z": np.zeros((size, state_z_dim), dtype=np.float32),
            "next_state_vec": np.zeros((size, state_vec_dim), dtype=np.float32),
            "done": np.zeros((size, 1), dtype=np.bool_)
        }
        
        self.position = 0
        self.full = False
    
    @abstractmethod
    def add(self, state_img: torch.Tensor, state_vec: torch.Tensor, action: torch.Tensor, reward: float, next_state_img: torch.Tensor, next_state_vec: torch.Tensor, done: bool):
        pass
    
    def sample(self, batch_size: int) -> dict:
        max_idx = self.size if self.full else self.position
        indices = np.random.choice(max_idx, batch_size, replace=False)
        return {key: self.buffer[key][indices] for key in self.buffer}
    
    def __len__(self) -> int:
        return self.size if self.full else self.position
    
    def _to_numpy(self, data: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        if isinstance(data, torch.Tensor):
            return data.detach().cpu().numpy()
        return data
