import torch
from .base import ReplayBufferBase

class OnPolicyBuffer(ReplayBufferBase):
    def __init__(self, size: int, state_z_dim: tuple, state_vec_dim: int, action_dim: int):
        super().__init__(size, state_z_dim, state_vec_dim, action_dim)
    
    def add(self, state_z: torch.Tensor, state_vec: torch.Tensor, action: torch.Tensor, reward: float, next_state_z: torch.Tensor, next_state_vec: torch.Tensor, done: bool):
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
    
    def clear(self):
        """サイズがいっぱいになったら上書きしながら維持する"""
        self.position = 0
        self.full = False
