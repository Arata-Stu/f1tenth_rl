import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np  

from src.models.Actor.actor import ActorSAC
from src.models.Critic.critic import Critic
from .base import BaseAgent

class SACAgent(BaseAgent):
    def __init__(self,
                 scans_dim: int,
                 vehicle_info_dim: int,
                 action_dim: int,
                 actor_lr: float=3e-4,
                 critic_lr: float=3e-4,
                 alpha_lr: float=3e-4, 
                 gamma: float=0.99,
                 tau: float=0.005,
                 hidden_dim: int=256,
                 ckpt_path: str=None):
        super().__init__(scans_dim, vehicle_info_dim, action_dim, gamma, tau, actor_lr, critic_lr)

        self.actor = ActorSAC(self.state_dim, action_dim, hidden_dim).to(self.device)
        self.critic = Critic(self.state_dim, action_dim, hidden_dim).to(self.device)
        self.critic_target = Critic(self.state_dim, action_dim, hidden_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.log_alpha = torch.tensor(0.0, requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)
        self.target_entropy = -action_dim

        if ckpt_path:
            self.load(ckpt_path)

    def select_action(self, scans: torch.Tensor, vehicle_info: torch.Tensor = None, evaluate: bool=False) -> np.ndarray:
        state = scans if vehicle_info is None else torch.cat([scans, vehicle_info], dim=-1)
        with torch.no_grad():
            if evaluate:
                _, _, action = self.actor.sample(state)
            else:
                action, _, _ = self.actor.sample(state)
        return action.cpu().numpy()[0]


    def update(self, buffer, batch_size: int=64):
        sample = buffer.sample(batch_size)
        scans = torch.FloatTensor(sample["scans"]).to(self.device)
        vehicle_info = torch.FloatTensor(sample["vehicle_info"]).to(self.device)
        action = torch.FloatTensor(sample["action"]).to(self.device)
        reward = torch.FloatTensor(sample["reward"]).to(self.device)
        next_next_scans = torch.FloatTensor(sample["next_scans"]).to(self.device)
        next_vehicle_info = torch.FloatTensor(sample["next_vehicle_info"]).to(self.device)
        done = torch.FloatTensor(sample["done"]).to(self.device)

        next_state = torch.cat([next_next_scans, next_vehicle_info], dim=-1)
        with torch.no_grad():
            next_action, next_log_prob, _ = self.actor.sample(next_state)
            target_q1, target_q2 = self.critic_target(next_state, next_action)
            target_q = torch.min(target_q1, target_q2) - torch.exp(self.log_alpha) * next_log_prob
            target_q = reward + (1 - done) * self.gamma * target_q

        state = torch.cat([scans, vehicle_info], dim=-1)
        current_q1, current_q2 = self.critic(state, action)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        action_new, log_prob, _ = self.actor.sample(state)
        q1_new, q2_new = self.critic(state, action_new)
        actor_loss = (torch.exp(self.log_alpha) * log_prob - torch.min(q1_new, q2_new)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha_loss": alpha_loss.item(),
            "alpha": torch.exp(self.log_alpha).item()
        }
