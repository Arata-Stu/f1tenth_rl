import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from src.models.Actor.actor import ActorTD3
from src.models.Critic.critic import Critic
from .base import BaseAgent


class TD3Agent(BaseAgent):
    def __init__(self, 
                 state_z_dim: int,
                 state_vec_dim: int,
                 action_dim: int,
                 actor_lr: float=3e-4,
                 critic_lr: float=3e-4,
                 gamma: float=0.99,
                 tau: float=0.005, 
                 hidden_dim: float=256,
                 policy_noise: float=0.2,
                 noise_clip: float=0.5,
                 policy_delay: float=2,
                 ckpt_path: str=None):
        """
        TD3エージェントの初期化
        - `policy_noise`: ターゲットポリシーのノイズ
        - `noise_clip`: ターゲットノイズのクリッピング
        - `policy_delay`: Critic を `policy_delay` 回更新した後に Actor を1回更新
        """
        super().__init__(state_z_dim, state_vec_dim, action_dim, gamma, tau, actor_lr, critic_lr)
        
        self.actor = ActorTD3(self.state_dim, action_dim, hidden_dim).to(self.device)
        self.actor_target = ActorTD3(self.state_dim, action_dim, hidden_dim).to(self.device)
        self.critic = Critic(self.state_dim, action_dim, hidden_dim).to(self.device)
        self.critic_target = Critic(self.state_dim, action_dim, hidden_dim).to(self.device)

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay

        self.total_iterations = 0  # Actorの更新を制御するためにカウント

        if ckpt_path:
            self.load(ckpt_path)

    def select_action(self, state_z: torch.Tensor, state_vec: torch.Tensor, evaluate: bool=False):
        """
        TD3のアクション選択
        - `evaluate=True` の場合、確定的なアクション（探索なし）
        - `evaluate=False` の場合、ノイズを加えて探索
        """

        state = torch.cat([state_z, state_vec], dim=-1).to(self.device)
        with torch.no_grad():
            action = self.actor(state)
        action = action.cpu().numpy()[0]

        if not evaluate:
            action += np.random.normal(0, self.policy_noise, size=action.shape)
            action = np.clip(action, -1, 1)  # アクション範囲を制限

        return action

    def update(self, buffer, batch_size: int=64):
        """
        TD3の学習ステップ
        - Critic を毎回更新
        - `policy_delay` 回に1回だけ Actor を更新
        - ターゲットポリシースムージングを適用
        """
        self.total_iterations += 1

        sample = buffer.sample(batch_size)
        state_z = torch.FloatTensor(sample["state_z"]).to(self.device)
        state_vec = torch.FloatTensor(sample["state_vec"]).to(self.device)
        action = torch.FloatTensor(sample["action"]).to(self.device)
        reward = torch.FloatTensor(sample["reward"]).to(self.device)
        next_state_z = torch.FloatTensor(sample["next_state_z"]).to(self.device)
        next_state_vec = torch.FloatTensor(sample["next_state_vec"]).to(self.device)
        done = torch.FloatTensor(sample["done"]).to(self.device)

        # ターゲットアクションの計算（ノイズを加える）
        next_state = torch.cat([next_state_z, next_state_vec], dim=-1)
        with torch.no_grad():
            noise = torch.clamp(
                torch.randn_like(action) * self.policy_noise,
                -self.noise_clip,
                self.noise_clip
            )
            next_action = torch.clamp(self.actor_target(next_state) + noise, -1, 1)

            target_q1, target_q2 = self.critic_target(next_state, next_action)
            target_q = torch.min(target_q1, target_q2)
            target_q = reward + (1 - done) * self.gamma * target_q

        # 現在の Q 値を計算
        state = torch.cat([state_z, state_vec], dim=-1)
        current_q1, current_q2 = self.critic(state, action)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = None  # 初期化

        # Actor の更新（policy_delay 回に1回）
        if self.total_iterations % self.policy_delay == 0:
            action_new = self.actor(state)
            q1_new, _ = self.critic(state, action_new)
            actor_loss = -q1_new.mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # ターゲットネットワークのソフト更新
            for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

            for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item() if actor_loss is not None else 0.0,  # None の場合は 0.0 を返す
        }


