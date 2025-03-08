from omegaconf import DictConfig
from .sac import SACAgent

def get_agents(agent_cfg: DictConfig, scans_dim: int, vehicle_info_dim: int, action_dim: int):
    if agent_cfg.name == "SAC":
        return SACAgent(scans_dim=scans_dim,
                        vehicle_info_dim=vehicle_info_dim,
                        action_dim=action_dim,
                        actor_lr=agent_cfg.actor_lr,
                        critic_lr=agent_cfg.critic_lr,
                        alpha_lr=agent_cfg.alpha_lr,
                        gamma=agent_cfg.gamma,
                        tau=agent_cfg.tau,
                        hidden_dim=agent_cfg.hidden_dim,
                        ckpt_path=agent_cfg.ckpt_path)
    else:
        raise NotImplementedError(f"Unknown agent: {agent_cfg.name}")