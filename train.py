import os
import traceback
import hydra
import numpy as np
import torch
import wandb
from functools import partial
from torch.utils.tensorboard import SummaryWriter
from omegaconf import DictConfig, OmegaConf

from src.envs.envs import make_env
from f1tenth_gym.maps.map_manager import MapManager
from src.reward.reward import make_raward
from src.agents.agents import get_agents
from src.buffers.buffers import get_buffers
from src.utils.helppers import convert_action, convert_scan


class Trainer:
    def __init__(self, config):
        """ トレーナークラスの初期化 """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.writer = SummaryWriter(log_dir=config.get("log_dir", "./logs"))
        
        # WandB の初期化（ネットワーク切断時でも学習継続可能に）
        self.use_wandb = config.get("use_wandb", False)
        if self.use_wandb:
            wandb.init(
                project=config.get("wandb_project", "RL_Training"),
                name=config.get("wandb_run_name", "run"),
                mode=config.get("wandb_mode", "online"),  # オンライン or オフライン
                config=dict(config)
            )
            
        param = config.vehicle
        self.convert_action = partial(convert_action, steer_range=param.s_max, speed_range=param.v_max)
        self.convert_scan = partial(convert_scan, max_range=30.0)
        
        map_cfg = config.envs.map
        self.map_manager = MapManager(
            map_name=map_cfg.name, map_ext=map_cfg.ext, speed=map_cfg.speed, downsample=map_cfg.downsample
        )
        
        self.env = make_env(env_cfg=config.envs, map_manager=self.map_manager, param=param)
        self.reward_manager = make_raward(reward_cfg=config.reward, map_manager=self.map_manager)
        
        self.agent = get_agents(
            agent_cfg=config.agent, scans_dim=60, vehicle_info_dim=0, action_dim=2
        )
        self.buffer = get_buffers(
            buffer_cfg=config.buffer, scans_dim=60, vehicle_info_dim=0, action_dim=2
        )
        
        self.batch_size = config.batch_size
        self.save_ckpt_dir = config.save_ckpt_dir
        os.makedirs(self.save_ckpt_dir, exist_ok=True)
        
    def train(self):
        """ 学習ループ """
        num_episodes = self.config.num_episodes
        num_steps = self.config.num_steps
        eval_interval = self.config.get("eval_interval", 10)
        num_eval_episodes = self.config.get("num_eval_episodes", 5)
        top_models = []
        
        try:
            for episode in range(num_episodes):
                obs, _ = self.env.reset(index=0)
                episode_reward = 0
                step_count = 0
                
                print(f"Episode {episode} started.")
                for step in range(num_steps):
                    scan_tensor = torch.tensor(self.convert_scan(obs["scans"][0]), dtype=torch.float32).to(self.device).unsqueeze(0)
                    action = self.convert_action(self.agent.select_action(scans=scan_tensor, vehicle_info=None, evaluate=False))
                    next_obs, reward, terminated, truncated, _ = self.env.step(np.array([action]))
                    reward += self.reward_manager.get_reward(obs=next_obs, pre_obs=obs)
                    
                    self.buffer.add(
                        self.convert_scan(obs["scans"][0]), None, action, reward,
                        self.convert_scan(next_obs["scans"][0]), None, terminated or truncated
                    )
                    
                    episode_reward += reward
                    step_count += 1

                    if len(self.buffer) >= self.batch_size:
                        update_info = self.agent.update(self.buffer, batch_size=self.batch_size)
                        global_step = episode * num_steps + step

                        for key, value in update_info.items():
                            self.writer.add_scalar(f"Actor/Loss/{key}", value, global_step)
                            if self.use_wandb:
                                wandb.log({f"Actor/Loss/{key}": value, "step": global_step})
                            
                    if terminated or truncated:
                        print(f"Episode {episode} finished after {step} steps.")
                        break
                    obs = next_obs
                
                self.writer.add_scalar("Training/Episode_Reward", episode_reward, episode)
                self.writer.add_scalar("Training/Episode_Steps", step_count, episode)
                
                if self.use_wandb:
                    wandb.log({"Training/Episode_Reward": episode_reward, "Training/Episode_Steps": step_count, "step": episode})
                
                if len(top_models) < 3 or episode_reward > min(top_models, key=lambda x: x[1])[1]:
                    if len(top_models) >= 3:
                        min_model = min(top_models, key=lambda x: x[1])
                        os.remove(f"{self.save_ckpt_dir}/best_{min_model[1]:.2f}_ep_{min_model[0]}.pt")
                        top_models.remove(min_model)
                    
                    model_path = f"{self.save_ckpt_dir}/best_{episode_reward:.2f}_ep_{episode}.pt"
                    self.agent.save(model_path, episode)
                    top_models.append((episode, episode_reward))
                
                print(f"Episode {episode}: Reward = {episode_reward:.2f}, Steps = {step_count}")
                
                if episode % eval_interval == 0:
                    avg_eval_reward = self.evaluate(num_eval_episodes)
                    self.writer.add_scalar("Evaluation/Average_Reward", avg_eval_reward, episode)
                    if self.use_wandb:
                        wandb.log({"Evaluation/Average_Reward": avg_eval_reward, "step": episode})
        
        except Exception as e:
            print(f"An error occurred: {e}")
            traceback.print_exc()
        finally:
            self.writer.close()
            if self.use_wandb:
                wandb.finish()
            print("Cleaned up resources.")
