import os
import traceback
import hydra
import numpy as np
import torch
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
        top_models = []
        
        try:
            for episode in range(num_episodes):
                obs, _ = self.env.reset(index=0)
                episode_reward = 0
                
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

                    if len(self.buffer) >= self.batch_size:
                        update_info = self.agent.update(self.buffer, batch_size=self.batch_size)
                        global_step = episode * num_steps + step

                        for key, value in update_info.items():
                            self.writer.add_scalar(f"Actor/Loss/{key}", value, global_step)
                            
                    if terminated or truncated:
                        print(f"Episode {episode} finished after {step} steps.")
                        break
                    obs = next_obs
                
                if len(top_models) < 3 or episode_reward > min(top_models, key=lambda x: x[1])[1]:
                    if len(top_models) >= 3:
                        min_model = min(top_models, key=lambda x: x[1])
                        os.remove(f"{self.save_ckpt_dir}/best_{min_model[1]:.2f}_ep_{min_model[0]}.pt")
                        top_models.remove(min_model)
                    
                    model_path = f"{self.save_ckpt_dir}/best_{episode_reward:.2f}_ep_{episode}.pt"
                    self.agent.save(model_path, episode)
                    top_models.append((episode, episode_reward))
                
                print(f"Episode {episode}: Reward = {episode_reward:.2f}")
        
        except Exception as e:
            print(f"An error occurred: {e}")
            traceback.print_exc()
        finally:
            self.writer.close()
            print("Cleaned up resources.")
    
    def evaluate(self, num_eval_episodes=10, model_path=None):
        """ 評価フェーズ """
        if model_path:
            self.agent.load(model_path)
            print(f"Loaded model from {model_path}")
        
        total_reward = 0
        for episode in range(num_eval_episodes):
            obs, _ = self.env.reset(index=0)
            episode_reward = 0
            print(f"Evaluation Episode {episode} started.")
            
            while True:
                scan_tensor = torch.tensor(self.convert_scan(obs["scans"][0]), dtype=torch.float32).to(self.device).unsqueeze(0)
                action = self.convert_action(self.agent.select_action(scans=scan_tensor, vehicle_info=None, evaluate=True))
                next_obs, reward, terminated, truncated, _ = self.env.step(np.array([action]))
                episode_reward += reward
                
                if terminated or truncated:
                    print(f"Evaluation Episode {episode} finished. Reward: {episode_reward:.2f}")
                    break
                obs = next_obs
            
            total_reward += episode_reward
            self.writer.add_scalar("Evaluation/Reward", episode_reward, episode)
        
        avg_reward = total_reward / num_eval_episodes
        print(f"Average Evaluation Reward: {avg_reward:.2f}")
        return avg_reward


@hydra.main(config_path="config", config_name="train", version_base="1.2")
def main(config: DictConfig):
    """ メイン関数 """
    trainer = Trainer(config)
    
    if config.get("mode", "train") == "train":
        trainer.train()
    elif config.get("mode") == "evaluate":
        model_path = config.get("eval_model_path", None)
        trainer.evaluate(num_eval_episodes=config.get("num_eval_episodes", 10), model_path=model_path)
    else:
        print("Invalid mode. Use 'train' or 'evaluate'.")

if __name__ == "__main__":
    main()