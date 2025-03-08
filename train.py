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

@hydra.main(config_path="config", config_name="train", version_base="1.2")
def main(config: DictConfig):
    OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
    print('------ Configuration ------')
    print(OmegaConf.to_yaml(config))
    print('---------------------------')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    writer = SummaryWriter(log_dir=config.get("log_dir", "./logs"))

    # 車両のパラメータ
    param = config.vehicle
    steer_range = param.s_max
    speed_range = param.v_max
    convert_action_ = partial(convert_action, steer_range=steer_range, speed_range=speed_range)
    convert_scan_ = partial(convert_scan, max_range=30.0)

    # マップの設定
    map_cfg = config.envs.map
    map_manager = MapManager(map_name=map_cfg.name,
                             map_ext=map_cfg.ext,
                             speed=map_cfg.speed,
                             downsample=map_cfg.downsample)
    
    # 環境の作成
    env = make_env(env_cfg=config.envs, map_manager=map_manager, param=param)

    ## 報酬関数の設定
    reward_manager = make_raward(reward_cfg=config.reward, map_manager=map_manager)

    ## エージェントの設定
    scans_dim = 60
    vehicle_info_dim = 0
    action_dim = 2
    agent = get_agents(agent_cfg=config.agent, scans_dim=scans_dim, vehicle_info_dim=vehicle_info_dim, action_dim=action_dim)
    ## バッファの設定
    buffer = get_buffers(buffer_cfg=config.buffer, scans_dim=scans_dim, vehicle_info_dim=vehicle_info_dim, action_dim=action_dim)

    ## 学習設定
    num_episodes = config.num_episodes
    num_steps = config.num_steps
    batch_size = config.batch_size
    save_ckpt_dir = config.save_ckpt_dir

    try:
        for episode in range(num_episodes):
            obs, info = env.reset(index=0)
            episode_reward = 0

            print(f"Episode {episode} started.")
            for step in range(num_steps):
                scan = convert_scan_(obs["scans"][0])
                scan_tensor = torch.tensor(scan, dtype=torch.float32).to(device).unsqueeze(0)
                actions = []
                action = agent.select_action(scans=scan_tensor, vehicle_info=None)
                action = convert_action_(action=action)
                actions.append(action)
                next_obs, reward, terminated, truncated, info = env.step(np.array(actions))
                reward += reward_manager.get_reward(obs=next_obs, pre_obs=obs)
                next_scan = convert_scan_(next_obs["scans"][0])

                done = terminated or truncated

                vehicle_info = None
                next_vehicle_info = None
                buffer.add(scan, vehicle_info, action, reward, next_scan, next_vehicle_info, done)

                episode_reward += reward

                if done:
                    obs, info = env.reset()
                    print(f"Episode {episode} finished after {step} steps.")
                    break

                obs = next_obs

    except Exception as e:
        print(f"An error occurred: {e}")
        traceback.print_exc()
    finally:
        writer.close()
        print("Cleaned up resources.")

if __name__ == "__main__":
    main()
