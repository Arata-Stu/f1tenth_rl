import os
import numpy as np
import matplotlib.pyplot as plt
from omegaconf import OmegaConf

from src.planner.purePusuit import PurePursuitPlanner
from src.envs.envs import make_env
from f1tenth_gym.maps.map_manager import MapManager
from src.reward.reward import make_raward

# 設定ファイルのロード
yaml_path = "./config/test.yaml"
cfg = OmegaConf.load(yaml_path)
# 車両のパラメータ
param = cfg.vehicle
map_manager = MapManager(map_name=cfg.envs.map.name, map_ext=cfg.envs.map.ext) 
reward_manager = make_raward(reward_cfg=cfg.reward, map_manager=map_manager)

# 環境の作成
env = make_env(env_cfg=cfg.envs, map_manager=map_manager, param=param)
obs, info = env.reset(index=300)
num_agents = cfg.envs.num_agents

wheelbase=(0.17145+0.15875)
planner = PurePursuitPlanner(wheelbase=wheelbase, map_manager=map_manager, lookahead=0.3 ,max_reacquire=20.) 

actions = []
# メインループ
while True:
    steer, speed = planner.plan(obs, gain=0.20)
    action = [steer, speed]
    actions.append(action)
    next_obs, reward, terminated, truncated, info = env.step(np.array(actions))
    reward += reward_manager.get_reward(obs=next_obs, pre_obs=obs)
    print("reward:", reward)
    if terminated or truncated:
        print("terminated")
        break

    obs = next_obs

print("Simulation finished")
env.close()
