from omegaconf import DictConfig

from .progress import ProgressReward
from f1tenth_gym.maps.map_manager import MapManager

def make_raward(reward_cfg: DictConfig, map_manager: MapManager=None):
    reward_name = reward_cfg.name
    if reward_name == "progress":
        return ProgressReward(ratio=reward_cfg.ratio, map_manager=map_manager)
    else:
        raise ValueError(f"Invalid reward type: {reward_name}")