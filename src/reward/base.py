from abc import ABC

class RewardBase(ABC):
    def __init__(self, ratio: float=1.0):
        self.reward = ratio

    def get_reward(self, obs, pre_obs):
        raise NotImplementedError

    def reset(self):
        pass
