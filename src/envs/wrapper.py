import gymnasium as gym
import numpy as np
import math

from f1tenth_gym.maps.map_manager import MapManager

class F110Wrapper(gym.Wrapper):
    """
    F110Env のラッパークラス。
    環境のインターフェースを簡略化し、追加機能を提供する。

    - step() の出力を整形
    - reset() のオプション処理を強化
    - レンダリング処理の簡略化
    - ラップタイム取得機能の追加
    """

    def __init__(self, env, map_manager: MapManager):
        super().__init__(env)
        self.ego_idx = env.ego_idx
        self.map_manager = map_manager

    def step(self, action):
        """
        環境を1ステップ進める。

        返り値:
        - observation: 観測データ
        - reward: ステップごとの報酬
        - terminated: エピソード終了フラグ
        - truncated: エピソード途中終了フラグ
        - info: 追加情報（ラップタイムなど）
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        reward = 0.0

        # spin
        if abs(obs['poses_theta'][0]) > 100.0:
            truncated = True
            
        # 1 lap 終了
        if obs['lap_counts'][0] == 2:
            terminated = True

        ## 報酬
        ### 時間経過で報酬を減らす
        reward -= self.env.timestep

        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None, index: int=0):
        """
        環境をリセットする。

        返り値:
        - observation: 初期観測データ
        - info: 追加情報（オプションで初期状態指定可能）
        """
        positions = []
        if self.map_manager.waypoints is not None:
            num_waypoints = len(self.map_manager.waypoints)
            num_agents = self.env.num_agents

            # 各エージェントに対して均等にWaypointを割り当て
            index_increment = num_waypoints / num_agents

            for i in range(num_agents):
                # 浮動小数点のインデックスを整数インデックスに変換
                waypoint_index = int(i * index_increment + index) % num_waypoints 
                next_waypoint_index = (waypoint_index + 1) % num_waypoints

                x, y = self.map_manager.waypoints[waypoint_index][:2]
                next_x, next_y = self.map_manager.waypoints[next_waypoint_index][:2]
            
                # 角度を計算（ラジアン）
                dx = next_x - x
                dy = next_y - y
                t = math.atan2(dy, dx)

                positions.append([x, y, t])

        else:
            # Waypointsが存在しない場合のデフォルトの位置
            positions = [[0, 0, 0] for _ in range(self.env.num_agents)]

        options = {
            "positions": positions
        }
        obs, info = self.env.reset(seed=seed, options=options)
        return obs, info

    def render(self, mode="human"):
        """
        環境のレンダリングを簡単に呼び出せるようにする。
        """
        return self.env.render(mode=mode)

    def get_lap_time(self):
        """
        Ego車両の現在のラップタイムを取得する。

        返り値:
        - float: 現在のラップタイム
        """
        return self.env.lap_times[self.ego_idx]

    def close(self):
        """
        環境を閉じる（リソース解放）。
        """
        self.env.close()
