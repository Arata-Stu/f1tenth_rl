import numpy as np

# LiDARのスキャンデータを Occupancy Grid に変換する関数
def lidar_to_occupancy_grid(scan, grid_size=100, grid_scale=0.1, lidar_min_angle=-2.35, lidar_max_angle=2.35, max_range=20.0):
    """
    LiDARのスキャンデータ（距離のみ）を Occupancy Grid に変換する。

    Args:
        scan (list or np.array): LiDARスキャンデータ（距離のリスト）
        grid_size (int): グリッドマップのサイズ (grid_size x grid_size)
        grid_scale (float): 1マスあたりのメートル単位のスケール
        lidar_min_angle (float): LiDARの開始角度（ラジアン）
        lidar_max_angle (float): LiDARの終了角度（ラジアン）
        max_range (float): LiDARの最大検出範囲（メートル）

    Returns:
        np.array: Occupancy Grid (2D numpy array)
    """

    occupancy_grid = np.zeros((grid_size, grid_size))  # Occupancy Grid の初期化
    num_beams = len(scan)
    angles = np.linspace(lidar_min_angle, lidar_max_angle, num_beams)  # 各ビームの角度

    for r, theta in zip(scan, angles):
        if 0 < r < max_range:  # 有効範囲内のデータのみ処理
            x = r * np.cos(theta)  # LiDAR座標系のX
            y = r * np.sin(theta)  # LiDAR座標系のY

            # グリッド座標系に変換
            grid_x = int(grid_size // 2 + x / grid_scale)
            grid_y = int(grid_size // 2 + y / grid_scale)

            # グリッド範囲内なら描画
            if 0 <= grid_x < grid_size and 0 <= grid_y < grid_size:
                occupancy_grid[grid_y, grid_x] = 1  # 障害物をマーク

    return occupancy_grid
