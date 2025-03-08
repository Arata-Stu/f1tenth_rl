import numpy as np

## actionを変換
def convert_action(action, steer_range: float=0.5, speed_range: float=8.0):
    
    steer = action[0] * steer_range
    speed = action[1] * speed_range
    
    return [steer, speed]

def convert_scan(scan, max_range: float=20.0):
    scan = np.clip(scan, 0, max_range)
    scan = scan / max_range
    return scan