import csv
import numpy as np


def save_csv(hist: dict, path: str) -> None:
    """シミュレーション結果をテレメトリー CSV として保存する。"""
    pos       = hist['pos']
    vel       = hist['vel']
    accel     = hist['accel']
    times     = hist['time']
    speeds    = hist['speed']
    elevation = hist['elevation']
    azimuth   = hist['azimuth']
    accel_mag = np.linalg.norm(accel, axis=1)

    header = [
        'time_s',
        'x_m', 'y_m', 'z_m',
        'vx_ms', 'vy_ms', 'vz_ms', 'speed_ms',
        'ax_ms2', 'ay_ms2', 'az_ms2', 'accel_mag_ms2',
        'elevation_deg', 'azimuth_deg',
    ]

    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i in range(len(times)):
            writer.writerow([
                f'{times[i]:.3f}',
                f'{pos[i,0]:.3f}',    f'{pos[i,1]:.3f}',    f'{pos[i,2]:.3f}',
                f'{vel[i,0]:.4f}',    f'{vel[i,1]:.4f}',    f'{vel[i,2]:.4f}',
                f'{speeds[i]:.4f}',
                f'{accel[i,0]:.4f}',  f'{accel[i,1]:.4f}',  f'{accel[i,2]:.4f}',
                f'{accel_mag[i]:.4f}',
                f'{elevation[i]:.4f}', f'{azimuth[i]:.4f}',
            ])
    print(f"保存: {path}  ({len(times)} 行)")
