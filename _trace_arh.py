"""ARHフェーズの軌道トレース"""
import csv, math, sys, numpy as np, logging
logging.disable(logging.CRITICAL)

sys.path.insert(0, '.')
import config
config.NAV_MODE = 'gps'  # nav ノイズなし

from geo import GEO_WAYPOINTS, geo_to_local_pt
from terrain import terrain_height_at
from simulator import simulate

def _wrap(f):
    def _resolve(t):
        lat, lon, alt = f(t)
        p = geo_to_local_pt(lat, lon, alt)
        p[2] += terrain_height_at(lat, lon)
        return p
    return _resolve

wps = []
for e in GEO_WAYPOINTS:
    if callable(e):
        wps.append(_wrap(e))
    else:
        la, lo, al = e
        p = geo_to_local_pt(la, lo, al)
        p[2] += terrain_height_at(la, lo)
        wps.append(p)

hist = simulate(wps)
pos = hist['pos']
vel_arr = hist['vel'] if 'vel' in hist else None
times = hist['time']
phases = hist['phase']

print(f"hit_ground={hist['hit_ground']}  t_end={times[-1]:.1f}s")
print()
print("  t(s)   x(m)    y(m)   z(m)    vx      vy    vz    phase")

for i, t in enumerate(times):
    if phases[i] == 'terminal' or (i > 0 and phases[i-1] == 'cruise' and phases[i] == 'terminal'):
        x, y, z = pos[i]
        if vel_arr is not None:
            vx, vy, vz = vel_arr[i]
        else:
            vx, vy, vz = 0, 0, 0
        print(f"{t:7.1f}  {x:7.0f}  {y:7.0f}  {z:4.0f}   {vx:7.1f}  {vy:6.1f}  {vz:5.1f}  {phases[i]}")

# 最終位置と船の位置
t_end = float(times[-1])
lat0, lon0 = 35.987488, 135.006355
olat, olon = 34.49480, 135.30278
R = 6371000
c0 = np.cos(np.radians(lat0))
lon_s = lon0 + (t_end * 100) / (R * c0) * (180 / np.pi)
sx = R * np.cos(np.radians(lat0)) * (lon_s - olon) * np.pi / 180
sy = R * (lat0 - olat) * np.pi / 180

vx, vy, vz = pos[-1]

# 正しいENU座標でシップ位置を計算（geo_to_local_pt 経由 = シミュレータと同じ系）
ship_enu = wps[-1](t_end)
sx_true, sy_true = ship_enu[0], ship_enu[1]
miss_true = np.sqrt((sx_true - vx)**2 + (sy_true - vy)**2)
print()
print(f"vehicle    =({vx:.0f},{vy:.0f},{vz:.0f})")
print(f"ship (ENU) =({sx_true:.0f},{sy_true:.0f})  [geo_to_local_pt 系]")
print(f"ship (bad) =({sx:.0f},{sy:.0f})  [cos(lat_ship) で誤計算]")
print(f"miss (true)={miss_true:.1f}m  (dx={sx_true-vx:.0f}  dy={sy_true-vy:.0f})")
print(f"miss (bad) ={np.sqrt((sx-vx)**2+(sy-vy)**2):.1f}m  (dx={sx-vx:.0f}  dy={sy-vy:.0f})")
