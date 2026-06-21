"""ARH 精度デバッグ: NAV_MODE='gps' で GPS ノイズを除外してミスを測定する。"""
import sys, numpy as np, logging
logging.disable(logging.CRITICAL)
sys.path.insert(0, '.')

import config
config.NAV_MODE = 'gps'   # 理想GPS: ノイズなし

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
vx, vy, vz = hist['pos'][-1]
t_end = float(hist['time'][-1])

lat0, lon0 = 35.987488, 135.006355
olat, olon = 34.49480, 135.30278
R = 6371000
c0 = np.cos(np.radians(lat0))
lon_s = lon0 + (t_end * 100) / (R * c0) * (180 / np.pi)
sx = R * np.cos(np.radians(lat0)) * (lon_s - olon) * np.pi / 180
sy = R * (lat0 - olat) * np.pi / 180

miss = np.sqrt((sx - vx)**2 + (sy - vy)**2)
print("hit_ground =", hist['hit_ground'])
print("t_end      = %.1f s" % t_end)
print("vehicle    = (%.0f, %.0f, %.0f)" % (vx, vy, vz))
print("ship       = (%.0f, %.0f)" % (sx, sy))
print("miss       = %.1f m   (dx=%.0f  dy=%.0f)" % (miss, sx-vx, sy-vy))
