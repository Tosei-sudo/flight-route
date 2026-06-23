#!/usr/bin/env python3
"""
Flight Route UI サーバー  —  標準ライブラリのみで実装

Usage:
    python ui/server.py           # デフォルト: http://localhost:8765
    python ui/server.py 9000      # ポート指定
"""

import json
import os
import sys
import threading
import traceback
import uuid
from http.server import HTTPServer, BaseHTTPRequestHandler

# プロジェクトルートを sys.path に追加
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

import numpy as np

# ── ウェイポイント状態 ─────────────────────────────────────────────────────────
_wp_lock = threading.Lock()
_waypoints: list[dict] = []


def _load_defaults() -> list[dict]:
    from config import GEO_WAYPOINTS
    labels = (['出発点']
              + [f'経由点{i}' for i in range(1, len(GEO_WAYPOINTS) - 1)]
              + ['目的地'])
    result = []
    for i, entry in enumerate(GEO_WAYPOINTS):
        lat, lon, alt = entry(0.0) if callable(entry) else entry
        result.append({"lat": lat, "lon": lon, "alt": float(alt), "label": labels[i]})
    return result


_waypoints = _load_defaults()

# ── シミュレーションジョブ ─────────────────────────────────────────────────────
_job_lock = threading.Lock()
_jobs: dict = {}          # job_id → {status, result|message}
_sim_gate = threading.Semaphore(1)  # 同時実行 1 本に制限


def _simulate_bg(job_id: str, wps: list) -> None:
    """バックグラウンドスレッドでシミュレーションを実行する。"""
    import config as cfg
    import geo as geo_mod
    from terrain import terrain_height_at
    from simulator import Simulator, SimParams

    geo_wps = [(wp["lat"], wp["lon"], float(wp.get("alt", 0))) for wp in wps]

    orig_cfg = cfg.GEO_WAYPOINTS
    orig_geo = geo_mod.GEO_WAYPOINTS

    with _sim_gate:
        try:
            # 座標原点を1番WPに合わせる
            cfg.GEO_WAYPOINTS = geo_wps
            geo_mod.GEO_WAYPOINTS = geo_wps

            resolvers = []
            for lat, lon, alt_agl in geo_wps:
                pos = geo_mod.geo_to_local_pt(lat, lon, alt_agl)
                pos[2] += terrain_height_at(lat, lon)
                resolvers.append(pos)

            sim = Simulator(SimParams(nav_mode='gps', use_global_planner=False))
            hist = sim.run(resolvers)

        except Exception:
            with _job_lock:
                _jobs[job_id] = {"status": "error", "message": traceback.format_exc()}
            return
        finally:
            cfg.GEO_WAYPOINTS = orig_cfg
            geo_mod.GEO_WAYPOINTS = orig_geo

    # ── 結果変換（間引いて最大 2000 点）────────────────────────────────────────
    pos_arr = hist['pos']
    n = len(pos_arr)
    step = max(1, n // 2000)
    idx = list(range(0, n, step))
    if idx[-1] != n - 1:
        idx.append(n - 1)

    traj = []
    for i in idx:
        lat_i, lon_i = geo_mod.local_to_geo(pos_arr[i])
        traj.append({
            "lat": round(float(lat_i), 6),
            "lon": round(float(lon_i), 6),
            "alt": round(float(pos_arr[i][2]), 0),
            "t":   round(float(hist['time'][i]), 1),
            "spd": round(float(hist['speed'][i]), 1),
        })

    total_dist = float(np.sum(np.linalg.norm(np.diff(pos_arr, axis=0), axis=1)))

    with _job_lock:
        _jobs[job_id] = {
            "status": "done",
            "result": {
                "trajectory":    traj,
                "hit_ground":    bool(hist["hit_ground"]),
                "total_dist_km": round(total_dist / 1000, 2),
                "total_time_s":  round(float(hist["time"][-1]), 1),
                "max_speed_ms":  round(float(hist["speed"].max()), 1),
            },
        }


# ── HTTP ハンドラ ──────────────────────────────────────────────────────────────

_STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)))

_MIME = {
    '.html': 'text/html; charset=utf-8',
    '.js':   'application/javascript',
    '.css':  'text/css',
    '.json': 'application/json',
    '.png':  'image/png',
    '.ico':  'image/x-icon',
}


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # 標準出力へのアクセスログを抑制

    # ── ヘルパー ──────────────────────────────────────────────────────────────

    def _cors(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')

    def _send_json(self, code: int, data):
        body = json.dumps(data, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self._cors()
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: str):
        ext = os.path.splitext(path)[1]
        mime = _MIME.get(ext, 'application/octet-stream')
        try:
            with open(path, 'rb') as f:
                body = f.read()
            self.send_response(200)
            self.send_header('Content-Type', mime)
            self.send_header('Content-Length', str(len(body)))
            self._cors()
            self.end_headers()
            self.wfile.write(body)
        except FileNotFoundError:
            self._send_json(404, {"error": f"not found: {path}"})

    def _read_json(self):
        n = int(self.headers.get('Content-Length', 0))
        return json.loads(self.rfile.read(n)) if n > 0 else {}

    # ── ルーティング ──────────────────────────────────────────────────────────

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_GET(self):
        path = self.path.split('?')[0].rstrip('/')

        if path in ('', '/'):
            self._send_file(os.path.join(_STATIC_DIR, 'index.html'))

        elif path == '/api/ping':
            self._send_json(200, {"ok": True})

        elif path == '/api/waypoints':
            with _wp_lock:
                self._send_json(200, list(_waypoints))

        elif path.startswith('/api/simulate/'):
            job_id = path[len('/api/simulate/'):]
            with _job_lock:
                job = dict(_jobs.get(job_id, {"status": "not_found"}))
            self._send_json(200 if job.get("status") != "not_found" else 404, job)

        else:
            self._send_json(404, {"error": "not found"})

    def do_POST(self):
        path = self.path.split('?')[0].rstrip('/')

        if path == '/api/waypoints':
            try:
                data = self._read_json()
                if not isinstance(data, list):
                    raise ValueError("list が必要です")
                with _wp_lock:
                    global _waypoints
                    _waypoints = data
                self._send_json(200, {"ok": True, "count": len(data)})
            except Exception as e:
                self._send_json(400, {"error": str(e)})

        elif path == '/api/simulate':
            try:
                data = self._read_json()
                wps = data.get("waypoints")
                if wps is None:
                    with _wp_lock:
                        wps = list(_waypoints)
                if len(wps) < 2:
                    self._send_json(400, {"error": "WP が 2 点以上必要です"})
                    return
                job_id = uuid.uuid4().hex[:8]
                with _job_lock:
                    _jobs[job_id] = {"status": "running"}
                threading.Thread(
                    target=_simulate_bg, args=(job_id, wps), daemon=True
                ).start()
                self._send_json(202, {"job_id": job_id})
            except Exception as e:
                self._send_json(400, {"error": str(e)})

        else:
            self._send_json(404, {"error": "not found"})


# ── エントリポイント ───────────────────────────────────────────────────────────

def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    server = HTTPServer(('localhost', port), _Handler)
    print(f"✈  Flight Route UI  →  http://localhost:{port}")
    print("   Ctrl+C で停止")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n停止しました")


if __name__ == '__main__':
    main()
