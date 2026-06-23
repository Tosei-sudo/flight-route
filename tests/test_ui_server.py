"""
ui/server.py のテスト

テスト用ポートで HTTPServer を起動し、各 API エンドポイントの
レスポンスを検証する。シミュレーション実行は mock で差し替え。
"""

import json
import threading
import time
import urllib.request
import urllib.error
from http.server import HTTPServer
from unittest.mock import patch, MagicMock

import pytest
import numpy as np

# server モジュールを直接インポート
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ui'))
import server as ui_server


# ━━━━━━━━━━━━━━━━━━━━━  テストサーバー フィクスチャ  ━━━━━━━━━━━━━━━━━━━━━

def _free_port() -> int:
    import socket
    with socket.socket() as s:
        s.bind(('localhost', 0))
        return s.getsockname()[1]


@pytest.fixture(scope='module')
def server_url():
    """テスト用 HTTPServer を起動し URL を返す。モジュール全体で共有。"""
    port = _free_port()
    httpd = HTTPServer(('localhost', port), ui_server._Handler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    time.sleep(0.05)       # 起動待ち
    yield f'http://localhost:{port}'
    httpd.shutdown()


def _get(url: str) -> tuple[int, dict]:
    req = urllib.request.Request(url)
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def _post(url: str, body) -> tuple[int, dict]:
    data = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data,
                                 headers={'Content-Type': 'application/json'},
                                 method='POST')
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


# ━━━━━━━━━━━━━━━━━━━━━  _load_defaults  ━━━━━━━━━━━━━━━━━━━━━

class TestLoadDefaults:
    def test_returns_list(self):
        result = ui_server._load_defaults()
        assert isinstance(result, list)

    def test_at_least_two_waypoints(self):
        result = ui_server._load_defaults()
        assert len(result) >= 2

    def test_first_wp_has_required_keys(self):
        wp = ui_server._load_defaults()[0]
        for key in ('lat', 'lon', 'alt', 'label'):
            assert key in wp, f"キー '{key}' がない"

    def test_lat_lon_are_floats(self):
        wp = ui_server._load_defaults()[0]
        assert isinstance(wp['lat'], float)
        assert isinstance(wp['lon'], float)

    def test_first_label_is_departure(self):
        wp = ui_server._load_defaults()[0]
        assert wp['label'] == '出発点'

    def test_last_label_is_destination(self):
        wp = ui_server._load_defaults()[-1]
        assert wp['label'] == '目的地'


# ━━━━━━━━━━━━━━━━━━━━━  GET /api/ping  ━━━━━━━━━━━━━━━━━━━━━

class TestPing:
    def test_status_200(self, server_url):
        status, _ = _get(f'{server_url}/api/ping')
        assert status == 200

    def test_returns_ok_true(self, server_url):
        _, body = _get(f'{server_url}/api/ping')
        assert body.get('ok') is True


# ━━━━━━━━━━━━━━━━━━━━━  GET /api/waypoints  ━━━━━━━━━━━━━━━━━━━━━

class TestGetWaypoints:
    def test_status_200(self, server_url):
        status, _ = _get(f'{server_url}/api/waypoints')
        assert status == 200

    def test_returns_list(self, server_url):
        _, body = _get(f'{server_url}/api/waypoints')
        assert isinstance(body, list)

    def test_at_least_two_items(self, server_url):
        _, body = _get(f'{server_url}/api/waypoints')
        assert len(body) >= 2

    def test_each_item_has_lat_lon(self, server_url):
        _, body = _get(f'{server_url}/api/waypoints')
        for wp in body:
            assert 'lat' in wp and 'lon' in wp


# ━━━━━━━━━━━━━━━━━━━━━  POST /api/waypoints  ━━━━━━━━━━━━━━━━━━━━━

class TestPostWaypoints:
    NEW_WPS = [
        {"lat": 35.0, "lon": 135.0, "alt": 0, "label": "テスト出発"},
        {"lat": 34.0, "lon": 136.0, "alt": 0, "label": "テスト目的"},
    ]

    def test_status_200(self, server_url):
        status, _ = _post(f'{server_url}/api/waypoints', self.NEW_WPS)
        assert status == 200

    def test_returns_ok_true(self, server_url):
        _, body = _post(f'{server_url}/api/waypoints', self.NEW_WPS)
        assert body.get('ok') is True

    def test_count_matches(self, server_url):
        _, body = _post(f'{server_url}/api/waypoints', self.NEW_WPS)
        assert body.get('count') == len(self.NEW_WPS)

    def test_get_reflects_update(self, server_url):
        _post(f'{server_url}/api/waypoints', self.NEW_WPS)
        _, body = _get(f'{server_url}/api/waypoints')
        assert body[0]['lat'] == pytest.approx(35.0)

    def test_invalid_body_returns_400(self, server_url):
        status, _ = _post(f'{server_url}/api/waypoints', {"not": "a list"})
        assert status == 400


# ━━━━━━━━━━━━━━━━━━━━━  POST /api/simulate  ━━━━━━━━━━━━━━━━━━━━━

class TestStartSimulate:
    WPS = [
        {"lat": 35.0, "lon": 135.0, "alt": 0, "label": "A"},
        {"lat": 34.0, "lon": 136.0, "alt": 0, "label": "B"},
    ]

    def test_returns_202(self, server_url):
        with patch.object(ui_server, '_simulate_bg'):
            status, _ = _post(f'{server_url}/api/simulate', {"waypoints": self.WPS})
        assert status == 202

    def test_returns_job_id(self, server_url):
        with patch.object(ui_server, '_simulate_bg'):
            _, body = _post(f'{server_url}/api/simulate', {"waypoints": self.WPS})
        assert 'job_id' in body
        assert isinstance(body['job_id'], str)

    def test_too_few_waypoints_returns_400(self, server_url):
        status, _ = _post(f'{server_url}/api/simulate',
                          {"waypoints": [{"lat": 35.0, "lon": 135.0, "alt": 0}]})
        assert status == 400

    def test_uses_stored_waypoints_when_body_empty(self, server_url):
        # WP を先にセット
        _post(f'{server_url}/api/waypoints', self.WPS)
        with patch.object(ui_server, '_simulate_bg'):
            status, body = _post(f'{server_url}/api/simulate', {})
        assert status == 202
        assert 'job_id' in body


# ━━━━━━━━━━━━━━━━━━━━━  GET /api/simulate/:id  ━━━━━━━━━━━━━━━━━━━━━

class TestGetJob:
    def test_not_found_returns_404(self, server_url):
        status, _ = _get(f'{server_url}/api/simulate/nonexistent')
        assert status == 404

    def test_running_job_status(self, server_url):
        # ジョブを直接登録してステータスを確認
        with ui_server._job_lock:
            ui_server._jobs['test-run'] = {"status": "running"}
        _, body = _get(f'{server_url}/api/simulate/test-run')
        assert body['status'] == 'running'

    def test_done_job_has_result(self, server_url):
        fake_result = {"trajectory": [], "hit_ground": False,
                       "total_dist_km": 100.0, "total_time_s": 300.0, "max_speed_ms": 200.0}
        with ui_server._job_lock:
            ui_server._jobs['test-done'] = {"status": "done", "result": fake_result}
        _, body = _get(f'{server_url}/api/simulate/test-done')
        assert body['status'] == 'done'
        assert 'result' in body
        assert body['result']['total_dist_km'] == pytest.approx(100.0)


# ━━━━━━━━━━━━━━━━━━━━━  静的ファイル配信  ━━━━━━━━━━━━━━━━━━━━━

class TestStaticFiles:
    def test_root_returns_html(self, server_url):
        req = urllib.request.Request(server_url + '/')
        with urllib.request.urlopen(req, timeout=5) as r:
            assert r.status == 200
            ct = r.headers.get('Content-Type', '')
            assert 'text/html' in ct

    def test_root_contains_vue(self, server_url):
        req = urllib.request.Request(server_url + '/')
        with urllib.request.urlopen(req, timeout=5) as r:
            content = r.read().decode()
        assert 'vue' in content.lower()

    def test_unknown_path_returns_404(self, server_url):
        status, _ = _get(f'{server_url}/api/does-not-exist')
        assert status == 404


# ━━━━━━━━━━━━━━━━━━━━━  CORS ヘッダー  ━━━━━━━━━━━━━━━━━━━━━

class TestCorsHeaders:
    def test_get_has_cors_header(self, server_url):
        req = urllib.request.Request(f'{server_url}/api/ping')
        with urllib.request.urlopen(req, timeout=5) as r:
            assert r.headers.get('Access-Control-Allow-Origin') == '*'

    def test_options_returns_204(self, server_url):
        req = urllib.request.Request(
            f'{server_url}/api/waypoints', method='OPTIONS')
        with urllib.request.urlopen(req, timeout=5) as r:
            assert r.status == 204
