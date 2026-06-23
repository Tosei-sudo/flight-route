"""
pytest 設定・共有フィクスチャ

プロジェクトルートを sys.path に追加し、テスト中のノイズを抑制する。
"""

import sys
import os
import logging
from unittest.mock import MagicMock

# プロジェクトルートを import パスに追加
_ROOT = os.path.dirname(os.path.dirname(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

# rasterio は terrain.py がモジュールレベルで import する。
# CI 環境には DEM タイルが存在しないため rasterio の実機能は不要。
# MagicMock を差し込んで ImportError を回避する。
if 'rasterio' not in sys.modules:
    sys.modules['rasterio'] = MagicMock()

# DEM 未検出・GPS ノイズなどの INFO ログをテスト出力から除外
logging.disable(logging.INFO)

# matplotlib を import するコードがあっても GUI ウィンドウを開かない
os.environ.setdefault('MPLBACKEND', 'Agg')
