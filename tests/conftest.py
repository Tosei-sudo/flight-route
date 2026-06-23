"""
pytest 設定・共有フィクスチャ

プロジェクトルートを sys.path に追加し、テスト中のノイズを抑制する。
"""

import sys
import os
import logging

# プロジェクトルートを import パスに追加
_ROOT = os.path.dirname(os.path.dirname(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

# DEM 未検出・GPS ノイズなどの INFO ログをテスト出力から除外
logging.disable(logging.INFO)

# matplotlib を import するコードがあっても GUI ウィンドウを開かない
os.environ.setdefault('MPLBACKEND', 'Agg')
