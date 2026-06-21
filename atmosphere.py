"""
ISA 標準大気モデル

高度に応じた空気密度比 ρ(h)/ρ₀ を返す。
対流圏 (0〜11,000 m) と成層圏 (11,000〜20,000 m) の 2 層で近似する。

参考値:
  0 m    → 1.000  (海面)
  1000 m → 0.907
  3000 m → 0.742
  5000 m → 0.601
 11000 m → 0.297  (対流圏界面)
 20000 m → 0.072
"""

import numpy as np

# ISA 定数
_T0  = 288.15    # K    海面気温
_L   = 0.0065   # K/m  気温減率（対流圏）
_T11 = 216.65   # K    対流圏界面気温（成層圏一定）
_EXP_TROP = 4.2561          # g*M/(R*L) - 1
_EXP_STRAT = 1.5769e-4      # g*M/(R*T11) [1/m]
_RHO_RATIO_11 = (_T11 / _T0) ** _EXP_TROP   # ≈ 0.2971


def air_density_ratio(alt_msl: float) -> float:
    """高度 alt_msl [m MSL] における密度比 ρ/ρ₀ を返す (ISA 標準大気)。

    0 m 以下は海面密度 (1.0) にクランプする。
    """
    h = max(0.0, float(alt_msl))
    if h < 11_000.0:
        return ((_T0 - _L * h) / _T0) ** _EXP_TROP
    return _RHO_RATIO_11 * float(np.exp(-_EXP_STRAT * (h - 11_000.0)))
