import logging
import numpy as np

logger = logging.getLogger(__name__)

from config import (MAX_ACCEL, MAX_SPEED, DRAG_K, DT,
                    INIT_AZIMUTH_DEG, INIT_ELEVATION_DEG, INIT_SPEED,
                    CAPTURE_R, HIT_RADIUS, COLLISION_LOOKAHEAD,
                    LAUNCH_CLIMB_RATIO, TERMINAL_GUIDANCE, POPUP_DIVE, ARH,
                    TERRAIN_TIME_CONST, TERRAIN_LOOKAHEAD, NAV_MODE,
                    FLIGHT_PROFILE, LOW_VALLEY_FAN_DEG, LOW_VALLEY_RAYS, LOW_VALLEY_COST)
from nav import make_sensor
from terrain import terrain_floor, terrain_height_at
from geo import local_to_geo
from obstacles import get_fixed_obstacles, get_moving_obstacles, FixedObstacle, MovingObstacle

def _auto_terminal_time() -> float:
    """popup_dive の guidance_time=None のとき、ポップアップ→ダイブに必要な時間を自動算出する。"""
    angle_rad      = abs(np.radians(POPUP_DIVE.dive_angle_deg))
    tan_a          = np.tan(angle_rad) if angle_rad > 1e-6 else float('inf')
    dive_horiz     = POPUP_DIVE.popup_height / tan_a

    climb_vz       = MAX_SPEED * 0.5
    climb_vh       = float(np.sqrt(max(MAX_SPEED**2 - climb_vz**2, 0.0)))
    climb_time     = POPUP_DIVE.popup_height / climb_vz
    horiz_in_climb = climb_vh * climb_time

    needed_horiz   = horiz_in_climb + dive_horiz * 2.0
    auto_t         = (needed_horiz / MAX_SPEED + climb_time) * 1.3
    return round(auto_t, 1)


# フェーズ定数
PHASE_LAUNCH   = 'launch'    # 離陸：地形クリアランス到達まで垂直優先上昇
PHASE_CRUISE   = 'cruise'    # 巡航：地形追従 + WP誘導
PHASE_TERMINAL = 'terminal'  # 終末：クリアランス解除・目標高度へダイブ


def _avoidance_steer(pos: np.ndarray, desired_vel: np.ndarray, desired_speed: float,
                     t: float,
                     fixed_obs: list[FixedObstacle],
                     moving_obs: list[MovingObstacle]) -> np.ndarray:
    """障害物回避成分を desired_vel に重畳して返す。

    固定障害物は dist_from_surface / repulsion_dir で形状に応じた距離と方向を取得。
    反発を横方向優先に分解することで、WP 引力と打ち消し合わずまわり込む動作を実現。
    """
    repulse = np.zeros(3)

    # 進行方向の水平単位ベクトル
    heading_h = np.array([desired_vel[0], desired_vel[1], 0.0])
    h_mag = float(np.linalg.norm(heading_h))
    heading_h = heading_h / h_mag if h_mag > 1e-9 else np.array([1.0, 0.0, 0.0])

    def _add_repulse(away: np.ndarray, dist: float, zone: float) -> None:
        weight = (1.0 - dist / zone) ** 2
        away_h = np.array([away[0], away[1], 0.0])
        dot    = float(np.dot(away_h, heading_h))
        lateral = away_h - dot * heading_h
        lat_mag = float(np.linalg.norm(lateral))
        if lat_mag > 1e-9:
            repulse[:] += (lateral / lat_mag) * weight * 2.0
        repulse[:] += heading_h * max(-dot, 0.0) * weight * 0.3

    # 固定障害物: SphereObstacle / BoxObstacle 共通インターフェース
    for obs in fixed_obs:
        dist = obs.dist_from_surface(pos)
        if 0.0 < dist < obs.zone:
            logger.debug("t=%.1fs 回避[固定] %s dist=%.0fm zone=%.0fm", t, obs.label, dist, obs.zone)
            _add_repulse(obs.repulsion_dir(pos), dist, obs.zone)

    # 移動体: 中心距離ベース（球状扱い）
    for obs in moving_obs:
        threat_pos = obs.pos_at(t + COLLISION_LOOKAHEAD)
        to_threat  = threat_pos - pos
        dist       = float(np.linalg.norm(to_threat))
        if 1.0 < dist < obs.zone:
            logger.debug("t=%.1fs 回避[移動] %s dist=%.0fm zone=%.0fm", t, obs.label, dist, obs.zone)
            _add_repulse(-(to_threat / dist), dist, obs.zone)

    if float(np.linalg.norm(repulse)) < 1e-9:
        return desired_vel

    new_vel = desired_vel + repulse * desired_speed
    new_spd = float(np.linalg.norm(new_vel))
    if new_spd < 1e-9:
        return desired_vel
    return new_vel * (desired_speed / new_spd)


def _vz_command(alt_error: float, current_vz: float, desired_speed: float) -> float:
    """高度誤差 + 現在の垂直速度 → 垂直速度コマンド。

    上昇: 比例制御 (TERRAIN_TIME_CONST)
    降下: √プロファイル + 先読みブレーキ
        stop_alt = pos + vz²/(2a) が target より低ければ即強制減速
        → 離散時間でも目標高度でオーバーシュートしない
    """
    if alt_error >= 0:
        return float(np.clip(alt_error / TERRAIN_TIME_CONST, 0.0, desired_speed * 0.5))

    remaining = -alt_error
    braking_vz = -float(np.sqrt(2.0 * MAX_ACCEL * remaining))
    max_descent = -desired_speed * 0.15
    return float(max(braking_vz, max_descent))


def _valley_dir(pos: np.ndarray, speed: float, to_wp_unit: np.ndarray) -> np.ndarray:
    """低空プロファイル用: ±LOW_VALLEY_FAN_DEG のファン状にスキャンし、
    「先読み地形の最大高度 + 迂回コスト」が最小の水平方向を返す。

    WP方向を中心に複数レイを投射し、各レイで TERRAIN_LOOKAHEAD 秒先まで
    8点サンプリングして最大地形高を求める。正面からの逸脱角度ごとにコストを
    加算することで、地形が低くなければ遠回りしない設計になっている。
    """
    best_dir   = to_wp_unit.copy()
    best_score = float('inf')
    best_angle = 0.0

    for angle_deg in np.linspace(-LOW_VALLEY_FAN_DEG, LOW_VALLEY_FAN_DEG, LOW_VALLEY_RAYS):
        a = np.radians(angle_deg)
        c, s = np.cos(a), np.sin(a)
        d = np.array([c * to_wp_unit[0] - s * to_wp_unit[1],
                      s * to_wp_unit[0] + c * to_wp_unit[1]])

        max_h = 0.0
        for t_scan in np.linspace(2.0, TERRAIN_LOOKAHEAD, 8):
            p = np.array([pos[0] + d[0] * speed * t_scan,
                          pos[1] + d[1] * speed * t_scan,
                          0.0])
            lat, lon = local_to_geo(p)
            max_h = max(max_h, terrain_height_at(lat, lon))

        # 逸脱が大きいほどコストを加算（LOW_VALLEY_COST m ≒ 完全逸脱時のペナルティ）
        deviation = abs(angle_deg) / LOW_VALLEY_FAN_DEG  # 0〜1
        score     = max_h + deviation * LOW_VALLEY_COST

        if score < best_score:
            best_score = score
            best_angle = angle_deg
            best_dir   = d.copy()

    logger.debug("valley: 最適角=%+.0f°  スコア=%.0fm (地形=%.0fm + コスト=%.0fm)",
                 best_angle, best_score,
                 best_score - (abs(best_angle) / LOW_VALLEY_FAN_DEG) * LOW_VALLEY_COST,
                 (abs(best_angle) / LOW_VALLEY_FAN_DEG) * LOW_VALLEY_COST)
    return best_dir


def _wp_pos(entry, t: float) -> np.ndarray:
    """ウェイポイントエントリ (np.ndarray | Callable[[float], np.ndarray]) を
    時刻 t の ENU MSL 座標に解決する。"""
    return entry(t) if callable(entry) else entry


def _intercept_pos(entry, t_now: float, from_pos_2d: np.ndarray, horiz_speed: float) -> np.ndarray:
    """移動目標の予測着弾点を反復収束で求める。

    ETA = dist(予測位置, 現在位置) / horiz_speed を繰り返し更新して収束させる。
    固定目標 (ndarray) はそのまま返す。
    """
    if not callable(entry):
        return entry
    aim = entry(t_now)
    eta = float(np.linalg.norm(aim[:2] - from_pos_2d)) / max(horiz_speed, 1.0)
    for _ in range(5):
        aim = entry(t_now + eta)
        new_eta = float(np.linalg.norm(aim[:2] - from_pos_2d)) / max(horiz_speed, 1.0)
        if abs(new_eta - eta) < 0.05:
            break
        eta = new_eta
    return aim


def turning_radius(speed: float) -> float:
    """指定速度における最小旋回半径 r = v² / a"""
    return speed ** 2 / MAX_ACCEL if speed > 0.1 else 0.0


def simulate(waypoints: list, profile: str | None = None) -> dict:
    """ウェイポイント列に沿った飛翔軌道をシミュレーションする。

    フェーズ遷移:
        LAUNCH  → CRUISE   : pos.z >= 地形クリアランス高度
        CRUISE  → TERMINAL : 最終WPへの残り推定時間 < TERMINAL_GUIDANCE_TIME

    Returns dict with keys:
        pos, vel, accel  : shape (N,3)
        time, speed      : shape (N,)
        elevation, azimuth : shape (N,)  [degrees]
        phase            : shape (N,)    ['launch'|'cruise'|'terminal']
        hit_ground       : bool
    """
    _profile = profile if profile is not None else FLIGHT_PROFILE
    final_is_moving = callable(waypoints[-1])
    if _profile == 'auto':
        _profile = 'standard' if final_is_moving else 'low'
        logger.info("  飛行プロファイル: AUTO → %s (%s)",
                    _profile.upper(), "移動目標" if final_is_moving else "固定目標")
    else:
        logger.info("  飛行プロファイル: %s", _profile.upper())

    # 終末誘導タイプ解決: 'auto' は最終WP種別で振り分け
    _terminal_type = TERMINAL_GUIDANCE
    if _terminal_type == 'auto':
        _terminal_type = 'arh' if final_is_moving else 'popup_dive'
    logger.info("  終末誘導: %s", _terminal_type.upper())

    if _terminal_type == 'popup_dive':
        terminal_time = (_auto_terminal_time() if POPUP_DIVE.guidance_time is None
                         else float(POPUP_DIVE.guidance_time))
        if POPUP_DIVE.guidance_time is None:
            logger.info("  終末誘導時間: 自動算出 %s s", terminal_time)
    else:  # arh
        terminal_time = float(ARH.engage_time)
        logger.info("  終末誘導時間: %s s (ARH engage)", terminal_time)

    pos = _wp_pos(waypoints[0], 0.0).copy()
    az  = np.radians(INIT_AZIMUTH_DEG)
    el  = np.radians(INIT_ELEVATION_DEG)
    vel = np.array([np.cos(el) * np.cos(az),
                    np.cos(el) * np.sin(az),
                    np.sin(el)]) * INIT_SPEED

    # ナビゲーションセンサー（誘導で使う推定位置を提供）
    sensor  = make_sensor(NAV_MODE, pos, vel)
    nav_pos = pos.copy()   # センサーが返す推定位置（誘導計算はこちらを使う）
    logger.info("  ナビゲーション: %s", NAV_MODE.upper())

    pos_log   = [pos.copy()]
    vel_log   = [vel.copy()]
    accel_log = [np.zeros(3)]
    t_log     = [0.0]
    phase_log = [PHASE_LAUNCH]
    t         = 0.0
    wp        = 1
    phase     = PHASE_LAUNCH
    hit_ground = False
    terminal_subphase = None   # 'popup' | 'dive'  (popup_dive のみ)
    popup_z           = None   # ポップアップ目標高度 (MSL)
    dive_horiz_dist   = None   # このX以下になったらダイブ開始 (m)
    dive_start        = None   # ダイブ開始点 [x, y, popup_z]
    arh_prev_los      = None   # 前ステップの LOS 単位ベクトル (ARH 用)
    _arh_debug_step   = 0      # ARH ステップカウンター（デバッグ用、後で削除）

    fixed_obs  = get_fixed_obstacles()
    moving_obs = get_moving_obstacles()

    while wp < len(waypoints) and t < 7200.0:
        # 誘導計算はすべてセンサー推定位置 (nav_pos) を使う
        # 真位置 (pos) は物理積分・衝突判定・ログにのみ使う
        wp_pos     = _wp_pos(waypoints[wp], t)
        to_wp_orig = wp_pos - nav_pos
        dist_horiz = float(np.linalg.norm(to_wp_orig[:2]))
        speed      = float(np.linalg.norm(vel))

        # 通過判定
        # - 通常巡航: 水平距離 < CAPTURE_R、または通り越し検知（高速時は旋回半径で判定）
        # - 終末ダイブ: 3D距離 < CAPTURE_R（高度方向も含めた精度で判定）
        # - ポップアップ中: overshoot検知を無効（上空通過による誤捕捉を防ぐ）
        dist_3d       = float(np.linalg.norm(to_wp_orig))
        overshoot_r   = max(CAPTURE_R * 10, turning_radius(speed) * 0.5)
        vel_toward_wp = float(np.dot(vel[:2], to_wp_orig[:2]))
        in_popup      = (phase == PHASE_TERMINAL and terminal_subphase == 'popup')
        in_dive       = (phase == PHASE_TERMINAL and terminal_subphase == 'dive')
        in_arh        = (phase == PHASE_TERMINAL and _terminal_type == 'arh')
        if in_dive or in_arh:
            passed = dist_3d < HIT_RADIUS
        else:
            passed = dist_horiz < CAPTURE_R or (not in_popup and vel_toward_wp < 0 and dist_horiz < overshoot_r)
        if passed:
            logger.info("t=%7.1f s   [%8s] WP%d 通過  pos=(%.0f, %.0f, %.0f) m  dist_h=%.0fm  dist_3d=%.0fm",
                        t, phase, wp, pos[0], pos[1], pos[2], dist_horiz, dist_3d)
            wp += 1
            continue

        # 希望速度（最終WPに近づくと減速）
        is_final_wp = (wp == len(waypoints) - 1)
        if is_final_wp and speed > 1.0:
            braking_dist = speed ** 2 / (2.0 * MAX_ACCEL)
            desired_speed = (max(10.0, MAX_SPEED * dist_horiz / (braking_dist * 2.0))
                             if dist_horiz < braking_dist * 2.0 else MAX_SPEED)
        else:
            desired_speed = MAX_SPEED

        # ── フェーズ遷移チェック ──────────────────────────────────────────────
        floor_z    = terrain_floor(nav_pos, vel, speed)
        time_to_go = dist_horiz / speed if speed > 1.0 else float('inf')

        if phase == PHASE_LAUNCH and nav_pos[2] >= floor_z:
            phase = PHASE_CRUISE
            logger.info("t=%7.1f s   [巡航] 移行  高度 %.0fm MSL  クリアランス %.0fm",
                        t, nav_pos[2], nav_pos[2] - terrain_height_at(*local_to_geo(nav_pos)))

        if phase == PHASE_CRUISE and is_final_wp and time_to_go < terminal_time:
            phase = PHASE_TERMINAL
            if _terminal_type == 'popup_dive':
                popup_z = wp_pos[2] + POPUP_DIVE.popup_height
                angle_rad       = abs(np.radians(POPUP_DIVE.dive_angle_deg))
                dive_horiz_dist = (POPUP_DIVE.popup_height / np.tan(angle_rad)
                                   if angle_rad > 1e-6 else 0.0)
                hdir_now       = to_wp_orig[:2] / dist_horiz if dist_horiz > 1e-9 else np.array([1.0, 0.0])
                dive_start_xy  = wp_pos[:2] - hdir_now * dive_horiz_dist
                dive_start     = np.array([dive_start_xy[0], dive_start_xy[1], popup_z])
                at_popup       = nav_pos[2] >= popup_z * 0.98
                close_enough   = dist_horiz <= dive_horiz_dist * 1.05
                terminal_subphase = 'dive' if (at_popup and close_enough) else 'popup'
                logger.info("t=%7.1f s   [終末/PopupDive] 移行  残り %.1fs  高度差 %+.0fm  "
                            "popup目標 %.0fm MSL  ダイブ開始距離 %.0fm → [%s]",
                            t, time_to_go, wp_pos[2] - nav_pos[2],
                            popup_z, dive_horiz_dist, terminal_subphase)
            else:  # arh
                logger.info("t=%7.1f s   [終末/ARH] 移行  残り %.1fs  目標まで %.0fm  "
                            "高度 %.0fm MSL  N=%.1f",
                            t, time_to_go, dist_horiz, nav_pos[2], ARH.nav_constant)
                logger.info("  [ARH] 位置=(%.0f,%.0f,%.0f)  速度=(%.1f,%.1f,%.1f)  |v|=%.1f",
                            pos[0], pos[1], pos[2], vel[0], vel[1], vel[2], speed)
                logger.info("  [ARH] 目標=(%.0f,%.0f)  方位差 Δx=%.0f  Δy=%.0f",
                            wp_pos[0], wp_pos[1], wp_pos[0]-pos[0], wp_pos[1]-pos[1])

        # ── ガイダンス（フェーズ別 desired_vel を確定）──────────────────────────
        horiz_dir = to_wp_orig[:2] / dist_horiz

        if phase == PHASE_LAUNCH:
            # 速度バジェットの LAUNCH_CLIMB_RATIO を上昇に割り当て
            vz_desired = float(np.clip(LAUNCH_CLIMB_RATIO * desired_speed,
                                       0.0, desired_speed * 0.95))
            horiz_spd  = float(np.sqrt(max(desired_speed ** 2 - vz_desired ** 2, 1.0)))
            desired_vel = np.array([horiz_dir[0] * horiz_spd,
                                    horiz_dir[1] * horiz_spd,
                                    vz_desired])

        elif phase == PHASE_TERMINAL:
            if _terminal_type == 'arh':
                # ── ARH: 3D 予測インターセプト直接誘導 ────────────────────
                # ETA 反復（3回）で将来の目標位置を推定し、そこへ 3D で直進する。
                # popup/dive の水平誘導と同原理を 3D に拡張した版。
                r_3d_dist = float(np.linalg.norm(wp_pos - nav_pos))
                eta = r_3d_dist / max(speed, 1.0)
                if callable(waypoints[wp]):
                    for _ in range(3):
                        wp_aim   = waypoints[wp](t + eta)
                        new_dist = float(np.linalg.norm(wp_aim - nav_pos))
                        eta      = new_dist / max(speed, 1.0)
                else:
                    wp_aim = wp_pos
                aim_vec  = wp_aim - nav_pos
                aim_dist = float(np.linalg.norm(aim_vec))
                desired_vel = (aim_vec / aim_dist * desired_speed
                               if aim_dist > 1e-9 else vel / max(speed, 1.0) * desired_speed)

                _arh_debug_step += 1
                if _arh_debug_step <= 5:
                    logger.info("ARH#%d t=%.1fs  pos=(%.0f,%.0f,%.0f)  vel=(%.1f,%.1f,%.1f)  "
                                "aim=(%.0f,%.0f,%.0f)  dv=(%.1f,%.1f,%.1f)  eta=%.1fs",
                                _arh_debug_step, t,
                                pos[0], pos[1], pos[2], vel[0], vel[1], vel[2],
                                wp_aim[0], wp_aim[1], wp_aim[2],
                                desired_vel[0], desired_vel[1], desired_vel[2], eta)

            else:
                # ── popup_dive: 予測着弾点へのポップアップ→ダイブ ─────────
                dive_el_abs = abs(np.radians(POPUP_DIVE.dive_angle_deg))
                if terminal_subphase == 'dive':
                    _aim_spd = desired_speed * np.cos(dive_el_abs)
                else:
                    _aim_spd = desired_speed
                wp_pos_aim = _intercept_pos(waypoints[wp], t, nav_pos[:2], _aim_spd)
                to_aim     = wp_pos_aim[:2] - nav_pos[:2]
                aim_dist   = float(np.linalg.norm(to_aim))
                aim_dir    = to_aim / aim_dist if aim_dist > 1e-9 else horiz_dir

                if terminal_subphase == 'popup':
                    if dive_horiz_dist is not None and aim_dist > 1e-9:
                        dive_start[:2] = wp_pos_aim[:2] - aim_dir * dive_horiz_dist
                    to_ds      = dive_start[:2] - nav_pos[:2]
                    dist_to_ds = float(np.linalg.norm(to_ds))
                    ds_dir     = to_ds / dist_to_ds if dist_to_ds > 1.0 else aim_dir
                    vz_desired  = _vz_command(popup_z - nav_pos[2], float(vel[2]), desired_speed)
                    horiz_spd   = float(np.sqrt(max(desired_speed ** 2 - vz_desired ** 2, 1.0)))
                    desired_vel = np.array([ds_dir[0] * horiz_spd,
                                            ds_dir[1] * horiz_spd,
                                            vz_desired])
                    at_popup   = nav_pos[2] >= popup_z * 0.98
                    near_start = dist_to_ds <= max(dive_horiz_dist * 0.1, CAPTURE_R * 2)
                    if at_popup and near_start:
                        terminal_subphase = 'dive'
                        logger.info("t=%7.1f s   [終末] ダイブ開始  高度 %.0fm  目標まで水平 %.0fm  ダイブ角 %.0f°",
                                    t, nav_pos[2], dist_horiz, POPUP_DIVE.dive_angle_deg)
                else:
                    dive_el     = np.radians(POPUP_DIVE.dive_angle_deg)
                    vz_dive     = desired_speed * np.sin(dive_el)
                    horiz_spd   = desired_speed * np.cos(dive_el_abs)
                    desired_vel = np.array([aim_dir[0] * horiz_spd,
                                            aim_dir[1] * horiz_spd,
                                            vz_dive])

        else:  # CRUISE
            if _profile == 'low':
                # 目標高度: 地形クリアランスギリギリ（谷では積極降下）
                target_z = floor_z
                # 水平方向: 谷を探して遠回りルートを選択
                valley = _valley_dir(nav_pos, speed, horiz_dir)
                # WP接近につれて WP引力を強める（最低30%は常にWP方向を維持）
                wp_pull     = float(np.clip(5000.0 / max(dist_horiz, 1.0), 0.30, 1.0))
                blended     = wp_pull * horiz_dir + (1.0 - wp_pull) * valley
                b_mag       = float(np.linalg.norm(blended))
                cruise_dir  = blended / b_mag if b_mag > 1e-9 else horiz_dir
            else:
                # 標準: 地形フロアと WP高度の大きい方を目標に制御
                target_z   = max(float(wp_pos[2]), floor_z)
                cruise_dir = horiz_dir
            vz_desired = _vz_command(target_z - nav_pos[2], float(vel[2]), desired_speed)
            horiz_spd  = float(np.sqrt(max(desired_speed ** 2 - vz_desired ** 2, 1.0)))
            desired_vel = np.array([cruise_dir[0] * horiz_spd,
                                    cruise_dir[1] * horiz_spd,
                                    vz_desired])

        # 障害物回避（ダイブ中・ARH終末中は無効 — 突入優先; ポップアップ中は有効）
        if phase != PHASE_TERMINAL or terminal_subphase == 'popup':
            desired_vel = _avoidance_steer(nav_pos, desired_vel, desired_speed,
                                           t, fixed_obs, moving_obs)

        # 速度誤差ガイダンス → 加速度
        dv     = desired_vel - vel
        dv_mag = float(np.linalg.norm(dv))
        if dv_mag > MAX_ACCEL * DT:
            accel = (dv / dv_mag) * MAX_ACCEL
        elif dv_mag > 1e-9:
            accel = dv / DT
        else:
            accel = np.zeros(3)

        # 空気抵抗（有翼機: 揚力が重力を相殺するため重力項なし）
        drag_accel  = -(DRAG_K * speed ** 2) * (vel / speed) if speed > 0.1 else np.zeros(3)
        total_accel = accel + drag_accel

        # オイラー積分（真の位置・速度を更新）
        vel += total_accel * DT
        spd  = float(np.linalg.norm(vel))
        if spd > MAX_SPEED:
            vel *= MAX_SPEED / spd
        pos += vel * DT
        t   += DT

        # センサー更新: 真位置・加速度からセンサー推定位置を算出
        nav_pos = sensor.update(pos, vel, total_accel, DT)

        # 地面衝突判定（離陸フェーズでは地面スタートなので初手スキップ）
        if phase != PHASE_LAUNCH:
            lat, lon = local_to_geo(pos)
            ground_z = terrain_height_at(lat, lon)
            if pos[2] <= ground_z:
                pos[2] = ground_z
                # 終末ダイブ中に最終WP付近の地面に着弾 → 命中として扱う
                final_wp_pos   = _wp_pos(waypoints[-1], t)
                horiz_to_final = float(np.linalg.norm(pos[:2] - final_wp_pos[:2]))
                dist3d_to_final = float(np.linalg.norm(pos - final_wp_pos))
                is_hit = wp == len(waypoints) - 1 and (
                    (in_dive and horiz_to_final < HIT_RADIUS * 3) or
                    (in_arh  and dist3d_to_final < HIT_RADIUS * 3)
                )
                if is_hit:
                    logger.info("t=%7.1f s   *** 目標命中 ***  pos=(%.0f, %.0f, %.0f) m  "
                                "水平 %.0fm  3D %.0fm",
                                t, pos[0], pos[1], pos[2], horiz_to_final, dist3d_to_final)
                    wp += 1
                else:
                    hit_ground = True
                    logger.error("t=%7.1f s   *** 地面衝突・飛翔終了 ***  pos=(%.0f, %.0f, %.0f) m",
                                 t, pos[0], pos[1], pos[2])
                pos_log.append(pos.copy())
                vel_log.append(vel.copy())
                accel_log.append(total_accel.copy())
                t_log.append(t)
                phase_log.append(phase)
                break

        logger.debug(
            "t=%6.1fs [%8s] pos=(%7.0f,%7.0f,%5.0f)m  spd=%5.1fm/s  "
            "accel=%5.1fm/s²  nav_err=%5.1fm",
            t, phase, pos[0], pos[1], pos[2], spd,
            float(np.linalg.norm(total_accel)),
            float(np.linalg.norm(nav_pos - pos)))
        pos_log.append(pos.copy())
        vel_log.append(vel.copy())
        accel_log.append(total_accel.copy())
        t_log.append(t)
        phase_log.append(phase)

    vel_arr   = np.array(vel_log)
    speeds    = np.linalg.norm(vel_arr, axis=1)
    safe_spd  = np.where(speeds > 0.01, speeds, 1.0)
    elevation = np.degrees(np.arcsin(np.clip(vel_arr[:, 2] / safe_spd, -1.0, 1.0)))
    azimuth   = np.degrees(np.unwrap(np.arctan2(vel_arr[:, 1], vel_arr[:, 0])))

    return {
        'pos':          np.array(pos_log),
        'vel':          vel_arr,
        'accel':        np.array(accel_log),
        'time':         np.array(t_log),
        'speed':        speeds,
        'elevation':    elevation,
        'azimuth':      azimuth,
        'phase':        np.array(phase_log),
        'hit_ground':   hit_ground,
        'profile_used': _profile,
    }
