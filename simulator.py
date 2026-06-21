import logging
from dataclasses import dataclass, field, replace
from concurrent.futures import ThreadPoolExecutor
import numpy as np

logger = logging.getLogger(__name__)

from config import (MAX_ACCEL, MAX_SPEED, DRAG_K, DT, GDB_OUTPUT_PATH,
                    INIT_AZIMUTH_DEG, INIT_ELEVATION_DEG, INIT_SPEED,
                    CAPTURE_R, HIT_RADIUS, COLLISION_LOOKAHEAD,
                    LAUNCH_CLIMB_RATIO, TERMINAL_GUIDANCE, POPUP_DIVE, ARH,
                    TERRAIN_TIME_CONST, TERRAIN_LOOKAHEAD, NAV_MODE,
                    FLIGHT_PROFILE, LOW_VALLEY_FAN_DEG, LOW_VALLEY_RAYS, LOW_VALLEY_COST,
                    PopupDiveParams, ARHParams)
from nav import make_sensor
from terrain import terrain_floor, terrain_height_at
from geo import local_to_geo
from obstacles import get_fixed_obstacles, get_moving_obstacles, FixedObstacle, MovingObstacle
from atmosphere import air_density_ratio as _air_density_ratio
from wind import get_wind_field as _get_wind_field


@dataclass
class SimParams:
    """シミュレーション1回分のパラメータ。デフォルト値は config モジュールから取得。"""
    max_accel:          float          = MAX_ACCEL
    max_speed:          float          = MAX_SPEED
    drag_k:             float          = DRAG_K
    dt:                 float          = DT
    init_azimuth_deg:   float          = INIT_AZIMUTH_DEG
    init_elevation_deg: float          = INIT_ELEVATION_DEG
    init_speed:         float          = INIT_SPEED
    capture_r:          float          = CAPTURE_R
    hit_radius:         float          = HIT_RADIUS
    collision_lookahead:float          = COLLISION_LOOKAHEAD
    launch_climb_ratio: float          = LAUNCH_CLIMB_RATIO
    terminal_guidance:  str            = TERMINAL_GUIDANCE
    popup_dive:         PopupDiveParams= field(default_factory=lambda: replace(POPUP_DIVE))
    arh:                ARHParams      = field(default_factory=lambda: replace(ARH))
    terrain_time_const: float          = TERRAIN_TIME_CONST
    terrain_lookahead:  float          = TERRAIN_LOOKAHEAD
    nav_mode:           str            = NAV_MODE
    flight_profile:     str            = FLIGHT_PROFILE
    low_valley_fan_deg: float          = LOW_VALLEY_FAN_DEG
    low_valley_rays:    int            = LOW_VALLEY_RAYS
    low_valley_cost:    float          = LOW_VALLEY_COST


# フェーズ定数
PHASE_LAUNCH   = 'launch'
PHASE_CRUISE   = 'cruise'
PHASE_TERMINAL = 'terminal'


def _wp_pos(entry, t: float) -> np.ndarray:
    return entry(t) if callable(entry) else entry


def _intercept_pos(entry, t_now: float, from_pos_2d: np.ndarray, horiz_speed: float) -> np.ndarray:
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


class Simulator:
    """1回のシミュレーションを実行するクラス。

    インスタンスごとに独立した SimParams を持つため、
    異なるパラメータで複数インスタンスを同時に実行できる。

    Examples
    --------
    # シングル実行（従来通り）
    result = Simulator().run(waypoints)

    # パラメータを変えた比較
    s1 = Simulator(max_speed=250.0)
    s2 = Simulator(max_speed=300.0)
    r1 = s1.run(waypoints)
    r2 = s2.run(waypoints)

    # 並列実行
    results = Simulator.run_parallel([
        (waypoints_a,),
        (waypoints_b, Simulator(max_accel=60.0)),
    ])
    """

    def __init__(self, params: SimParams | None = None, **overrides):
        """
        Parameters
        ----------
        params   : SimParams インスタンス（省略時はデフォルト値）
        **overrides : SimParams フィールドを直接上書き。例: max_speed=250.0
        """
        base = params if params is not None else SimParams()
        self.p = replace(base, **overrides) if overrides else base

    # ── ヘルパーメソッド ────────────────────────────────────────────────────────

    def turning_radius(self, speed: float) -> float:
        return speed ** 2 / self.p.max_accel if speed > 0.1 else 0.0

    def _auto_terminal_time(self) -> float:
        p = self.p
        angle_rad  = abs(np.radians(p.popup_dive.dive_angle_deg))
        tan_a      = np.tan(angle_rad) if angle_rad > 1e-6 else float('inf')
        dive_horiz = p.popup_dive.popup_height / tan_a
        climb_vz   = p.max_speed * 0.5
        climb_vh   = float(np.sqrt(max(p.max_speed**2 - climb_vz**2, 0.0)))
        climb_time = p.popup_dive.popup_height / climb_vz
        needed     = climb_vh * climb_time + dive_horiz * 2.0
        return round((needed / p.max_speed + climb_time) * 1.3, 1)

    def _vz_command(self, alt_error: float, current_vz: float, desired_speed: float) -> float:
        p = self.p
        if alt_error >= 0:
            return float(np.clip(alt_error / p.terrain_time_const, 0.0, desired_speed * 0.5))
        remaining  = -alt_error
        braking_vz = -float(np.sqrt(2.0 * p.max_accel * remaining))
        return float(max(braking_vz, -desired_speed * 0.15))

    def _valley_dir(self, pos: np.ndarray, speed: float, to_wp_unit: np.ndarray) -> np.ndarray:
        p = self.p
        best_dir, best_score = to_wp_unit.copy(), float('inf')
        for angle_deg in np.linspace(-p.low_valley_fan_deg, p.low_valley_fan_deg, p.low_valley_rays):
            a = np.radians(angle_deg)
            c, s = np.cos(a), np.sin(a)
            d = np.array([c*to_wp_unit[0] - s*to_wp_unit[1],
                          s*to_wp_unit[0] + c*to_wp_unit[1]])
            max_h = 0.0
            for t_scan in np.linspace(2.0, p.terrain_lookahead, 8):
                pt   = np.array([pos[0]+d[0]*speed*t_scan, pos[1]+d[1]*speed*t_scan, 0.0])
                lat, lon = local_to_geo(pt)
                max_h = max(max_h, terrain_height_at(lat, lon))
            deviation = abs(angle_deg) / p.low_valley_fan_deg
            score     = max_h + deviation * p.low_valley_cost
            if score < best_score:
                best_score, best_dir = score, d.copy()
        return best_dir

    def _avoidance_steer(self, pos: np.ndarray, desired_vel: np.ndarray, desired_speed: float,
                         t: float,
                         fixed_obs: list[FixedObstacle],
                         moving_obs: list[MovingObstacle]) -> np.ndarray:
        p = self.p
        repulse  = np.zeros(3)
        heading_h = np.array([desired_vel[0], desired_vel[1], 0.0])
        h_mag     = float(np.linalg.norm(heading_h))
        heading_h = heading_h / h_mag if h_mag > 1e-9 else np.array([1.0, 0.0, 0.0])

        def _add(away: np.ndarray, dist: float, zone: float) -> None:
            w      = (1.0 - dist / zone) ** 2
            away_h = np.array([away[0], away[1], 0.0])
            dot    = float(np.dot(away_h, heading_h))
            lat_v  = away_h - dot * heading_h
            lat_m  = float(np.linalg.norm(lat_v))
            if lat_m > 1e-9:
                repulse[:] += (lat_v / lat_m) * w * 2.0
            repulse[:] += heading_h * max(-dot, 0.0) * w * 0.3

        for obs in fixed_obs:
            dist = obs.dist_from_surface(pos)
            if 0.0 < dist < obs.zone:
                _add(obs.repulsion_dir(pos), dist, obs.zone)

        for obs in moving_obs:
            threat = obs.pos_at(t + p.collision_lookahead)
            to_t   = threat - pos
            dist   = float(np.linalg.norm(to_t))
            if 1.0 < dist < obs.zone:
                _add(-(to_t / dist), dist, obs.zone)

        if float(np.linalg.norm(repulse)) < 1e-9:
            return desired_vel
        new_vel = desired_vel + repulse * desired_speed
        new_spd = float(np.linalg.norm(new_vel))
        return new_vel * (desired_speed / new_spd) if new_spd > 1e-9 else desired_vel

    # ── メイン ─────────────────────────────────────────────────────────────────

    def run(self, waypoints: list, profile: str | None = None) -> dict:
        """ウェイポイント列に沿った飛翔軌道をシミュレーションする。

        Returns
        -------
        dict: pos, vel, accel, time, speed, elevation, azimuth, phase,
              hit_ground, profile_used
        """
        p = self.p
        _wind_field = _get_wind_field(GDB_OUTPUT_PATH)
        final_is_moving = callable(waypoints[-1])
        _profile = profile if profile is not None else p.flight_profile
        if _profile == 'auto':
            _profile = 'standard' if final_is_moving else 'low'
            logger.info("  飛行プロファイル: AUTO → %s (%s)",
                        _profile.upper(), "移動目標" if final_is_moving else "固定目標")
        else:
            logger.info("  飛行プロファイル: %s", _profile.upper())

        _terminal_type = p.terminal_guidance
        if _terminal_type == 'auto':
            _terminal_type = 'arh' if final_is_moving else 'popup_dive'
        logger.info("  終末誘導: %s", _terminal_type.upper())

        if _terminal_type == 'popup_dive':
            terminal_time = (self._auto_terminal_time() if p.popup_dive.guidance_time is None
                             else float(p.popup_dive.guidance_time))
            if p.popup_dive.guidance_time is None:
                logger.info("  終末誘導時間: 自動算出 %s s", terminal_time)
        else:
            terminal_time = float(p.arh.engage_time)
            logger.info("  終末誘導時間: %s s (ARH engage)", terminal_time)

        pos = _wp_pos(waypoints[0], 0.0).copy()
        az  = np.radians(p.init_azimuth_deg)
        el  = np.radians(p.init_elevation_deg)
        vel = np.array([np.cos(el)*np.cos(az),
                        np.cos(el)*np.sin(az),
                        np.sin(el)]) * p.init_speed

        sensor  = make_sensor(p.nav_mode, pos, vel)
        nav_pos = pos.copy()
        logger.info("  ナビゲーション: %s", p.nav_mode.upper())

        pos_log   = [pos.copy()]
        vel_log   = [vel.copy()]
        accel_log = [np.zeros(3)]
        t_log     = [0.0]
        phase_log = [PHASE_LAUNCH]
        t         = 0.0
        wp        = 1
        phase     = PHASE_LAUNCH
        hit_ground        = False
        terminal_subphase = None
        popup_z           = None
        dive_horiz_dist   = None
        dive_start        = None
        arh_debug_step    = 0

        fixed_obs  = get_fixed_obstacles()
        moving_obs = get_moving_obstacles()

        while wp < len(waypoints) and t < 7200.0:
            wp_pos     = _wp_pos(waypoints[wp], t)
            to_wp_orig = wp_pos - nav_pos
            dist_horiz = float(np.linalg.norm(to_wp_orig[:2]))
            speed      = float(np.linalg.norm(vel))

            dist_3d     = float(np.linalg.norm(to_wp_orig))
            overshoot_r = max(p.capture_r * 10, self.turning_radius(speed) * 0.5)
            vel_toward  = float(np.dot(vel[:2], to_wp_orig[:2]))
            in_popup    = (phase == PHASE_TERMINAL and terminal_subphase == 'popup')
            in_dive     = (phase == PHASE_TERMINAL and terminal_subphase == 'dive')
            in_arh      = (phase == PHASE_TERMINAL and _terminal_type == 'arh')

            if in_dive or in_arh:
                passed = dist_3d < p.hit_radius
            else:
                passed = (dist_horiz < p.capture_r or
                          (not in_popup and vel_toward < 0 and dist_horiz < overshoot_r))
            if passed:
                logger.info("t=%7.1f s   [%8s] WP%d 通過  pos=(%.0f, %.0f, %.0f) m  dist_h=%.0fm  dist_3d=%.0fm",
                            t, phase, wp, pos[0], pos[1], pos[2], dist_horiz, dist_3d)
                wp += 1
                continue

            is_final_wp = (wp == len(waypoints) - 1)
            if is_final_wp and speed > 1.0 and not in_arh:
                braking_dist  = speed ** 2 / (2.0 * p.max_accel)
                desired_speed = (max(10.0, p.max_speed * dist_horiz / (braking_dist * 2.0))
                                 if dist_horiz < braking_dist * 2.0 else p.max_speed)
            else:
                desired_speed = p.max_speed

            floor_z    = terrain_floor(nav_pos, vel, speed)
            time_to_go = dist_horiz / speed if speed > 1.0 else float('inf')

            if phase == PHASE_LAUNCH and nav_pos[2] >= floor_z:
                phase = PHASE_CRUISE
                logger.info("t=%7.1f s   [巡航] 移行  高度 %.0fm MSL  クリアランス %.0fm",
                            t, nav_pos[2], nav_pos[2] - terrain_height_at(*local_to_geo(nav_pos)))

            if phase == PHASE_CRUISE and is_final_wp and time_to_go < terminal_time:
                phase = PHASE_TERMINAL
                if _terminal_type == 'popup_dive':
                    popup_z         = wp_pos[2] + p.popup_dive.popup_height
                    angle_rad       = abs(np.radians(p.popup_dive.dive_angle_deg))
                    dive_horiz_dist = (p.popup_dive.popup_height / np.tan(angle_rad)
                                       if angle_rad > 1e-6 else 0.0)
                    hdir_now        = to_wp_orig[:2] / dist_horiz if dist_horiz > 1e-9 else np.array([1.0, 0.0])
                    dive_start_xy   = wp_pos[:2] - hdir_now * dive_horiz_dist
                    dive_start      = np.array([dive_start_xy[0], dive_start_xy[1], popup_z])
                    at_popup        = nav_pos[2] >= popup_z * 0.98
                    close_enough    = dist_horiz <= dive_horiz_dist * 1.05
                    terminal_subphase = 'dive' if (at_popup and close_enough) else 'popup'
                    logger.info("t=%7.1f s   [終末/PopupDive] 移行  残り %.1fs  高度差 %+.0fm  "
                                "popup目標 %.0fm MSL  ダイブ開始距離 %.0fm → [%s]",
                                t, time_to_go, wp_pos[2] - nav_pos[2],
                                popup_z, dive_horiz_dist, terminal_subphase)
                else:
                    logger.info("t=%7.1f s   [終末/ARH] 移行  残り %.1fs  目標まで %.0fm  "
                                "高度 %.0fm MSL  N=%.1f",
                                t, time_to_go, dist_horiz, nav_pos[2], p.arh.nav_constant)

            # ── ガイダンス ──────────────────────────────────────────────────────
            horiz_dir = to_wp_orig[:2] / dist_horiz

            if phase == PHASE_LAUNCH:
                vz_desired  = float(np.clip(p.launch_climb_ratio * desired_speed,
                                            0.0, desired_speed * 0.95))
                horiz_spd   = float(np.sqrt(max(desired_speed**2 - vz_desired**2, 1.0)))
                desired_vel = np.array([horiz_dir[0]*horiz_spd,
                                        horiz_dir[1]*horiz_spd,
                                        vz_desired])

            elif phase == PHASE_TERMINAL:
                if _terminal_type == 'arh':
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
                    arh_debug_step += 1
                    if arh_debug_step <= 5 or dist_3d < 1000:
                        logger.info("ARH#%d t=%.1fs  pos=(%.0f,%.0f,%.1f)  vel=(%.1f,%.1f,%.2f)  "
                                    "aim=(%.0f,%.0f)  dv=(%.1f,%.1f,%.2f)  eta=%.2fs  dist3d=%.0f",
                                    arh_debug_step, t,
                                    pos[0], pos[1], pos[2], vel[0], vel[1], vel[2],
                                    wp_aim[0], wp_aim[1],
                                    desired_vel[0], desired_vel[1], desired_vel[2], eta, dist_3d)
                else:
                    dive_el_abs = abs(np.radians(p.popup_dive.dive_angle_deg))
                    _aim_spd    = (desired_speed * np.cos(dive_el_abs)
                                   if terminal_subphase == 'dive' else desired_speed)
                    wp_pos_aim  = _intercept_pos(waypoints[wp], t, nav_pos[:2], _aim_spd)
                    to_aim      = wp_pos_aim[:2] - nav_pos[:2]
                    aim_dist    = float(np.linalg.norm(to_aim))
                    aim_dir     = to_aim / aim_dist if aim_dist > 1e-9 else horiz_dir

                    if terminal_subphase == 'popup':
                        if dive_horiz_dist is not None and aim_dist > 1e-9:
                            dive_start[:2] = wp_pos_aim[:2] - aim_dir * dive_horiz_dist
                        to_ds      = dive_start[:2] - nav_pos[:2]
                        dist_to_ds = float(np.linalg.norm(to_ds))
                        ds_dir     = to_ds / dist_to_ds if dist_to_ds > 1.0 else aim_dir
                        vz_desired  = self._vz_command(popup_z - nav_pos[2], float(vel[2]), desired_speed)
                        horiz_spd   = float(np.sqrt(max(desired_speed**2 - vz_desired**2, 1.0)))
                        desired_vel = np.array([ds_dir[0]*horiz_spd, ds_dir[1]*horiz_spd, vz_desired])
                        at_popup    = nav_pos[2] >= popup_z * 0.98
                        near_start  = dist_to_ds <= max(dive_horiz_dist * 0.1, p.capture_r * 2)
                        if at_popup and near_start:
                            terminal_subphase = 'dive'
                            logger.info("t=%7.1f s   [終末] ダイブ開始  高度 %.0fm  目標まで水平 %.0fm  ダイブ角 %.0f°",
                                        t, nav_pos[2], dist_horiz, p.popup_dive.dive_angle_deg)
                    else:
                        dive_el     = np.radians(p.popup_dive.dive_angle_deg)
                        vz_dive     = desired_speed * np.sin(dive_el)
                        horiz_spd   = desired_speed * np.cos(dive_el_abs)
                        desired_vel = np.array([aim_dir[0]*horiz_spd, aim_dir[1]*horiz_spd, vz_dive])

            else:  # CRUISE
                if _profile == 'low':
                    target_z   = floor_z
                    valley     = self._valley_dir(nav_pos, speed, horiz_dir)
                    wp_pull    = float(np.clip(5000.0 / max(dist_horiz, 1.0), 0.30, 1.0))
                    blended    = wp_pull * horiz_dir + (1.0 - wp_pull) * valley
                    b_mag      = float(np.linalg.norm(blended))
                    cruise_dir = blended / b_mag if b_mag > 1e-9 else horiz_dir
                else:
                    target_z   = max(float(wp_pos[2]), floor_z)
                    cruise_dir = horiz_dir
                vz_desired  = self._vz_command(target_z - nav_pos[2], float(vel[2]), desired_speed)
                horiz_spd   = float(np.sqrt(max(desired_speed**2 - vz_desired**2, 1.0)))
                desired_vel = np.array([cruise_dir[0]*horiz_spd, cruise_dir[1]*horiz_spd, vz_desired])

            if phase != PHASE_TERMINAL or terminal_subphase == 'popup':
                desired_vel = self._avoidance_steer(nav_pos, desired_vel, desired_speed,
                                                    t, fixed_obs, moving_obs)

            # ── 加速度・物理積分 ────────────────────────────────────────────────
            dv     = desired_vel - vel
            dv_mag = float(np.linalg.norm(dv))
            if dv_mag > p.max_accel * p.dt:
                accel = (dv / dv_mag) * p.max_accel
            elif dv_mag > 1e-9:
                accel = dv / p.dt
            else:
                accel = np.zeros(3)

            # 対気速度（風に乗った気塊に対する相対速度）で抗力を計算
            lat_p, lon_p = local_to_geo(pos)
            wind        = _wind_field.wind_enu(lat_p, lon_p, float(pos[2]))
            vel_air     = vel - wind
            speed_air   = float(np.linalg.norm(vel_air))
            rho_ratio   = _air_density_ratio(float(pos[2]))
            drag_accel  = (-(p.drag_k * rho_ratio * speed_air**2)
                           * (vel_air / speed_air)) if speed_air > 0.1 else np.zeros(3)
            total_accel = accel + drag_accel

            vel += total_accel * p.dt
            spd  = float(np.linalg.norm(vel))
            if spd > p.max_speed:
                vel *= p.max_speed / spd
            pos += vel * p.dt
            t   += p.dt

            nav_pos = sensor.update(pos, vel, total_accel, p.dt)

            if phase != PHASE_LAUNCH:
                lat, lon = local_to_geo(pos)
                ground_z = terrain_height_at(lat, lon)
                if pos[2] <= ground_z:
                    pos[2] = ground_z
                    final_wp_pos    = _wp_pos(waypoints[-1], t)
                    horiz_to_final  = float(np.linalg.norm(pos[:2] - final_wp_pos[:2]))
                    dist3d_to_final = float(np.linalg.norm(pos - final_wp_pos))
                    is_hit = wp == len(waypoints) - 1 and (
                        (in_dive and horiz_to_final  < p.hit_radius * 3) or
                        (in_arh  and dist3d_to_final < p.hit_radius * 3)
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

    # ── 並列実行 ────────────────────────────────────────────────────────────────

    @staticmethod
    def run_parallel(runs: list, max_workers: int | None = None) -> list[dict]:
        """複数のシミュレーションを並列実行する。

        Parameters
        ----------
        runs : list of (waypoints,) or (waypoints, Simulator)
            各エントリは waypoints のみ、または (waypoints, Simulator) のタプル。
            Simulator を省略するとデフォルトパラメータで実行。
        max_workers : int | None
            スレッド数。None で CPU コア数に自動設定。

        Returns
        -------
        list[dict]
            入力順に対応した結果リスト。

        Examples
        --------
        results = Simulator.run_parallel([
            (wps_a,),
            (wps_b, Simulator(max_speed=250.0)),
        ])
        """
        def _run(item):
            if isinstance(item, (list, np.ndarray)):
                return Simulator().run(item)
            wps, *rest = item
            sim = rest[0] if rest else Simulator()
            return sim.run(wps)

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [pool.submit(_run, r) for r in runs]
            return [f.result() for f in futures]


# ── 後方互換ラッパー ──────────────────────────────────────────────────────────

def simulate(waypoints: list, profile: str | None = None) -> dict:
    """後方互換: Simulator().run() と同じ。"""
    return Simulator().run(waypoints, profile)


def turning_radius(speed: float) -> float:
    """後方互換: Simulator().turning_radius() と同じ。"""
    return Simulator().turning_radius(speed)
