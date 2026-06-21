import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from config import MAX_ACCEL, MAX_SPEED, PLOT_3D_ONLY, TERRAIN_CLEARANCE, FLIGHT_PROFILE
from terrain import terrain_height_at
from geo import local_to_geo
from obstacles import get_fixed_obstacles, get_moving_obstacles, SphereObstacle, BoxObstacle

# 日本語フォント設定（Windows: Yu Gothic / Meiryo / MS Gothic）
for _font in ['Yu Gothic', 'Meiryo', 'MS Gothic', 'IPAexGothic']:
    if _font in [f.name for f in matplotlib.font_manager.fontManager.ttflist]:
        matplotlib.rcParams['font.family'] = _font
        break


def plot(hist: dict, waypoints: np.ndarray, path: str = 'flight_route.png', show: bool = True) -> None:
    """シミュレーション結果を可視化する。"""
    pos        = hist['pos']
    speeds     = hist['speed']
    times      = hist['time']
    accels     = hist['accel']
    accel_mags = np.linalg.norm(accels, axis=1)
    elevation  = hist['elevation']
    azimuth    = hist['azimuth']

    norm = plt.Normalize(vmin=0, vmax=MAX_SPEED)
    fixed_obs  = get_fixed_obstacles()
    moving_obs = get_moving_obstacles()

    if PLOT_3D_ONLY:
        fig = plt.figure(figsize=(10, 8))
        ax1 = fig.add_subplot(1, 1, 1, projection='3d')
    else:
        fig = plt.figure(figsize=(18, 10))
        ax1 = fig.add_subplot(2, 3, 1, projection='3d')

    # ── 3D 飛翔軌道（速度で色付け）──────────────────────────────────────────
    pts  = pos.reshape(-1, 1, 3)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    lc   = Line3DCollection(segs, cmap='plasma', norm=norm, linewidth=1.5)
    lc.set_array(speeds[:-1])
    ax1.add_collection(lc)

    ax1.scatter(*waypoints.T, c='red', s=100, marker='^', zorder=5)
    wp_labels = (['出発点']
                 + [f'経由点{i}' for i in range(1, len(waypoints) - 1)]
                 + ['目的地'])
    for wp, lbl in zip(waypoints, wp_labels):
        ax1.text(wp[0] + 500, wp[1] + 500, wp[2] + 100, lbl, fontsize=8)

    # 固定障害物（3D: 球はワイヤーフレーム球 / 箱は12辺ワイヤーフレーム）
    theta = np.linspace(0, 2 * np.pi, 60)
    for obs in fixed_obs:
        ax1.scatter(*obs.pos, c='red', s=120, marker='x', zorder=6, depthshade=False)
        ax1.text(obs.pos[0], obs.pos[1], obs.pos[2] + 200, obs.label,
                 color='red', fontsize=7)
        if isinstance(obs, SphereObstacle):
            # 緯度リング × 3 + 経線リング × 2 でワイヤーフレーム球を表現
            r = obs.radius
            for lat_deg in (-30, 0, 30):
                lat = np.radians(lat_deg)
                r_xy = r * np.cos(lat)
                z_off = r * np.sin(lat)
                ax1.plot(obs.pos[0] + r_xy * np.cos(theta),
                         obs.pos[1] + r_xy * np.sin(theta),
                         np.full_like(theta, obs.pos[2] + z_off),
                         'r-', alpha=0.25, lw=0.7)
            phi = np.linspace(0, 2 * np.pi, 60)
            for lon_deg in (0, 90):
                lon = np.radians(lon_deg)
                ax1.plot(obs.pos[0] + r * np.cos(phi) * np.cos(lon),
                         obs.pos[1] + r * np.cos(phi) * np.sin(lon),
                         obs.pos[2] + r * np.sin(phi),
                         'r-', alpha=0.25, lw=0.7)
        elif isinstance(obs, BoxObstacle):
            # 直方体の12辺を描画
            c, h = obs.pos, obs.half_extents
            corners = np.array([
                [c[0]-h[0], c[1]-h[1], c[2]-h[2]],
                [c[0]+h[0], c[1]-h[1], c[2]-h[2]],
                [c[0]+h[0], c[1]+h[1], c[2]-h[2]],
                [c[0]-h[0], c[1]+h[1], c[2]-h[2]],
                [c[0]-h[0], c[1]-h[1], c[2]+h[2]],
                [c[0]+h[0], c[1]-h[1], c[2]+h[2]],
                [c[0]+h[0], c[1]+h[1], c[2]+h[2]],
                [c[0]-h[0], c[1]+h[1], c[2]+h[2]],
            ])
            edges = [(0,1),(1,2),(2,3),(3,0),
                     (4,5),(5,6),(6,7),(7,4),
                     (0,4),(1,5),(2,6),(3,7)]
            for i, j in edges:
                ax1.plot(*zip(corners[i], corners[j]),
                         color='crimson', alpha=0.5, lw=1.0)

    # 移動体（3D: 橙のマーカー + 軌跡矢印）
    for obs in moving_obs:
        p0 = obs.pos_init
        pf = obs.pos_at(float(times[-1]))
        ax1.scatter(*p0, c='orange', s=100, marker='D', zorder=6, depthshade=False)
        ax1.quiver(p0[0], p0[1], p0[2],
                   pf[0]-p0[0], pf[1]-p0[1], pf[2]-p0[2],
                   color='orange', alpha=0.6, arrow_length_ratio=0.1)
        ax1.text(p0[0], p0[1], p0[2] + 200, obs.label, color='darkorange', fontsize=7)

    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('高度 (m)')
    ax1.set_title('3D 飛翔軌道')
    plt.colorbar(lc, ax=ax1, label='速度 (m/s)', shrink=0.55, pad=0.15)

    if not PLOT_3D_ONLY:
        # ── 速度 vs 時刻 ─────────────────────────────────────────────────────
        ax2 = fig.add_subplot(2, 3, 2)
        ax2.plot(times, speeds, lw=1.2, color='royalblue', label='速度')
        ax2.axhline(MAX_SPEED, color='red', ls='--', alpha=0.7,
                    label=f'Max {MAX_SPEED} m/s')
        ax2.set(xlabel='時刻 (s)', ylabel='速度 (m/s)', title='速度 vs 時刻')
        ax2.legend()
        ax2.grid(alpha=0.3)

        # ── 高度 vs 時刻（地形プロファイル + フェーズ帯）────────────────────
        ax3 = fig.add_subplot(2, 3, 3)
        terrain_prof = np.array([terrain_height_at(*local_to_geo(p)) for p in pos])

        # フェーズ背景色
        phases = hist.get('phase', np.full(len(times), 'cruise'))
        phase_colors = {'launch': '#fff3cd', 'cruise': '#ffffff', 'terminal': '#fde8e8'}
        prev_ph, t_start = phases[0], times[0]
        for i in range(1, len(times)):
            if phases[i] != prev_ph or i == len(times) - 1:
                ax3.axvspan(t_start, times[i], color=phase_colors.get(prev_ph, '#fff'),
                            alpha=0.6, zorder=0)
                prev_ph, t_start = phases[i], times[i]

        ax3.fill_between(times, terrain_prof, alpha=0.35, color='sienna', label='地形')
        ax3.plot(times, terrain_prof + TERRAIN_CLEARANCE,
                 lw=0.8, ls='--', color='red', alpha=0.6,
                 label=f'クリアランス +{TERRAIN_CLEARANCE:.0f}m')
        ax3.plot(times, pos[:, 2], lw=1.2, color='forestgreen', label='飛翔高度 MSL')
        if hist.get('hit_ground'):
            ax3.axvline(times[-1], color='red', lw=1.5, ls=':', alpha=0.8,
                        label='地面衝突')
            ax3.scatter(times[-1], pos[-1, 2], c='red', s=80, zorder=6)

        # 巡航フェーズの平均高度・AGL統計
        cruise_mask = phases == 'cruise'
        if cruise_mask.any():
            agl_prof   = pos[:, 2] - terrain_prof
            mean_msl   = float(pos[cruise_mask, 2].mean())
            mean_agl   = float(agl_prof[cruise_mask].mean())
            min_agl    = float(agl_prof[cruise_mask].min())
            ax3.axhline(mean_msl, color='forestgreen', ls=':', lw=1.2, alpha=0.7)
            ax3.text(times[cruise_mask][0] + (times[-1] - times[cruise_mask][0]) * 0.02,
                     mean_msl,
                     f'巡航平均 {mean_msl:.0f}m MSL  /  AGL平均 {mean_agl:.0f}m  最低 {min_agl:.0f}m',
                     fontsize=7, color='forestgreen', va='bottom')

        # フェーズ凡例
        from matplotlib.patches import Patch
        ax3.legend(handles=[
            *ax3.get_legend_handles_labels()[0],
            Patch(color='#fff3cd', alpha=0.8, label='離陸'),
            Patch(color='#ffffff', alpha=0.8, label='巡航', ec='gray', lw=0.5),
            Patch(color='#fde8e8', alpha=0.8, label='終末誘導'),
        ], fontsize=7)
        ax3.set(xlabel='時刻 (s)', ylabel='高度 MSL (m)', title='高度 vs 時刻')
        ax3.grid(alpha=0.3)

        # ── 加速度 vs 時刻 ───────────────────────────────────────────────────
        ax4 = fig.add_subplot(2, 3, 4)
        ax4.plot(times, accel_mags, lw=1.2, color='darkorange', label='加速度')
        ax4.axhline(MAX_ACCEL, color='red', ls='--', alpha=0.7,
                    label=f'Max {MAX_ACCEL} m/s^2')
        ax4.set(xlabel='時刻 (s)', ylabel='加速度 (m/s²)', title='加速度大きさ vs 時刻')
        ax4.legend()
        ax4.grid(alpha=0.3)

        # ── 姿勢 vs 時刻（仰角・方角）────────────────────────────────────────
        ax5 = fig.add_subplot(2, 3, 5)
        ax5.plot(times, elevation, lw=1.2, color='purple', label='仰角')
        ax5.axhline(0, color='gray', ls=':', alpha=0.5)
        ax5.set(xlabel='時刻 (s)', ylabel='仰角 (度)', title='姿勢 vs 時刻')
        ax5.grid(alpha=0.3)

        ax5b = ax5.twinx()
        ax5b.plot(times, azimuth, lw=1.2, color='teal', ls='--', label='方角')
        ax5b.set_ylabel('方角 (度)')

        lines  = ax5.get_lines() + ax5b.get_lines()
        labels = [str(l.get_label()) for l in lines]
        ax5.legend(lines, labels, loc='upper right', fontsize=8)

        # ── 上面図（XY 平面、障害物ゾーン可視化）────────────────────────────
        ax6 = fig.add_subplot(2, 3, 6)
        ax6.plot(pos[:, 0], pos[:, 1], lw=1.2, color='royalblue', label='飛翔軌道')
        ax6.scatter(waypoints[:, 0], waypoints[:, 1],
                    c='red', s=80, marker='^', zorder=5)
        for wp, lbl in zip(waypoints, wp_labels):
            ax6.text(wp[0] + 200, wp[1] + 200, lbl, fontsize=7)

        for obs in fixed_obs:
            ax6.scatter(obs.pos[0], obs.pos[1], c='red', s=80, marker='x', zorder=6)
            if isinstance(obs, SphereObstacle):
                ax6.add_patch(mpatches.Circle(
                    (obs.pos[0], obs.pos[1]), obs.zone,
                    color='red', alpha=0.10, fill=True, zorder=2))
                ax6.add_patch(mpatches.Circle(
                    (obs.pos[0], obs.pos[1]), obs.radius,
                    color='red', alpha=0.30, fill=True, zorder=3))
                ax6.text(obs.pos[0], obs.pos[1] + obs.zone * 0.1,
                         obs.label, color='red', fontsize=7)
            elif isinstance(obs, BoxObstacle):
                hx, hy = obs.half_extents[0], obs.half_extents[1]
                zone_r  = obs.zone
                ax6.add_patch(mpatches.Circle(
                    (obs.pos[0], obs.pos[1]), zone_r,
                    color='crimson', alpha=0.08, fill=True, zorder=2))
                ax6.add_patch(mpatches.Rectangle(
                    (obs.pos[0] - hx, obs.pos[1] - hy), 2*hx, 2*hy,
                    color='crimson', alpha=0.35, fill=True, zorder=3))
                ax6.text(obs.pos[0], obs.pos[1] + zone_r * 0.1,
                         obs.label, color='crimson', fontsize=7)

        for obs in moving_obs:
            p0 = obs.pos_init
            pf = obs.pos_at(float(times[-1]))
            for t_show in np.linspace(0, float(times[-1]), 4):
                p = obs.pos_at(t_show)
                ax6.add_patch(mpatches.Circle(
                    (p[0], p[1]), obs.zone,
                    color='orange', alpha=0.08, fill=True, zorder=2))
            ax6.annotate('', xy=(pf[0], pf[1]), xytext=(p0[0], p0[1]),
                         arrowprops=dict(arrowstyle='->', color='orange', lw=1.5))
            ax6.scatter(p0[0], p0[1], c='orange', s=80, marker='D', zorder=6)
            ax6.text(p0[0], p0[1] + obs.zone * 0.1,
                     obs.label, color='darkorange', fontsize=7)

        ax6.set_aspect('equal')
        ax6.set(xlabel='X (m)', ylabel='Y (m)', title='上面図（障害物ゾーン）')
        ax6.grid(alpha=0.3)

    _profile_label = hist.get('profile_used', FLIGHT_PROFILE).upper()
    if FLIGHT_PROFILE.lower() == 'auto':
        _profile_label = f'AUTO→{_profile_label}'
    plt.suptitle(f'飛翔ルートシミュレーション  [{_profile_label}]',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches='tight')
    print(f"保存: {path}")
    if show:
        plt.show()
    plt.close(fig)
