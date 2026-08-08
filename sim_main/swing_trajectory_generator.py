import numpy as np
import math
import matplotlib.pyplot as plt

def compute_bezier_with_velocity(s, control_points, T_swing=0.3):
    """
    임의의 제어점을 받아 베지에 곡선의 위치(p_ref)와 
    해석적 미분 속도(v_ref)를 동시에 계산하는 함수
    """
    s = np.clip(s, 0.0, 1.0)
    pts = np.array(control_points)
    n = len(pts) - 1  # 곡선의 차수
    
    p_ref = np.zeros(3)
    dp_ds = np.zeros(3)
    
    # 1) 위치(Position) 계산
    for i in range(n + 1):
        coeff = math.comb(n, i) * ((1 - s) ** (n - i)) * (s ** i)
        p_ref += coeff * pts[i]
        
    # 2) 위상에 대한 미분 (dp / ds) 계산 (해석적 미분)
    n_sub = n - 1
    if n_sub >= 0:
        pts_diff = pts[1:] - pts[:-1]
        for i in range(n_sub + 1):
            coeff_diff = math.comb(n_sub, i) * ((1 - s) ** (n_sub - i)) * (s ** i)
            dp_ds += n * coeff_diff * pts_diff[i]
            
    # 3) 시간에 대한 해석적 미분 속도 (v_ref = (dp/ds) * (ds/dt))
    ds_dt = 1.0 / T_swing
    v_ref = dp_ds * ds_dt
    
    return p_ref, v_ref


if __name__ == "__main__":
    # 1. 가상의 4족 보행 발 스윙 제어점(Control Points) 설정 (3차 베지에 예시)
    start_pt = np.array([0.0, 0.1, 0.0])                     # p0: 디딤발 떼는 위치
    ctrl_pt1 = start_pt + np.array([0.00, 0.0, 0.1])         # p1: 위+앞으로 들어올림
    target_pt = np.array([0.2, 0.1, 0.0])                    # p3: 다음 디딜 목표 위치
    ctrl_pt2 = target_pt + np.array([0.00, 0.0, 0.1])       # p2: 목표 지점 위에서 내려오는 궤적 가중치

    control_points = [start_pt, ctrl_pt1, ctrl_pt2, target_pt]
    T_swing = 0.3  # 총 스윙 시간 [초]

    # 2. 진행률 s를 0.0부터 1.0까지 100개의 구간으로 쪼개어 곡선 좌표들 및 속도 계산
    s_samples = np.linspace(0.0, 1.0, 100)
    time_samples = s_samples * T_swing  # 시간 축으로 변환 (0 ~ T_swing)
    
    bezier_trajectory = []
    bezier_vel_trajectory = []

    for s in s_samples:
        point, velocity = compute_bezier_with_velocity(s, control_points, T_swing)
        bezier_trajectory.append(point)
        bezier_vel_trajectory.append(velocity)

    # 넘파이 배열로 변환 (Shape: 100, 3)
    bezier_trajectory = np.array(bezier_trajectory)
    bezier_vel_trajectory = np.array(bezier_vel_trajectory)

    # =========================================================================
    # [추가] 3. 2x1 서브플롯 생성 (위: 위치 궤적, 아래: 속도 궤적)
    # =========================================================================
    fig_profile, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # --- [상단] 베지에 위치 궤적 (Position) ---
    axes[0].plot(time_samples, bezier_trajectory[:, 0], label='X (Forward)', color='#E63946', linewidth=2)
    axes[0].plot(time_samples, bezier_trajectory[:, 1], label='Y (Lateral)', color='#457B9D', linewidth=2)
    axes[0].plot(time_samples, bezier_trajectory[:, 2], label='Z (Height)', color='#1D3557', linewidth=2)
    axes[0].set_title('Bezier Swing Trajectory - Position Profile', fontsize=13, fontweight='bold')
    axes[0].set_ylabel('Position [m]', fontsize=11)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc='upper right')

    # --- [하단] 베지에 해석적 미분 속도 궤적 (Velocity) ---
    axes[1].plot(time_samples, bezier_vel_trajectory[:, 0], label='Vel X', color='#E63946', linewidth=2, linestyle='--')
    axes[1].plot(time_samples, bezier_vel_trajectory[:, 1], label='Vel Y', color='#457B9D', linewidth=2, linestyle='--')
    axes[1].plot(time_samples, bezier_vel_trajectory[:, 2], label='Vel Z', color='#1D3557', linewidth=2, linestyle='--')
    axes[1].set_title('Bezier Swing Trajectory - Analytic Velocity Profile', fontsize=13, fontweight='bold')
    axes[1].set_xlabel(f'Time [s] (Swing Period = {T_swing}s)', fontsize=11)
    axes[1].set_ylabel('Velocity [m/s]', fontsize=11)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc='upper right')

    plt.tight_layout()

    # =========================================================================
    # 4. Matplotlib 3D 서브플롯을 이용한 입체 시각화 (기존 코드 유지)
    # =========================================================================
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 궤적 생성선 (파란색 실선)
    ax.plot(bezier_trajectory[:, 0], bezier_trajectory[:, 1], bezier_trajectory[:, 2], 
            label='Generated Bezier Path', color='#1D3557', linewidth=3, zorder=2)

    # 제어점(Control Points) 위치 플롯 (빨간색 점과 점선 연결)
    pts_arr = np.array(control_points)
    ax.plot(pts_arr[:, 0], pts_arr[:, 1], pts_arr[:, 2], 
            linestyle='--', color='#E63946', alpha=0.5, zorder=1)
    ax.scatter(pts_arr[:, 0], pts_arr[:, 1], pts_arr[:, 2], 
            color='#E63946', s=60, label='Control Points (p0~p3)', zorder=3)

    # 제어점마다 텍스트 라벨 달아주기 (p0, p1, p2, p3)
    for idx, pt in enumerate(control_points):
        ax.text(pt[0], pt[1], pt[2] + 0.005, f'p{idx}', fontsize=12, fontweight='bold', color='#E63946')

    # 그래프 디자인 가공
    ax.set_title('3D Bezier Swing Trajectory Verification', fontsize=14, fontweight='bold')
    ax.set_xlabel('X [m] (Forward)', fontsize=11)
    ax.set_ylabel('Y [m] (Lateral)', fontsize=11)
    ax.set_zlabel('Z [m] (Height)', fontsize=11)

    ax.set_zlim(0.0, 0.2)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    # 뷰 각도 조절
    ax.view_init(elev=20, azim=-60)

    plt.tight_layout()
    plt.show()