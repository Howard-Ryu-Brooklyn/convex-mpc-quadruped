"""스윙 궤적 프로파일 시각화 (수동 실행용 스크립트).

원래 sim_main/swing_trajectory_generator.py 였다. 유일한 실제 로직이던
compute_bezier_with_kinematics 가 sim_main/bezier.py 로 옮겨가면서
남은 것은 데모 플롯뿐이라 scripts/ 로 이동했다.

실행: python scripts/plot_swing_profile.py
"""

import numpy as np
import matplotlib.pyplot as plt

from quadruped_mpc.core.bezier import bezier_with_derivatives

def main():
    # 1. 가상의 4족 보행 발 스윙 제어점(Control Points) 설정 (3차 베지에 예시)
    start_pt = np.array([0.0, 0.1, 0.0])                     # p0: 디딤발 떼는 위치
    ctrl_pt1 = start_pt + np.array([0.00, 0.0, 0.1])         # p1: 위+앞으로 들어올림
    target_pt = np.array([0.0, 0.1, 0.0])                    # p3: 다음 디딜 목표 위치
    ctrl_pt2 = target_pt + np.array([0.00, 0.0, 0.1])       # p2: 목표 지점 위에서 내려오는 궤적 가중치

    control_points = [start_pt, ctrl_pt1, ctrl_pt2, target_pt]
    T_swing = 0.3  # 총 스윙 시간 [초]

    # 2. 진행률 s를 0.0부터 1.0까지 100개의 구간으로 쪼개어 계산
    s_samples = np.linspace(0.0, 1.0, 100)
    time_samples = s_samples * T_swing  # 시간 축으로 변환 (0 ~ T_swing)
    
    bezier_trajectory = []
    bezier_vel_trajectory = []
    bezier_acc_trajectory = []

    for s in s_samples:
        point, velocity, acceleration = bezier_with_derivatives(s, control_points, T_swing)
        bezier_trajectory.append(point)
        bezier_vel_trajectory.append(velocity)
        bezier_acc_trajectory.append(acceleration)

    # 넘파이 배열로 변환 (Shape: 100, 3)
    bezier_trajectory = np.array(bezier_trajectory)
    bezier_vel_trajectory = np.array(bezier_vel_trajectory)
    bezier_acc_trajectory = np.array(bezier_acc_trajectory)

    # =========================================================================
    # 3. 3x1 서브플롯 생성 (위: 위치, 중간: 속도, 아래: 가속도)
    # =========================================================================
    fig_profile, axes = plt.subplots(3, 1, figsize=(10, 11), sharex=True)

    # --- [상단] 베지에 위치 궤적 (Position) ---
    axes[0].plot(time_samples, bezier_trajectory[:, 0], label='X (Forward)', color='#E63946', linewidth=2)
    axes[0].plot(time_samples, bezier_trajectory[:, 1], label='Y (Lateral)', color='#457B9D', linewidth=2)
    axes[0].plot(time_samples, bezier_trajectory[:, 2], label='Z (Height)', color='#1D3557', linewidth=2)
    axes[0].set_title('Bezier Swing Trajectory - Position Profile', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Position [m]', fontsize=10)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc='upper right')

    # --- [중간] 베지에 해석적 미분 속도 궤적 (Velocity) ---
    axes[1].plot(time_samples, bezier_vel_trajectory[:, 0], label='Vel X', color='#E63946', linewidth=2, linestyle='--')
    axes[1].plot(time_samples, bezier_vel_trajectory[:, 1], label='Vel Y', color='#457B9D', linewidth=2, linestyle='--')
    axes[1].plot(time_samples, bezier_vel_trajectory[:, 2], label='Vel Z', color='#1D3557', linewidth=2, linestyle='--')
    axes[1].set_title('Bezier Swing Trajectory - Analytic Velocity Profile', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Velocity [m/s]', fontsize=10)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc='upper right')

    # --- [하단] 베지에 해석적 미분 가속도 궤적 (Acceleration) ---
    axes[2].plot(time_samples, bezier_acc_trajectory[:, 0], label='Acc X', color='#E63946', linewidth=2, linestyle=':')
    axes[2].plot(time_samples, bezier_acc_trajectory[:, 1], label='Acc Y', color='#457B9D', linewidth=2, linestyle=':')
    axes[2].plot(time_samples, bezier_acc_trajectory[:, 2], label='Acc Z', color='#1D3557', linewidth=2, linestyle=':')
    axes[2].set_title('Bezier Swing Trajectory - Analytic Acceleration Profile', fontsize=12, fontweight='bold')
    axes[2].set_xlabel(f'Time [s] (Swing Period = {T_swing}s)', fontsize=10)
    axes[2].set_ylabel('Acceleration [m/s^2]', fontsize=10)
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(loc='upper right')

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

if __name__ == "__main__":
    main()
