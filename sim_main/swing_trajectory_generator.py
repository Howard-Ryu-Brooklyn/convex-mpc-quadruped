import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb  # math.comb 대신 넘파이 배열 호환용으로 쓰거나 원본 함수 그대로 사용

# [유저 제공 함수]
def compute_bezier(s, control_points):
    pts = np.array(control_points)
    n = len(pts) - 1
    p_s = np.zeros(3)
    for i in range(n + 1):
        coeff = math.comb(n, i) * ((1 - s) ** (n - i)) * (s ** i)
        p_s += coeff * pts[i]
    return p_s

if __name__ == "__main__":

    # 1. 가상의 4족 보행 발 스윙 제어점(Control Points) 설정 (3차 베지에 예시)
    # 순서: [시작점, 이륙 가중점, 착지 진입 가중점, 최종 목표점]
    start_pt = np.array([0.0, 0.1, 0.0])                     # p0: 디딤발 떼는 위치
    ctrl_pt1 = start_pt + np.array([0.00, 0.0, 0.1])         # p1: 위+앞으로 들어올림 (높이 8cm 보정)
    target_pt = np.array([0.2, 0.1, 0.0])                    # p3: 다음 디딜 목표 위치
    ctrl_pt2 = target_pt + np.array([0.00, 0.0, 0.1])       # p2: 목표 지점 위에서 내려오는 궤적 가중치

    control_points = [start_pt, ctrl_pt1, ctrl_pt2, target_pt]

    # 2. 진행률 s를 0.0부터 1.0까지 100개의 구간으로 쪼개어 곡선 좌표들 계산
    s_samples = np.linspace(0.0, 1.0, 100)
    bezier_trajectory = []

    for s in s_samples:
        point = compute_bezier(s, control_points)
        bezier_trajectory.append(point)

    # 인덱싱을 편하게 하기 위해 2차원 넘파이 배열로 변환 (Shape: 100, 3)
    bezier_trajectory = np.array(bezier_trajectory)

    # 3. Matplotlib 3D 서브플롯을 이용한 입체 시각화
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

    # 로봇 다리가 아래로 뻗으므로 Z축 범위를 시각적으로 이쁘게 고정
    ax.set_zlim(0.0, 0.2)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    # 뷰 각도 조절 (보기 편한 대각선 시점)
    ax.view_init(elev=20, azim=-60)

    plt.tight_layout()
    plt.show()
