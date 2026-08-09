import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
import mpl_toolkits.mplot3d.art3d as art3d

import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.art3d as art3d
import matplotlib.animation as animation
from IPython.display import HTML

# 인자에 history_p 대신 history_x를 추가합니다.
def animate_quadruped(history_x, history_R, history_r_feet, history_F_G, body_length, body_width, dt=0.033, save=False, history_q=None, link_lengths=(0.1, 0.34, 0.34)):
    """
    4족보행 로봇의 시뮬레이션 결과를 3D 애니메이션으로 생성합니다.
    (history_x에서 위치 p 데이터를 실시간으로 추출하여 사용합니다.)
    """
    print("🎥 3D 시각화 애니메이션 렌더링 시작...")
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 초기 3D 그래프 축 범위 설정 (나중에 update 함수에서 계속 바뀜)
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')
    
    # 지면 그리기 (로봇이 멀리 갈 수 있으므로 지면을 충분히 크게 50m 크기로 확장)
    square = np.array([[-50, -50, 0], [50, -50, 0], [50, 50, 0], [-50, 50, 0]])
    ground = art3d.Poly3DCollection([square], alpha=0.1, facecolor='gray')
    ax.add_collection3d(ground)
    
    # 로봇의 몸체와 4개의 다리를 그릴 객체 초기화
    body_plot, = ax.plot([], [], [], 'k-', linewidth=3)
    leg_plots = [ax.plot([], [], [], 'o-', linewidth=3, markersize=5)[0] for _ in range(4)]
        
    grf_quivers = [None for _ in range(4)]

    # 다리 색상: FR(빨강), FL(파랑), RR(초록), RL(보라)
    colors = ['r', 'b', 'g', 'm']
    for i, line in enumerate(leg_plots):
        line.set_color(colors[i])

    L, W = body_length / 2, body_width / 2
    body_local = np.array([
        [ L,  L, -L, -L], # X 좌표 (FR, FL, RR, RL)
        [-W,  W, -W,  W], # Y 좌표 (FR, FL, RR, RL)
        [ 0,  0,  0,  0]  # Z 좌표
    ])

    draw_order = [0, 1, 3, 2, 0]

    def update(frame):
        # ----------------------------------------------------
        # [핵심 수정] history_x에서 인덱스 3, 4, 5 (X, Y, Z)를 추출
        # ----------------------------------------------------
        x_state = history_x[frame]
        p = x_state[3:6].reshape(3, 1) # 열 벡터로 형태 맞춤
        
        # ====================================================
        # [카메라 추종 기능 추가] 
        # 로봇의 현재 위치(p)를 중심으로 화면 범위를 계속 갱신합니다.
        # ====================================================
        view_window = 1.0  # 로봇 중심으로 앞뒤좌우 1.0m씩 보여줌 (시야 크기 조절 가능)
        ax.set_xlim(p[0, 0] - view_window, p[0, 0] + view_window)
        ax.set_ylim(p[1, 0] - view_window, p[1, 0] + view_window)
        ax.set_zlim(0, 1.2) # Z축은 높이 변화를 명확히 보기 위해 0부터 고정 유지
        # ====================================================

        R = history_R[frame] 
        feet_offset = history_r_feet[frame] 
        F_G = history_F_G[frame] 

        body_world = R @ body_local + p
        feet_world = feet_offset + p 

        # 몸체 그리기
        body_draw = body_world[:, draw_order]
        body_plot.set_data(body_draw[0, :], body_draw[1, :])
        body_plot.set_3d_properties(body_draw[2, :])
        
        # 다리 및 GRF 그리기
        for i in range(4):
            # history_q는 (N, 3, 4) 크기라고 가정 [q_abad, q_hip, q_knee] * 4
            if history_q is not None:
                q_leg = history_q[frame, :, :]
                q_abad, q_hip, q_knee = q_leg[0,i], q_leg[1,i], q_leg[2,i]
                
                # 왼쪽 다리(FL, RL)는 +Y 방향, 오른쪽 다리(FR, RR)는 -Y 방향으로 Ab/Ad 오프셋 적용
                sign_y = 1 if i in [1, 3] else -1
                l_abad, l_thigh, l_calf = link_lengths
                
                # 조인트 회전 행렬
                Rx = np.array([[1, 0, 0], [0, np.cos(q_abad), -np.sin(q_abad)], [0, np.sin(q_abad), np.cos(q_abad)]])
                Ry_hip = np.array([[np.cos(q_hip), 0, np.sin(q_hip)], [0, 1, 0], [-np.sin(q_hip), 0, np.cos(q_hip)]])
                Ry_knee = np.array([[np.cos(q_hip+q_knee), 0, np.sin(q_hip+q_knee)], [0, 1, 0], [-np.sin(q_hip+q_knee), 0, np.cos(q_hip+q_knee)]])
                
                p0 = body_world[:, i] # 어깨 조인트
                p1 = p0 + R @ Rx @ np.array([0, sign_y * l_abad, 0])             # Ab/Ad 끝
                p2 = p1 + R @ Rx @ Ry_hip @ np.array([0, 0, -l_thigh])           # 무릎
                p3 = p2 + R @ Rx @ Ry_knee @ np.array([0, 0, -l_calf])           # 발끝
                
                leg_x = [p0[0], p1[0], p2[0], p3[0]]
                leg_y = [p0[1], p1[1], p2[1], p3[1]]
                leg_z = [p0[2], p1[2], p2[2], p3[2]]
                
                leg_plots[i].set_data(leg_x, leg_y)
                leg_plots[i].set_3d_properties(leg_z)
                
                # GRF 렌더링을 위해 p3(발끝) 좌표 사용
                foot_pos = np.array([p3[0], p3[1], p3[2]])
            else:
                # 관절 데이터가 없으면 단순 직선 연결 (디버깅용)
                leg_x = [body_world[0, i], feet_world[0, i]]
                leg_y = [body_world[1, i], feet_world[1, i]]
                leg_z = [body_world[2, i], feet_world[2, i]]
                leg_plots[i].set_data(leg_x, leg_y)
                leg_plots[i].set_3d_properties(leg_z)
                foot_pos = feet_world[:, i]
            
            # [GRF 화살표 업데이트] 
            if grf_quivers[i] is not None:
                try: grf_quivers[i].remove()
                except: pass
                grf_quivers[i] = None
                
            force_vec = F_G[:, i] 

            if np.linalg.norm(force_vec) > 1e-3:
                scale = 0.0005
                fv_scaled = force_vec * scale
                grf_quivers[i] = ax.quiver(foot_pos[0], foot_pos[1], foot_pos[2],
                                          fv_scaled[0], fv_scaled[1], fv_scaled[2],
                                          pivot='tail', color='gray', linewidth=2, arrow_length_ratio=0.3)
                
        valid_quivers = [q for q in grf_quivers if q is not None]
        return [body_plot] + leg_plots + valid_quivers

    # 프레임 개수 기준을 history_x로 변경
    ani = animation.FuncAnimation(fig, update, frames=len(history_x), interval=dt*1000, blit=False)
    plt.close(fig)
    print("✅ 렌더링 완료!")

    if save:
        print("🎬 동영상 파일로 저장 중입니다...")
        ani.save('quadruped_mpc.mp4', writer='ffmpeg', fps=int(1/dt), dpi=200)
        print("✅ 저장 완료! 'quadruped_mpc.mp4' 파일을 확인하세요.")
        
    return HTML(ani.to_jshtml())

def plot_state_tracking(history_x, history_xref, dt):
    """
    MPC의 12차원 상태 변수 추종 성능을 시각화하는 함수
    Row 1: 자세 (Roll, Pitch, Yaw)
    Row 2: 위치 (X, Y, Z)
    Row 3: 각속도 (Wx, Wy, Wz)
    Row 4: 선속도 (Vx, Vy, Vz)
    """
    # NumPy 배열로 변환 (N x 13)
    hx = np.array(history_x)
    href = np.array(history_xref)
    
    # 시간 축 생성
    time = np.arange(hx.shape[0]) * dt
    
    # 그래프 제목 및 Y축 라벨 설정
    state_names = [
        'Roll (rad)', 'Pitch (rad)', 'Yaw (rad)',
        'Pos X (m)', 'Pos Y (m)', 'Pos Z (m)',
        'AngVel Wx (rad/s)', 'AngVel Wy (rad/s)', 'AngVel Wz (rad/s)',
        'LinVel Vx (m/s)', 'LinVel Vy (m/s)', 'LinVel Vz (m/s)'
    ]
    
    # 4행 3열짜리 커다란 도화지 생성
    fig, axs = plt.subplots(4, 3, figsize=(11, 10))
    fig.suptitle('MPC State Tracking Performance', fontsize=18, fontweight='bold')
    
    # 12개의 상태에 대해 각각 그래프 그리기
    for i in range(12):
        row = i // 3
        col = i % 3
        ax = axs[row, col]
        
        # 실제 상태 (파란색 실선)
        ax.plot(time, hx[:, i], label='Actual', color='b', linewidth=2)
        # 목표 참조 궤적 (빨간색 점선)
        ax.plot(time, href[:, i], label='Reference', color='r', linestyle='--', linewidth=2)
        
        ax.set_title(state_names[i], fontsize=12)
        ax.grid(True, linestyle=':', alpha=0.7)
        
        # 맨 아랫줄에만 X축(시간) 라벨 표시
        if row == 3:
            ax.set_xlabel('Time (s)', fontsize=10)
            
        # 첫 번째 그래프에만 범례(Legend) 표시
        if i == 0:
            ax.legend(loc='upper right')

    # 그래프 간격 자동 조절 및 출력
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def plot_force_and_contact(history_F_G, history_Sa, sim_tf):
    """
    각 다리별 Z축 지면 반발력(Fz)과 접촉 상태(Sa)를 시각화합니다.
    - history_F_G: shape (N, 3, 4) -> 3은 x,y,z / 4는 leg index
    - history_Sa: shape (N, 4) -> 0: GROUND, 1: AIR
    """
    # Numpy 배열로 변환
    F_G_arr = np.array(history_F_G)
    Sa_arr = np.array(history_Sa)
    
    # 시간 배열 생성
    num_data = F_G_arr.shape[0]
    time = np.linspace(0, sim_tf, num_data)
    
    leg_names = ['FR (Front Right)', 'FL (Front Left)', 'RR (Rear Right)', 'RL (Rear Left)']
    
    # 4행 1열의 서브플롯 생성
    fig, axs = plt.subplots(4, 1, figsize=(11, 5), sharex=True)
    
    for i in range(4):
        ax1 = axs[i]
        
        # 1. Z축 지면 반발력 플롯 (파란색 실선)
        # F_G_arr[:, 2, i] -> 모든 시간에 대한, Z축(인덱스 2), i번째 다리
        fz_data = F_G_arr[:, 2, i]
        ax1.plot(time, fz_data, label='Fz (Force Z)', color='royalblue', linewidth=2)
        ax1.set_ylabel('Force (N)', color='royalblue', fontweight='bold')
        ax1.tick_params(axis='y', labelcolor='royalblue')
        ax1.set_title(leg_names[i], fontweight='bold')
        ax1.grid(True, linestyle='--', alpha=0.6)
        
        # 2. 접촉 상태(Sa) 플롯을 위한 두 번째 Y축 (빨간색 점선)
        ax2 = ax1.twinx()
        # Sa = 0 (Ground), 1 (Air)
        sa_data = Sa_arr[:, i]
        # 직관성을 위해 step 그래프로 그림 (상태 변화가 뚝뚝 끊기도록)
        ax2.step(time, sa_data, label='Sa (1=AIR, 0=GROUND)', color='crimson', linestyle='--', linewidth=2, where='post')
        ax2.set_ylabel('State (Sa)', color='crimson', fontweight='bold')
        ax2.tick_params(axis='y', labelcolor='crimson')
        
        # Y축 범위 고정 (Sa는 0과 1만 가지므로 보기 좋게 여백 추가)
        ax2.set_ylim(-0.2, 1.2)
        ax2.set_yticks([0, 1])
        ax2.set_yticklabels(['GROUND (0)', 'AIR (1)'])
        
        # 범례 합치기
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right')

    plt.xlabel('Time (s)', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.show()

def plot_leg_angles(history_q, sim_ts):
    """
    4족보행 로봇의 각 다리별 조인트 각도를 2x2 서브플롯으로 그려주는 함수 (시각적 구분 개선 버전)
    """
    # 데이터를 (N, 12) 형태의 Numpy 배열로 변환
    q_arr = np.array(history_q).reshape(-1, 12)
    
    N = q_arr.shape[0]
    time = np.arange(N) * sim_ts
    
    leg_names = ['Front Right (FR)', 'Front Left (FL)', 'Back Right (BR)', 'Back Left (BL)']
    
    # ─── 시각적 구분을 위한 스타일 설정 ───
    joint_colors = ['#E63946', '#2A9D8F', '#1D3557'] # 시인성이 높은 칼라 팰릿 (빨강, 청록, 네이비)
    joint_labels = ['Ab/Ad', 'Hip', 'Knee']
    
    # 선이 겹쳐도 보이도록 각 관절별 스타일을 다르게 설정
    line_styles = ['-', '--', ':']          # Ab/Ad: 실선, Hip: 파선, Knee: 점선
    line_widths = [1.5, 3.0, 4.5]           # Knee가 뒤에 그려지므로 두께를 가장 두껍게 설정
    alphas = [0.9, 0.75, 0.6]               # 겹친 선의 하단이 비치도록 투명도 차등 부여
    
    # 2x2 그래프 생성
    fig, axs = plt.subplots(2, 2, figsize=(11, 10))
    axs = axs.flatten()
    
    for i in range(4):
        ax = axs[i]
        
        # 각 다리당 3개의 관절 각도 추출 및 라디안 -> 디그리 변환
        rad_to_deg = 180 / np.pi
        q_abad = q_arr[:, i * 3] * rad_to_deg
        q_hip  = q_arr[:, i * 3 + 1] * rad_to_deg
        q_knee = q_arr[:, i * 3 + 2] * rad_to_deg
        
        # 순서 중요: 얇은 선이 굵은 선에 묻히지 않도록 가장 두꺼운 Knee(2)부터 빽으로 먼저 그림
        ax.plot(time, q_knee, label=joint_labels[2], color=joint_colors[2], 
                linestyle=line_styles[2], linewidth=line_widths[2], alpha=alphas[2])
        
        ax.plot(time, q_hip,  label=joint_labels[1], color=joint_colors[1], 
                linestyle=line_styles[1], linewidth=line_widths[1], alpha=alphas[1])
        
        ax.plot(time, q_abad, label=joint_labels[0], color=joint_colors[0], 
                linestyle=line_styles[0], linewidth=line_widths[0], alpha=alphas[0])
        
        # 데이터가 너무 듬성듬성하거나 경향성이 겹칠 때를 대비해, 
        # 특정 간격(예: 50스텝)마다 마커를 찍어 점선 무늬가 겹쳐도 마커로 구별되게 보완
        marker_interval = max(1, N // 20)  # 데이터 길이에 맞춰 유연하게 조절
        ax.plot(time[::marker_interval], q_abad[::marker_interval], color=joint_colors[0], marker='o', linestyle='None', markersize=5, alpha=0.8)
        ax.plot(time[::marker_interval], q_hip[::marker_interval],  color=joint_colors[1], marker='s', linestyle='None', markersize=5, alpha=0.8)
        ax.plot(time[::marker_interval], q_knee[::marker_interval], color=joint_colors[2], marker='^', linestyle='None', markersize=6, alpha=0.8)
        
        # 그래프 디자인 가공
        ax.set_title(f'Leg {i+1}: {leg_names[i]} Joint Angles', fontsize=14, fontweight='bold')
        ax.set_xlabel('Time [s]', fontsize=12)
        ax.set_ylabel('Angle [deg]', fontsize=12) # 라디안 가공에 맞춰 단위 변경
        ax.legend(loc='upper right', framealpha=0.9)
        ax.grid(True, linestyle='-', alpha=0.15) # 배경 그리드가 선을 가리지 않게 아주 연하게 세팅
        
    plt.tight_layout()
    plt.show()


def plot_foot_comparison(history_r_des, history_r_actual, dt, leg_idx=0):
    """
    목표 발 좌표(r_des)와 실제 발 좌표(r_actual)의 추종 성능을 비교하는 플롯
    
    :param history_r_des: (N, 3, 4) 형태의 목표 발 좌표 히스토리
    :param history_r_actual: (N, 3, 4) 형태의 실제 발 좌표 히스토리
    :param dt: 제어 및 시뮬레이션 주기 (타임스텝)
    :param leg_idx: 확인할 다리 인덱스 (0:FR, 1:FL, 2:BR, 3:BL)
    """
    # 넘파이 배열 변환 및 특정 다리(leg_idx)의 데이터만 추출 -> Shape: (N, 3)
    r_des_data = np.array(history_r_des)[:, :, leg_idx]
    r_act_data = np.array(history_r_actual)[:, :, leg_idx]
    
    N = r_des_data.shape[0]
    time = np.arange(N) * dt
    
    leg_names = ['Front Right (FR)', 'Front Left (FL)', 'Back Right (BR)', 'Back Left (BL)']
    axis_names = ['X Axis (Forward)', 'Y Axis (Lateral)', 'Z Axis (Height)']
    
    fig, axs = plt.subplots(3, 1, figsize=(11, 5), sharex=True)
    
    for i in range(3):
        ax = axs[i]
        
        # 💡 시각적 팁: 실제 데이터(실선)를 먼저 굵게 깔고, 목표 데이터(파선)를 위에 얹어 겹쳐도 둘 다 보이게 조절
        ax.plot(time, r_act_data[:, i], label='Actual Position (r)', 
                color='#1D3557', linewidth=2.5, alpha=0.9)
        
        ax.plot(time, r_des_data[:, i], label='Desired Position (r_des)', 
                color='#E63946', linestyle='--', linewidth=2.0, alpha=0.9)
        
        # 그래프 세부 디자인 가공
        ax.set_ylabel(f'{axis_names[i]} [m]', fontsize=11)
        ax.legend(loc='upper right', framealpha=0.9)
        ax.grid(True, linestyle='-', alpha=0.15)
        
    # 최하단 및 최상단 라벨링
    axs[2].set_xlabel('Time [s]', fontsize=12)
    axs[0].set_title(f'Foot Trajectory Tracking Performance: {leg_names[leg_idx]}', 
                     fontsize=14, fontweight='bold', pad=15)
    
    plt.tight_layout()
    plt.show()


def plot_r_feet_wf_over_time(history_r_feet_wf, dt):
    """
    월드 좌표계 발 위치(r_feet_wf)의 X, Y, Z 성분을 시간에 따라 각각 플롯합니다.
    :param history_r_feet_wf: (N, 3, 4) 형태의 배열 리스트 (N: 프레임 수, 3: XYZ, 4: 다리)
    :param dt: 제어 루프의 타임스텝 주기 (예: 0.001)
    """
    print("📊 r_feet_wf (월드 좌표계 발 위치) 성분별 그래프 생성 중...")
    
    # 데이터를 Numpy 배열로 변환 (N, 3, 4)
    r_data = np.array(history_r_feet_wf)
    N = r_data.shape[0]
    time = np.arange(N) * dt
    
    # 다리 이름 및 색상 설정
    leg_names = ['FR (Front Right)', 'FL (Front Left)', 'RR (Rear Right)', 'RL (Rear Left)']
    colors = ['r', 'b', 'g', 'm'] # 각 다리를 구분할 색상
    
    # 3행 1열의 서브플롯 생성 (X, Y, Z 각각)
    fig, axs = plt.subplots(3, 1, figsize=(11, 5), sharex=True)
    axis_labels = ['X Position [m] (Forward)', 'Y Position [m] (Lateral)', 'Z Position [m] (Height)']
    
    for ax_idx in range(3): # 0:X, 1:Y, 2:Z
        for leg_idx in range(4): # 0:FR, 1:FL, 2:RR, 3:RL
            # 각 다리의 특정 축(X, Y, Z) 데이터를 시간에 따라 플롯
            axs[ax_idx].plot(time, r_data[:, ax_idx, leg_idx], 
                             label=leg_names[leg_idx], 
                             color=colors[leg_idx], 
                             linewidth=1.5,
                             alpha=0.8)
            
        axs[ax_idx].set_ylabel(axis_labels[ax_idx], fontsize=12, fontweight='bold')
        axs[ax_idx].grid(True, linestyle='--', alpha=0.6)
        axs[ax_idx].legend(loc='upper right', fontsize=9)
        
    axs[2].set_xlabel('Time [s]', fontsize=12, fontweight='bold')
    axs[0].set_title('World Frame Foot Positions (r_feet_wf) Over Time', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.show()


def plot_foot_trajectory_3d(history_p_feet_wf, history_Sa=None, history_x=None, history_p_feet_des_wf=None):
    """
    시간에 따른 4개 다리의 실제 절대 좌표 궤적, 목표 착지점, 몸통(CoM)의 궤적을 3D로 시각화합니다.
    """
    p_feet = np.array(history_p_feet_wf) # Shape: (N, 3, 4)
    
    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 1. 몸통(CoM) 궤적 그리기
    if history_x is not None:
        x_state = np.array(history_x)
        com_x = x_state[:, 3]
        com_y = x_state[:, 4]
        com_z = x_state[:, 5]
        ax.plot(com_x, com_y, com_z, color='k', linewidth=3, linestyle='-.', label='CoM (Body) Trajectory')

    leg_names = ['FR', 'FL', 'RR', 'RL']
    colors = ['r', 'b', 'g', 'm']
    
    # 목표 착지점 데이터가 있다면 형태를 변환
    if history_p_feet_des_wf is not None:
        p_feet_des = np.array(history_p_feet_des_wf)
    
    # 2. 4개의 다리에 대해 루프
    for i in range(4):
        # 실제 발 궤적
        x = p_feet[:, 0, i]
        y = p_feet[:, 1, i]
        z = p_feet[:, 2, i]
        
        # 스윙 궤적 선
        ax.plot(x, y, z, color=colors[i], alpha=0.4, linestyle='-', label=f'{leg_names[i]} Actual')
        
        # 접촉 상태(Sa)에 따른 마커
        if history_Sa is not None:
            sa = np.array(history_Sa)[:, i]
            
            # 스윙(Air)
            swing_idx = (sa == 1)
            ax.scatter(x[swing_idx], y[swing_idx], z[swing_idx], 
                       color=colors[i], s=10, alpha=0.6, marker='^')
            
            # 스탠스(Ground)
            stance_idx = (sa == 0)
            ax.scatter(x[stance_idx], y[stance_idx], z[stance_idx], 
                       color='k', s=25, alpha=1.0, marker='o', edgecolors=colors[i])
        else:
            ax.scatter(x, y, z, color=colors[i], s=10, alpha=0.5)

        # 3. [추가] 목표 착지점(Des) 렌더링
        if history_p_feet_des_wf is not None:
            x_des = p_feet_des[:, 0, i]
            y_des = p_feet_des[:, 1, i]
            z_des = p_feet_des[:, 2, i]
            
            # 목표 착지점은 별표 마커(*)로 크게 표시
            ax.scatter(x_des, y_des, z_des, color=colors[i], s=60, alpha=0.7, marker='*', 
                       label=f'{leg_names[i]} Target (Des)')

    # 4. 그래프 설정
    ax.set_xlabel('X [m] (Forward)')
    ax.set_ylabel('Y [m] (Lateral)')
    ax.set_zlabel('Z [m] (Height)')
    ax.set_title('3D Foot Trajectories vs Target Placements')
    
    # 3D 비율 맞추기
    max_range = np.array([p_feet[:, 0, :].max() - p_feet[:, 0, :].min(), 
                          p_feet[:, 1, :].max() - p_feet[:, 1, :].min(), 
                          p_feet[:, 2, :].max() - p_feet[:, 2, :].min()]).max() / 2.0
    
    mid_x = (p_feet[:, 0, :].max() + p_feet[:, 0, :].min()) * 0.5
    mid_y = (p_feet[:, 1, :].max() + p_feet[:, 1, :].min()) * 0.5
    mid_z = (p_feet[:, 2, :].max() + p_feet[:, 2, :].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(0, mid_z + max_range)
    
    # 범례 중복 방지 처리
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper left', bbox_to_anchor=(1.05, 1))
    
    plt.tight_layout()
    plt.show()


def plot_swing_progress(history_s, dt=0.033):
    """
    시간에 따른 스윙 다리의 진행률 's' (0.0 ~ 1.0)를 시각화합니다.
    """
    s_array = np.array(history_s) # Shape: (N, 4)
    time = np.arange(s_array.shape[0]) * dt
    
    leg_names = ['FR (Front-Right)', 'FL (Front-Left)', 'RR (Rear-Right)', 'RL (Rear-Left)']
    colors = ['r', 'b', 'g', 'm']
    
    fig, axs = plt.subplots(4, 1, figsize=(11, 5), sharex=True)
    
    for i in range(4):
        ax = axs[i]
        # 해당 다리의 s 값 그리기
        ax.plot(time, s_array[:, i], color=colors[i], linewidth=2)
        
        # 그래프 꾸미기
        ax.set_ylim(-0.1, 1.1)
        ax.set_ylabel('Swing Progress (s)')
        ax.set_title(f'Leg {i}: {leg_names[i]}')
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Stance 구간(s=0)을 회색 음영으로 칠해 구분감 부여
        stance_idx = (s_array[:, i] == 0.0)
        ax.fill_between(time, -0.1, 1.1, where=stance_idx, color='gray', alpha=0.2)
        
    axs[-1].set_xlabel('Time [s]')
    fig.suptitle('Swing Phase Progress (s) over Time', fontsize=16)
    
    plt.tight_layout()
    plt.show()