import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
import mpl_toolkits.mplot3d.art3d as art3d

def animate_quadruped(history_p, history_R, history_r_feet, history_F_G, body_length, body_width, dt=0.033, save=False):
    """
    4족보행 로봇의 시뮬레이션 결과를 3D 애니메이션으로 생성합니다.
    """
    print("🎥 3D 시각화 애니메이션 렌더링 시작...")
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 3D 그래프 축 범위 설정
    ax.set_xlim(-1, 2)
    ax.set_ylim(-1.5, 1.5)
    ax.set_zlim(0, 1.0)
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')
    
    # 지면 그리기
    square = np.array([[-10, -10, 0], [10, -10, 0], [10, 10, 0], [-10, 10, 0]])
    ground = art3d.Poly3DCollection([square], alpha=0.1, facecolor='gray')
    ax.add_collection3d(ground)
    
    # 로봇의 몸체(사각형)와 4개의 다리를 그릴 선(Line) 객체 초기화
    body_plot, = ax.plot([], [], [], 'k-', linewidth=3)
    leg_plots = [ax.plot([], [], [], 'o-', linewidth=2, markersize=4)[0] for _ in range(4)]
        
    # 지면 반발력 벡터(Quiver)를 None으로 초기화 (가장 안전한 버그 방지 방식)
    grf_quivers = [None for _ in range(4)]

    # 다리 색상 구분: FR(빨강), FL(파랑), RR(초록), RL(보라)
    colors = ['r', 'b', 'g', 'm']
    for i, line in enumerate(leg_plots):
        line.set_color(colors[i])

    # =================================================================
    # [핵심 수정 1] 어깨 좌표를 발(Feet) 좌표와 동일하게 (FR, FL, RR, RL) 순서로 원상복구
    # =================================================================
    L, W = body_length / 2, body_width / 2
    body_local = np.array([
        [ L,  L, -L, -L], # X 좌표 (FR, FL, RR, RL)
        [-W,  W, -W,  W], # Y 좌표 (FR, FL, RR, RL) - 순서 완벽 매칭!
        [ 0,  0,  0,  0]  # Z 좌표
    ])

    # 몸통을 그릴 때 꼬이지 않는 사각형 외곽선을 만들기 위한 인덱스 순서
    draw_order = [0, 1, 3, 2, 0]

    def update(frame):
        # 1. 현재 프레임 데이터 로드
        p = history_p[frame] 
        R = history_R[frame] 
        feet_offset = history_r_feet[frame] 
        F_G = history_F_G[frame] 

        # 2. 월드 절대 좌표 계산
        body_world = R @ body_local + p.reshape(3, 1)
        feet_world = feet_offset + p.reshape(3, 1) 

        # =================================================================
        # [핵심 수정 2] 몸체 그리기 (draw_order를 사용해 순서대로 점을 이음)
        # =================================================================
        body_draw = body_world[:, draw_order]
        body_plot.set_data(body_draw[0, :], body_draw[1, :])
        body_plot.set_3d_properties(body_draw[2, :])
        
        # 4. 다리 및 GRF 벡터 그리기 (하나의 for 루프 안에서 안전하게 처리)
        for i in range(4):
            # [다리 렌더링] 인덱스 i가 어깨와 발에 1:1로 정확히 매칭됨 (다리 꼬임 해결!)
            leg_x = [body_world[0, i], feet_world[0, i]]
            leg_y = [body_world[1, i], feet_world[1, i]]
            leg_z = [body_world[2, i], feet_world[2, i]]
            
            leg_plots[i].set_data(leg_x, leg_y)
            leg_plots[i].set_3d_properties(leg_z)
            
            # [GRF 화살표 업데이트] 이전 프레임 지우기
            if grf_quivers[i] is not None:
                try:
                    grf_quivers[i].remove()
                except Exception:
                    pass
                grf_quivers[i] = None
                
            foot_pos = feet_world[:, i] 
            force_vec = F_G[:, i] 

            # [Matplotlib 버그 회피] 힘이 0.001 N 이상일 때만 화살표를 그림
            if np.linalg.norm(force_vec) > 1e-3:
                scale = 0.0005
                fv_scaled = force_vec * scale
                grf_quivers[i] = ax.quiver(foot_pos[0], foot_pos[1], foot_pos[2],
                                          fv_scaled[0], fv_scaled[1], fv_scaled[2],
                                          pivot='tail', color='gray', linewidth=2, arrow_length_ratio=0.3)
                
        # 유효한 화살표 객체만 렌더러에 반환
        valid_quivers = [q for q in grf_quivers if q is not None]
        return [body_plot] + leg_plots + valid_quivers

    # 애니메이션 생성
    ani = animation.FuncAnimation(
        fig, update, frames=len(history_p), interval=dt*1000, blit=False
    )
    
    plt.close(fig)
    print("✅ 렌더링 완료!")

    if (save):
        num_frames = len(history_p)
        ani = animation.FuncAnimation(fig, update, frames=num_frames, interval=dt*1000, blit=False)
        
        # ----------------------------------------------------
        # 💾 애니메이션 파일로 저장하기 (이 부분을 추가하세요!)
        # ----------------------------------------------------
        print("🎬 동영상 파일로 저장 중입니다... (몇 분 정도 소요될 수 있습니다)")
        
        # 옵션 1: MP4 동영상으로 저장 (고화질, 추천)
        ani.save('quadruped_mpc.mp4', writer='ffmpeg', fps=int(1/dt), dpi=200)
        
        # 옵션 2: GIF 움짤로 저장 (웹 공유용, 용량이 큼)
        # ani.save('quadruped_mpc.gif', writer='pillow', fps=int(1/dt), dpi=100)
        
        print("✅ 저장 완료! 'quadruped_mpc.mp4' 파일을 확인하세요.")
        
        plt.close(fig) # 불필요한 빈 피규어 출력 방지
    return HTML(ani.to_jshtml()) # 주피터 노트북 출력용

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
    fig, axs = plt.subplots(4, 3, figsize=(16, 12))
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
    fig, axs = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    
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