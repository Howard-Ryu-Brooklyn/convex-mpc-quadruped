import numpy as np
import matplotlib.pyplot as plt

def plot_saved_history(filename='sim_history.npz', dt=0.033):
    # 1. 데이터 로드
    print(f"📂 '{filename}' 데이터 불러오는 중...")
    data = np.load(filename)
    
    # 변수에 할당 (형태 확인용 주석 추가)
    h_x = data['x']               # (N, 13) 
    h_xref = data['xref']         # (N, 13)
    h_F_G = data['F_G']           # (N, 3, 4)
    h_Sa = data['Sa']             # (N, 4)
    h_p_feet = data['p_feet_act_bf']  # (N, 3, 4)
    h_s = data['s']               # (N, 4)
    
    # 시간 축(Time vector) 생성
    N = h_x.shape[0]
    time = np.arange(N) * dt
    
    # 2. 그래프 그리기 세팅
    fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    # ---------------------------------------------------------
    # [Plot 1] 로봇 몸체 Z축(Height) 궤적 (추종 성능 확인)
    # ---------------------------------------------------------
    # h_x[:, 5] : X_current의 5번째 인덱스 = Z position
    axs[0].plot(time, h_xref[:, 5], 'r--', label='Reference Z', linewidth=2)
    axs[0].plot(time, h_x[:, 5], 'b-', label='Actual Z')
    axs[0].set_ylabel('CoM Height [m]')
    axs[0].set_title('CoM Z-Position Tracking')
    axs[0].legend()
    axs[0].grid(True)
    
    # ---------------------------------------------------------
    # [Plot 2] FR 다리 (Index 0)의 발바닥 Z축 높이 및 스윙 상태
    # ---------------------------------------------------------
    # h_p_feet[:, 2, 0] : Z축(2), FR다리(0)
    ax2 = axs[1].twinx() # Contact 상태(0 or 1)를 그리기 위한 듀얼 Y축
    ax2.fill_between(time, 0, h_Sa[:, 0], color='gray', alpha=0.2, label='Contact (Stance)')
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(['Swing', 'Stance'])
    
    axs[1].plot(time, h_p_feet[:, 2, 0], 'g-', label='FR Foot Z')
    axs[1].set_ylabel('Foot Z [m]')
    axs[1].set_title('FR Foot Trajectory & Contact State')
    axs[1].legend(loc='upper left')
    axs[1].grid(True)
    
    # ---------------------------------------------------------
    # [Plot 3] 4개 다리의 Z축 지면 반발력 (Fz)
    # ---------------------------------------------------------
    leg_names = ['FR', 'FL', 'RR', 'RL']
    colors = ['r', 'g', 'b', 'm']
    for i in range(4):
        # h_F_G[:, 2, i] : F_G의 Z축(2), i번째 다리
        axs[2].plot(time, h_F_G[:, 2, i], color=colors[i], label=f'{leg_names[i]} Fz')
        
    axs[2].set_xlabel('Time [s]')
    axs[2].set_ylabel('Force Z [N]')
    axs[2].set_title('Ground Reaction Forces (Fz)')
    axs[2].legend(loc='upper right')
    axs[2].grid(True)
    
    plt.tight_layout()
    plt.show()


def plot_motor_torques(filename='sim_history.npz', dt=0.033):
    """
    저장된 npz 파일에서 토크 데이터를 불러와 다리별로 플롯하는 함수
    """
    print(f"📂 '{filename}' 데이터 불러오는 중...")
    try:
        data = np.load(filename)
        h_tau = data['tau']  # Shape: (N, 12)
    except KeyError:
        print("❌ 저장된 파일에 'tau' 데이터가 없습니다. 시뮬레이터 로깅 코드를 확인하세요.")
        return

    # 시간 축 생성
    N = h_tau.shape[0]
    time = np.arange(N) * dt

    # 다리 및 관절 이름 정의
    leg_names = ['FR', 'FL', 'RR', 'RL']
    joint_names = ['AbAd', 'Hip', 'Knee']
    colors = ['r', 'g', 'b'] # 각 관절별 색상 지정

    # 4개의 서브플롯 생성 (각 다리당 1개의 그래프)
    fig, axs = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    fig.suptitle('Motor Torques over Time', fontsize=16)

    for i in range(4): # 4개 다리 반복
        ax = axs[i]
        
        # 각 다리마다 3개의 관절 토크 그리기
        for j in range(3):
            joint_idx = (i * 3) + j
            ax.plot(time, h_tau[:, joint_idx], color=colors[j], label=f'{joint_names[j]}')
            
        ax.set_ylabel('Torque [Nm]')
        ax.set_title(f'[{leg_names[i]}] Leg Torques')
        ax.legend(loc='upper right')
        ax.grid(True)
        
        # 토크 한계선(Limit) 표시 (선택사항, 필요시 파라미터에 맞게 수정)
        ax.axhline(250, color='gray', linestyle='--', alpha=0.5)
        ax.axhline(-250, color='gray', linestyle='--', alpha=0.5)

    axs[-1].set_xlabel('Time [s]')
    plt.tight_layout()
    plt.show()


def plot_swing_trajectory(filename='sim_history.npz', dt=1/9000.0):
    """
    저장된 데이터를 불러와 목표 스윙 궤적(베지에 곡선)과 실제 발의 궤적을 
    월드 프레임(World Frame) 기준으로 2D 및 3D로 비교 도시합니다.
    """
    print(f"📂 '{filename}' 데이터 불러오는 중...")
    try:
        data = np.load(filename)
        h_p_des = data['p_feet_des_wf'] # (N, 3, 4)
        h_p_act = data['p_feet_act_wf'] # (N, 3, 4)
        h_Sa = data['Sa']               # (N, 4)
    except KeyError as e:
        print(f"❌ 데이터 로드 실패: {e}")
        print("시뮬레이터에서 'p_feet_des_wf', 'p_feet_act_wf'가 제대로 저장되었는지 확인하세요.")
        return

    # 데이터 길이 및 시간 축 생성
    N = min(h_p_des.shape[0], h_p_act.shape[0], h_Sa.shape[0])
    h_p_des = h_p_des[:N]
    h_p_act = h_p_act[:N]
    h_Sa = h_Sa[:N]
    
    time = np.arange(N) * dt
    leg_names = ['FR', 'FL', 'RR', 'RL']
    
    # =========================================================================
    # 1. 2D 트래킹 플롯 (시간에 따른 X, Z축 변화)
    # =========================================================================
    fig2d, axs = plt.subplots(4, 3, figsize=(14, 12), sharex=True)
    fig2d.suptitle('Swing Trajectory Tracking over Time (World Frame)', fontsize=16, fontweight='bold')

    for i in range(4):
        swing_mask = (h_Sa[:, i] == 1)
        
        # X축 (Forward) 플롯
        axs[i, 0].plot(time, h_p_des[:, 0, i], 'r--', label='Ref X', linewidth=2)
        axs[i, 0].plot(time, h_p_act[:, 0, i], 'b-', label='Actual X', alpha=0.7, linewidth=1.5)
        axs[i, 0].fill_between(time, axs[i, 0].get_ylim()[0], axs[i, 0].get_ylim()[1], 
                               where=swing_mask, color='gray', alpha=0.2, label='Swing Phase')
        axs[i, 0].set_ylabel(f'[{leg_names[i]}] X [m]')
        if i == 0: axs[i, 0].legend(loc='upper right')
        axs[i, 0].grid(True, linestyle=':', alpha=0.6)

        # Y축 (Lateral) 플롯
        axs[i, 1].plot(time, h_p_des[:, 1, i], 'r--', label='Ref Y', linewidth=2)
        axs[i, 1].plot(time, h_p_act[:, 1, i], 'b-', label='Actual Y', alpha=0.7, linewidth=1.5)
        axs[i, 1].fill_between(time, axs[i, 0].get_ylim()[0], axs[i, 0].get_ylim()[1], 
                               where=swing_mask, color='gray', alpha=0.2, label='Swing Phase')
        axs[i, 1].set_ylabel(f'[{leg_names[i]}] Y [m]')
        if i == 0: axs[i, 0].legend(loc='upper right')
        axs[i, 1].grid(True, linestyle=':', alpha=0.6)

        # Z축 (Height) 플롯
        axs[i, 2].plot(time, h_p_des[:, 2, i], 'r--', label='Ref Z (Bezier)', linewidth=2)
        axs[i, 2].plot(time, h_p_act[:, 2, i], 'b-', label='Actual Z', alpha=0.7, linewidth=1.5)
        axs[i, 2].fill_between(time, axs[i, 1].get_ylim()[0], axs[i, 1].get_ylim()[1], 
                               where=swing_mask, color='gray', alpha=0.2)
        axs[i, 2].set_ylabel(f'[{leg_names[i]}] Z [m]')
        if i == 0: axs[i, 1].legend(loc='upper right')
        axs[i, 2].grid(True, linestyle=':', alpha=0.6)

    axs[3, 0].set_xlabel('Time [s]')
    axs[3, 1].set_xlabel('Time [s]')
    fig2d.tight_layout()
    fig2d.subplots_adjust(top=0.92)

    # =========================================================================
    # 2. 3D 공간 궤적 플롯 (X, Y, Z)
    # =========================================================================
    fig3d = plt.figure(figsize=(14, 12))
    fig3d.suptitle('3D Spatial Foot Trajectory (World Frame)', fontsize=16, fontweight='bold')

    for i in range(4):
        # 2x2 서브플롯으로 4개 다리의 3D 궤적을 각각 생성
        ax3d = fig3d.add_subplot(2, 2, i+1, projection='3d')
        
        # 베지에 목표 궤적 (빨간색 점선)
        ax3d.plot(h_p_des[:, 0, i], h_p_des[:, 1, i], h_p_des[:, 2, i], 
                  'r--', label='Reference Trajectory', linewidth=2, alpha=0.8)
        
        # 실제 발의 궤적 (파란색 실선)
        ax3d.plot(h_p_act[:, 0, i], h_p_act[:, 1, i], h_p_act[:, 2, i], 
                  'b-', label='Actual Trajectory', linewidth=1.5, alpha=0.7)
        
        # 스윙(Swing) 구간만 별도의 색(초록색 점)으로 강조하여 지면 접촉 구간과 구분
        swing_indices = np.where(h_Sa[:, i] == 1)[0]
        if len(swing_indices) > 0:
            ax3d.scatter(h_p_act[swing_indices, 0, i], h_p_act[swing_indices, 1, i], h_p_act[swing_indices, 2, i], 
                         color='limegreen', s=5, alpha=0.6, label='Swing Phase Points', zorder=3)

        # 축 라벨 및 타이틀 세팅
        ax3d.set_title(f'{leg_names[i]} Leg 3D Path', fontweight='bold')
        ax3d.set_xlabel('X (Forward) [m]')
        ax3d.set_ylabel('Y (Lateral) [m]')
        ax3d.set_zlabel('Z (Height) [m]')
        
        if i == 0:
            ax3d.legend(loc='upper left')

    fig3d.tight_layout()
    fig3d.subplots_adjust(top=0.92)

    # 창 두 개를 한꺼번에 띄움
    plt.show()


def plot_swing_and_contact(filename='sim_history.npz', dt=1/1000.0):
    """
    저장된 데이터를 불러와 4개 다리의 스윙 진행률(s)과 
    지면 접촉 상태(Contact Sequence, Sa)를 시간에 따라 시각화합니다.
    """
    print(f"📂 '{filename}' 데이터 불러오는 중...")
    try:
        data = np.load(filename)
        h_s = data['s']    # Shape: (N, 4) - 스윙 진행률 (0.0 ~ 1.0)
        h_Sa = data['Sa']  # Shape: (N, 4) - 접촉 상태 (1: Stance, 0: Swing)
    except KeyError as e:
        print(f"❌ 데이터 로드 실패: {e}")
        print("시뮬레이터에서 's'와 'Sa'가 제대로 저장되었는지 확인하세요.")
        return

    # 데이터 길이 및 시간 축 생성
    N = min(h_s.shape[0], h_Sa.shape[0])
    h_s = h_s[:N]
    h_Sa = h_Sa[:N]
    
    time = np.arange(N) * dt
    leg_names = ['FR', 'FL', 'RR', 'RL']
    
    # 4개의 서브플롯 생성 (각 다리당 1개의 그래프)
    fig, axs = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    fig.suptitle('Swing Phase (s) & Contact Sequence over Time', fontsize=16, fontweight='bold')

    for i in range(4):
        ax = axs[i]
        
        # 1. 스윙 진행률 (s) 플롯: 0에서 1로 상승하는 톱니파(Sawtooth) 형태
        ax.plot(time, h_s[:, i], color='b', linewidth=2, label='Swing Progress (s)')
        
        # 2. 지면 접촉 상태 (Contact Sequence) 시각화 (배경색 칠하기)
        # Sa == 1 (지면 접촉, Stance)인 구간을 옅은 초록색으로 칠함
        stance_mask = (h_Sa[:, i] == 1)
        ax.fill_between(time, -0.1, 1.1, where=stance_mask, 
                        color='mediumseagreen', alpha=0.3, label='Stance Phase (Sa=1)')
        
        # 스윙 구간 시각화 (선택 사항: Sa == 0 인 구간을 옅은 회색으로)
        swing_mask = (h_Sa[:, i] == 0)
        ax.fill_between(time, -0.1, 1.1, where=swing_mask, 
                        color='lightgray', alpha=0.3, label='Swing Phase (Sa=0)')

        # 그래프 꾸미기
        ax.set_ylim(-0.1, 1.1)
        ax.set_ylabel(f'[{leg_names[i]}]\nPhase')
        ax.grid(True, linestyle=':', alpha=0.6)
        
        # 첫 번째 그래프에만 범례 추가
        if i == 0:
            ax.legend(loc='upper right', bbox_to_anchor=(1.0, 1.15), ncol=3)

    axs[-1].set_xlabel('Time [s]', fontsize=12)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92) # Title이 겹치지 않도록 여백 조정
    plt.show()

if __name__ == "__main__":
    # 데이터가 저장된 루프 주기(step_vis)에 맞춰 dt를 넣어주세요. (예: 30Hz면 1/30 초)
    # 파라미터 파일의 sim_ts * step_vis 값과 동일해야 합니다.
    sim_freq = 9000
    sim_dt = 1/sim_freq

    plot_saved_history(filename='sim_history.npz', dt=sim_dt)

    # plot_motor_torques(filename='sim_history.npz', dt=sim_dt)
    
    # plot_swing_trajectory('sim_history.npz')

    # plot_swing_and_contact('sim_history.npz')