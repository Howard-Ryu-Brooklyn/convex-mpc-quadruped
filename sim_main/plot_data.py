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
    h_p_feet = data['p_feet_wf']  # (N, 3, 4)
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

import numpy as np
import matplotlib.pyplot as plt

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


if __name__ == "__main__":
    # 데이터가 저장된 루프 주기(step_vis)에 맞춰 dt를 넣어주세요. (예: 30Hz면 1/30 초)
    # 파라미터 파일의 sim_ts * step_vis 값과 동일해야 합니다.
    sim_freq = 9000
    sim_dt = 1/sim_freq
    
    plot_saved_history(filename='sim_history.npz', dt=sim_dt)

    plot_motor_torques(filename='sim_history.npz', dt=sim_dt)