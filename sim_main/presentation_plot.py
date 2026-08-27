import numpy as np
import matplotlib.pyplot as plt

def plot_saved_history(filename='sim_history.npz', dt=0.033, save_image=True):
    # 1. 데이터 로드
    print(f"📂 '{filename}' 데이터 불러오는 중...")
    try:
        data = np.load(filename)
    except FileNotFoundError:
        print(f"❌ '{filename}' 파일을 찾을 수 없습니다. 시뮬레이션을 먼저 실행해주세요.")
        return
    
    # 변수에 할당 (보내주신 원본 코드 규격 반영)
    h_x = data['x']                         # (N, 13) 
    h_xref = data['xref']                   # (N, 13)
    h_F_G = data['F_G']                     # (N, 3, 4)
    h_Sa = data['Sa']                       # (N, 4)
    h_p_feet = data['p_feet_act_bf'] if 'p_feet_act_bf' in data else data.get('p_feet_des_bf', np.zeros((h_x.shape[0], 3, 4)))
    
    # 시간 축(Time vector) 생성
    N = h_x.shape[0]
    time = np.arange(N) * dt
    
    # 2. 3행 2열 그래프 그리기 세팅 (GridSpec 활용으로 깨짐 완벽 방지)
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle('4-Legged Robot Convex MPC Simulation Results', fontsize=14, fontweight='bold')
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.25)
    
    # ---------------------------------------------------------
    # [1행 1열] CoM Z축(Height) 궤적 (위치 목표 추종)
    # ---------------------------------------------------------
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(time, h_xref[:, 5], 'r--', label='Reference Z', linewidth=2)
    ax1.plot(time, h_x[:, 5], 'b-', label='Actual Z')
    ax1.set_ylabel('CoM Height [m]')
    ax1.set_title('1. CoM Z-Position Tracking (Position Goal)')
    ax1.legend(loc='lower right')
    ax1.grid(True)
    
    # ---------------------------------------------------------
    # [1행 2열] CoM X축 전진 속도(Vx) 지령 및 추종 (속도 목표 및 속도)
    # ---------------------------------------------------------
    # 13차원 상태 벡터 기준 Vx 인덱스 (보통 9, 없으면 6)
    vx_idx = 9 if h_x.shape[1] > 9 else 6 
    ax2 = fig.add_subplot(gs[0, 1], sharex=ax1)
    ax2.plot(time, h_xref[:, vx_idx], 'r--', label='Reference Vx', linewidth=2)
    ax2.plot(time, h_x[:, vx_idx], 'g-', label='Actual Vx')
    ax2.set_ylabel('Velocity Vx [m/s]')
    ax2.set_title('2. CoM Forward Velocity (Velocity Goal & Actual)')
    ax2.legend(loc='lower right')
    ax2.grid(True)

    # ---------------------------------------------------------
    # [2행 1열] FR 다리 (Index 0)의 발바닥 Z축 높이 및 스윙 상태
    # ---------------------------------------------------------
    ax3 = fig.add_subplot(gs[1, 0], sharex=ax1)
    ax3_twin = ax3.twinx() # Contact 상태(0 or 1)를 위한 듀얼 Y축
    ax3_twin.fill_between(time, 0, h_Sa[:, 0], color='gray', alpha=0.2, label='Contact (Stance)')
    ax3_twin.set_yticks([0, 1])
    ax3_twin.set_yticklabels(['Swing', 'Stance'])
    ax3_twin.grid(False) # 듀얼 축 그리드 충돌 방지
    
    ax3.plot(time, h_p_feet[:, 2, 0], 'g-', label='FR Foot Z')
    ax3.set_ylabel('Foot Z [m]')
    ax3.set_title('3. FR Foot Trajectory & Contact State')
    ax3.legend(loc='upper left')
    ax3.grid(True)

    # ---------------------------------------------------------
    # [2행 2열] 4개 다리의 Z축 지면 반발력 (Fz)
    # ---------------------------------------------------------
    ax4 = fig.add_subplot(gs[1, 1], sharex=ax1)
    leg_names = ['FR', 'FL', 'RR', 'RL']
    colors = ['r', 'g', 'b', 'm']
    for i in range(4):
        ax4.plot(time, h_F_G[:, 2, i], color=colors[i], label=f'{leg_names[i]} Fz')
        
    ax4.set_ylabel('Force Z [N]')
    ax4.set_title('4. Ground Reaction Forces (Fz)')
    ax4.legend(loc='upper right', fontsize=8)
    ax4.grid(True)

    # ---------------------------------------------------------
    # [3행 전체 병합] 전체 다리 스윙 높이 또는 추가 정보 (아래쪽에 넓게 배치)
    # ---------------------------------------------------------
    ax5 = fig.add_subplot(gs[2, :], sharex=ax1) # 3행의 0열과 1열을 통째로 병합하여 사용
    if h_p_feet.ndim == 3 and h_p_feet.shape[2] >= 4:
        for i in range(4):
            ax5.plot(time, h_p_feet[:, 2, i], color=colors[i], label=f'{leg_names[i]} Foot Z')
    ax5.set_xlabel('Time [s]')
    ax5.set_ylabel('Foot Z [m]')
    ax5.set_title('5. All Legs Swing Trajectory (Foot Heights)')
    ax5.legend(loc='upper right', ncol=4, fontsize=8)
    ax5.grid(True)

    # 레이아웃 간격 자동 정돈 (깨짐 방지)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    # 3. 플롯 이미지 저장 기능
    if save_image:
        filename_png = 'simulation_results_plot.png'
        fig.savefig(filename_png, dpi=300)
        print(f"💾 플롯 고해상도 이미지 저장 완료: '{filename_png}'")

    plt.show()

if __name__ == "__main__":
    plot_saved_history()