import numpy as np
import matplotlib
matplotlib.use('Agg')  # macOS 충돌 방지
import matplotlib.pyplot as plt

def main():
    # 저장된 데이터 불러오기
    try:
        data = np.load("sim_data.npz")
        time_array = data["time"]
        torque_array = data["torque"]
        qpos_array = data["qpos"]
    except FileNotFoundError:
        print("저장된 데이터 파일('sim_data.npz')이 없습니다. 먼저 시뮬레이션을 실행해주세요.")
        return

    leg_names = ["FR (Front Right)", "FL (Front Left)", "RR (Rear Right)", "RL (Rear Left)"]
    joint_labels = ["Abduction", "Hip", "Knee"]

    fig, axes = plt.subplots(4, 2, figsize=(14, 16), sharex=True)

    for leg_idx, leg_name in enumerate(leg_names):
        start_idx = leg_idx * 3
        
        # 토크 플롯 (좌측)
        ax_torque = axes[leg_idx, 0]
        for j in range(3):
            ax_torque.plot(time_array, torque_array[:, start_idx + j], label=joint_labels[j])
        ax_torque.set_title(f"{leg_name} - Torques")
        ax_torque.set_ylabel("Torque [Nm]")
        ax_torque.grid(True)
        ax_torque.legend(loc="upper right", fontsize=8)

        # 각도 플롯 (우측)
        ax_pos = axes[leg_idx, 1]
        for j in range(3):
            ax_pos.plot(time_array, qpos_array[:, start_idx + j], label=joint_labels[j])
        ax_pos.set_title(f"{leg_name} - Positions")
        ax_pos.set_ylabel("Angle [rad]")
        ax_pos.grid(True)
        ax_pos.legend(loc="upper right", fontsize=8)

    axes[3, 0].set_xlabel("Time [s]")
    axes[3, 1].set_xlabel("Time [s]")

    plt.tight_layout()
    plt.savefig("simulation_result_by_leg.png", dpi=300)
    print("다리별 그래프가 'simulation_result_by_leg.png'로 성공적으로 저장되었습니다!")

if __name__ == "__main__":
    main()