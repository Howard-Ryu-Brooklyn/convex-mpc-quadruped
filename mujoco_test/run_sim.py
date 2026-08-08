import mujoco
import mujoco.viewer
import time
import numpy as np

def get_Rotation_Matrix_Body2World(model, data):
    trunk_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk")
    trunk_quat = data.xquat[trunk_body_id]
    R_world_to_body = np.zeros(9)
    mujoco.mju_quat2Mat(R_world_to_body, trunk_quat)
    return R_world_to_body.reshape(3, 3).T

def get_All_Leg_Jacop(model, data):
    foot_names = ["FR_calf", "FL_calf", "RR_calf", "RL_calf"]
    joint_start_idx = 6
    leg_jacobians = {}
    for idx, foot_name in enumerate(foot_names):
        foot_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, foot_name)
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jacBody(model, data, jacp, jacr, foot_body_id)
        c_start = joint_start_idx + (idx * 3)
        leg_jacobians[foot_name] = jacp[:, c_start : c_start + 3]
    return leg_jacobians

def main():
    xml_path = "mit_cheetah3/scene.xml"
    # xml_path = "unitree_a1_motor/scene.xml"
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    total_mass = 70.0
    g_force = total_mass * 9.81
    g_vec = np.array([0.0, 0.0, g_force / 4.0])
    F_G = np.tile(g_vec.reshape(3, 1), (1, 4))
    Tm = np.zeros((12,))

    time_buffer = []
    torque_buffer = []
    qpos_buffer = []
    foot_names = ["FR_calf", "FL_calf", "RR_calf", "RL_calf"]

    home_qpos = np.array([
        0.0,  0.9, -1.8,  # FR
        0.0,  0.9, -1.8,  # FL
        0.0,  0.9, -1.8,  # RR
        0.0,  0.9, -1.8   # RL
    ])

    print("시뮬레이션 시작! (창을 닫으면 데이터가 저장됩니다)")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running() and data.time < 5.0:  # 테스트를 위해 5초로 제한
            step_start = time.time()

            current_time = data.time
            
            # =================================================================
            # [페이즈 1] 초기화 구간 (예: 0초 ~ 2초)
            # 목적: 로봇이 주저앉지 않고 홈포지션 자세를 단단히 잡도록 함
            # =================================================================
            if current_time < 1.0:
                # 방법 A: XML의 keyframe(home) 포지션을 목표 각도로 삼고 강력한 PD 토크를 직접 계산해서 줌
                # 또는 MuJoCo의 position 액추에이터를 쓰고 있다면 target 각도를 고정해 둠
                # 여기서는 예시로 홈포지션 관절 각도 정의 (앞서 본 12차원 홈 각도)
                home_qpos = np.array([0, 0.9, -1.8,  0, 0.9, -1.8,  0, 0.9, -1.8,  0, 0.9, -1.8])
                current_qpos = data.qpos[7:]
                current_qvel = data.qvel[6:] # 베이스 속도 제외 12개 관절 속도
                
                # 강한 가상의 PD 제어기로 로봇을 꼲꼲하게 세움
                Kp_init = 300.0
                Kd_init = 10.0
                init_torque = Kp_init * (home_qpos - current_qpos) - Kd_init * current_qvel
                
                data.ctrl[:] = init_torque

            # =================================================================
            # [페이즈 2] 본 제어 구간 (2초 이후 ~)
            # 목적: 우리가 만든 자코비안 기반 지면 반력 토크(Tm)나 MPC 토크 투입
            # =================================================================
            else:
                Rw_b = get_Rotation_Matrix_Body2World(model, data)
                leg_jacobians = get_All_Leg_Jacop(model, data)
                 
                Tm = np.zeros((12,))
                for idx, foot_name in enumerate(foot_names):
                    J_i = leg_jacobians[foot_name]
                    F_i = F_G[:, idx]
                    tau_i = J_i.T @ Rw_b.T @ F_i
                    Tm[idx * 3 : (idx + 1) * 3] = tau_i

                # 2. [매우 중요] 순수 GRF 토크만 주면 주저앉으므로, 
                # 홈 포지션 유지용 미니 PD 토크를 살짝 섞어서 자세 붕괴를 막아줍니다.
                current_qpos = data.qpos[7:]
                current_qvel = data.qvel[6:]
                Kp_stabilize = 100.0  # 약한 강성
                Kd_stabilize = 10.0
                tau_stabilize = Kp_stabilize * (home_qpos - current_qpos) - Kd_stabilize * current_qvel

                # 최종 제어 입력 = GRF 변환 토크 + 자세 안정화 토크
                data.ctrl[:] = Tm + tau_stabilize
                
            # actual_torques = data.actuator_force
            time_buffer.append(data.time)
            torque_buffer.append(data.ctrl.copy())
            qpos_buffer.append(data.qpos[7:].copy())
            
            mujoco.mj_step(model, data)
            viewer.sync()
            
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

    # 데이터를 파일로 안전하게 저장
    np.savez("sim_data.npz", time=time_buffer, torque=torque_buffer, qpos=qpos_buffer)
    print("시뮬레이션 종료 및 데이터 저장 완료 ('sim_data.npz')")

if __name__ == "__main__":
    main()