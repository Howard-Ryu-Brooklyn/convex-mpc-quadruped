import numpy as np
import config as cfg
from config import get_13d_state
from kinematics import get_q, get_r_feet_bf
from swing_trajectory_generator import compute_bezier_with_kinematics
from visualization import animate_quadruped, plot_state_tracking, plot_force_and_contact, plot_leg_angles, plot_foot_comparison, plot_foot_trajectory_3d, plot_r_feet_wf_over_time, plot_foot_trajectory_3d, plot_swing_progress
import convex_mpc as cvx_mpc
from gait_planning import get_gait_parameters, get_contact_state
import mujoco
import mujoco.viewer
import scipy.spatial.transform as transform
import mediapy as media # 영상 저장을 위한 라이브러리 (pip install mediapy)

# =====================================================================
# 헬퍼 함수 1: 회전 행렬 (Body -> World)
# =====================================================================
def get_Rotation_Matrix_Body2World(model, data, trunk_name="trunk"):
    """Trunk Body의 Body to World 회전 행렬 R_b2w (3x3) 반환"""
    trunk_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, trunk_name)
    trunk_quat = data.xquat[trunk_body_id]  # [w, x, y, z]
    R_b2w_flat = np.zeros(9)
    mujoco.mju_quat2Mat(R_b2w_flat, trunk_quat)
    return R_b2w_flat.reshape(3, 3)

# =====================================================================
# 헬퍼 함수 2: 동역학 행렬 (Mass Matrix M & Bias Vector)
# =====================================================================
def get_dynamics_matrices(model, data):
    """전체 시스템의 관성 행렬 M(18x18) 및 바이어스(C*q_dot + G) 추출"""
    n_v = model.nv # 18
    M = np.zeros((n_v, n_v), dtype=np.float64)
    mujoco.mj_fullM(model, data, M)
    bias = data.qfrc_bias.copy()
    return M, bias

# =====================================================================
# 헬퍼 함수 3: 4개 발의 위치 및 선속도 일괄 추출 (3, 4)
# =====================================================================
def get_all_feet_kinematics(model, data, foot_names, Rb_w):
    """
    4개 발의 World Frame 및 Body Frame 기준 위치/속도를 일괄 추출
    Returns:
        p_feet_wf (3, 4), v_feet_wf (3, 4)
        p_feet_bf (3, 4), v_feet_bf (3, 4)
    """
    p_com_wf = data.qpos[0:3].copy()
    v_com_wf = data.qvel[0:3].copy()
    
    p_feet_wf = np.zeros((3, 4))
    v_feet_wf = np.zeros((3, 4))
    p_feet_bf = np.zeros((3, 4))
    v_feet_bf = np.zeros((3, 4))
    
    site_vel_6d = np.zeros(6)
    
    for i, name in enumerate(foot_names):
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
        
        # 위치 추출
        p_feet_wf[:, i] = data.site_xpos[site_id].copy()
        p_feet_bf[:, i] = Rb_w @ (p_feet_wf[:, i] - p_com_wf)
        
        # 속도 추출 (MuJoCo 공식 API)
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_SITE, site_id, site_vel_6d, 0)
        v_feet_wf[:, i] = site_vel_6d[3:6].copy()
        v_feet_bf[:, i] = Rb_w @ (v_feet_wf[:, i] - v_com_wf)
        
    return p_feet_wf, v_feet_wf, p_feet_bf, v_feet_bf

# =====================================================================
# 헬퍼 함수 4: 4개 발의 자코비안 일괄 추출 (3, 18, 4) 및 수치 미분 J_dot
# =====================================================================
def get_all_feet_jacobians_and_derivatives(model, data, foot_names, prev_J_full, dt):
    """
    4개 발에 대한 (3, 18) 풀 자코비안 및 미분값(J_dot)을 일괄 계산
    Returns:
        J_full_all (3, 18, 4)
        J_dot_all (3, 18, 4)
    """
    n_v = model.nv
    J_full_all = np.zeros((3, n_v, 4))
    J_dot_all = np.zeros((3, n_v, 4))
    
    jacp = np.zeros((3, n_v))
    jacr = np.zeros((3, n_v))
    
    for i, name in enumerate(foot_names):
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
        mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
        
        J_current = jacp.copy()
        J_full_all[:, :, i] = J_current
        
        # 방어 코드 적용된 J_dot 계산
        if prev_J_full is None or np.all(prev_J_full[:, :, i] == 0):
            J_dot_all[:, :, i] = np.zeros_like(J_current)
        else:
            J_dot_all[:, :, i] = (J_current - prev_J_full[:, :, i]) / dt
            
    return J_full_all, J_dot_all

# =====================================================================
# 헬퍼 함수 5: 작업 공간 관성 행렬 (Lambda) 및 스윙 피드포워드 토크
# =====================================================================
def compute_operational_space_inertia(M, J_full):
    """M(18x18)과 J_full(3x18)로부터 Lambda(3x3) 계산"""
    M_inv = np.linalg.inv(M + np.eye(M.shape[0]) * 1e-6)
    Lambda_inv = J_full @ M_inv @ J_full.T
    return np.linalg.inv(Lambda_inv + np.eye(3) * 1e-6)

def compute_swing_leg_feedforward_torque_clean(M, bias, J_full, J_dot, a_ref):
    """일관된 인자로 전달받아 18차원 피드포워드 토크 계산"""
    Lambda = compute_operational_space_inertia(M, J_full)
    q_dot = data.qvel.copy() if 'data' in globals() else np.zeros(M.shape[0])
    
    accel_term = a_ref - (J_dot @ q_dot)
    tau_task = J_full.T @ Lambda @ accel_term
    tau_ff_full = tau_task + bias
    return tau_ff_full


def get_13d_state_from_mujoco(model, data):
    """MuJoCo data에서 Convex MPC용 13차원 상태 벡터 x_k 추출"""
    # 1. CoM 위치 & 속도
    pos = data.qpos[0:3].reshape(3, 1)
    vel = data.qvel[0:3].reshape(3, 1)
    
    # 2. 자세 (Quaternion -> Euler Roll, Pitch, Yaw)
    quat_wxyz = data.qpos[3:7]
    quat_xyzw = [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]
    r = transform.Rotation.from_quat(quat_xyzw)
    yaw, pitch, roll = r.as_euler('ZYX', degrees=False).reshape(3, 1)
    
   # 3. 각속도 (Body Frame -> World Frame 변환)
    ang_vel_body = data.qvel[3:6].reshape(3, 1)
    
    # 현재 상태의 회전행렬 구하기
    trunk_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk")
    trunk_quat = data.xquat[trunk_body_id]
    Rw_b = np.zeros(9)
    mujoco.mju_quat2Mat(Rw_b, trunk_quat)
    Rw_b = Rw_b.reshape(3, 3)
    
    ang_vel_world = Rw_b @ ang_vel_body # World Frame으로 회전
    
    euler_angles = np.array([roll, pitch, yaw], dtype=float).reshape(3, 1)
        
    x_13d = np.vstack([
        euler_angles,               # 3x1 (Roll, Pitch, Yaw)
        pos.reshape(3, 1),          # 3x1 (X, Y, Z)
        ang_vel_world.reshape(3, 1),# 3x1 (Wx, Wy, Wz)
        vel.reshape(3, 1),          # 3x1 (Vx, Vy, Vz)
        np.array([[1.0]])           # 1x1 (Gravity)
    ])
    return x_13d


def align_robot_to_ground(model, data, foot_site_names, foot_radius=0.025):
    """
    설정된 초기 관절 각도(qpos) 상태에서 발바닥 구(Sphere) 최하단이 
    지면(z=0)에 딱 닿도록 trunk의 z 위치를 자동 보정하는 함수
    """
    # 1. 현재 qpos 상태에서 기구학(전방 운동학) 연산 수행
    mujoco.mj_forward(model, data)
    
    # 2. 4개 발바닥 사이트(Site)의 월드 Z 좌표 추출
    foot_z_list = []
    for site_name in foot_site_names:
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if site_id != -1:
            # 사이트 중심 Z 위치 - 발 반지름(Radius) = 발바닥 최하단 높이
            foot_bottom_z = data.site_xpos[site_id][2] - foot_radius
            foot_z_list.append(foot_bottom_z)
            
    # 3. 가장 낮은 발바닥의 Z 위치 확인
    min_foot_z = np.min(foot_z_list)
    
    # 4. Trunk의 Z 위치(qpos[2]) 보정 (발바닥 최하단이 z=0에 오도록 차이만큼 차감)
    data.qpos[2] -= min_foot_z
    
    # 5. 보정된 위치로 기구학 재갱신
    mujoco.mj_forward(model, data)
    print(f"✅ 지면 밀착 완료: Trunk 높이 {data.qpos[2]:.4f} m (보정 오프셋: {-min_foot_z:.4f} m)")



# utils
DEG2RAD = np.pi / 180

# ==========================================
# 1. 시뮬레이션 파라미터 및 제어 주기 설정
# ==========================================
loop_blue_hz = 30    # [Hz] MPC 연산 주기 25~50
loop_red_hz = 1000   # [Hz] 상태 추정/스윙 제어 주기
loop_green_hz = 4500 # [Hz] 저수준 제어 주기
loop_sim_hz = 9000   # [Hz] 물리 엔진(동역학) 주기

sim_freq = loop_sim_hz
sim_ts = 1 / sim_freq
sim_tf = 4 # [sec] 전체 시뮬레이션 시간
num_steps = round(sim_tf / sim_ts)

step_blue = round(sim_freq / loop_blue_hz)   
step_red = round(sim_freq / loop_red_hz)     
step_green = round(sim_freq / loop_green_hz)    

dt_red = 1/loop_red_hz
dt_green = 1/loop_green_hz

# ==========================================
# 2. 초기 상태 및 변수 선언
# ==========================================
# 다리가 구부러진 상태를 시뮬레이션 하기 위해 높이를 다리가 쫙펴진 높이의 절반으로 설정

# 보행 스케줄러
gait_mode = ["standing", "trotting", "flying_trot", "bounding", "galloping"]
gait_period, gait_duty, gait_phase_offset = get_gait_parameters(gait_mode[4])

# Gait 파라미터 (예: Trot의 경우 보통 0.15초 ~ 0.25초)
T_swing = gait_period * (1-gait_duty)   # 발이 공중에 떠서 이동하는 목표 시간 (150ms)
T_stance = gait_period * gait_duty  # 발이 땅을 딛고 있는 목표 시간 (150ms)
clearance_height = 0.1 # 발을 들어올리는 최대 포물선 높이 (10cm)

# 스윙 상태 추적을 위한 타이머 및 저장 공간
Sa_current = np.zeros(4)
swing_time_counter = np.zeros(4)      # 각 다리가 공중에 떠 있는 시간을 기록 (0.0으로 초기화)
swing_start_pos_wf = np.zeros((3,4)) # 발이 땅에서 떨어진 순간의 위치를 영구 저장
p_feet_swing_traj_wf = np.zeros((3,4))
v_feet_swing_traj_wf = np.zeros((3,4))
a_feet_swing_traj_wf = np.zeros((3,4))

p_feet_swing_traj_bf = np.zeros((3,4))
v_feet_swing_traj_bf = np.zeros((3,4))
a_feet_swing_traj_bf = np.zeros((3,4))
current_s = np.zeros(4)

# 스윙 다리 Task-Space PD 제어 가인 (튜닝 가능)
# 🌟 1. 논문의 동적 게인 설정을 위한 파라미터 세팅 (루프 밖에서 1번만 정의해도 무방)
# 오차를 얼마나 빨리 줄일 것인가 (고유 진동수, rad/s)
omega_n = np.array([60.0, 60.0, 60.0]) # X, Y, Z축에 대한 진동수 (사용자가 튜닝)
# 감쇠비 (Damping Ratio, 보통 1.0인 임계 감쇠(Critical Damping) 사용)
zeta = np.array([1.0, 1.0, 1.0])

# Ttrot = 0.33s  10 timestep =  (mpc주기에 맞춰 타임스텝을 설정)
# Thound = 0.5s  16 timestep
# MPC는 최소 보행 한주기 정도는 내다 볼 수 있게 설계해야함
# horizon * mpc_dt = Tgait
# MPC 계산할때는 위 주기에서 10~16스텝 쪼갠걸로 행렬 계산
# MPC
mpc_dt = 1/loop_blue_hz
horizon = 10 # math.ceil(gait_period/mpc_dt) # 보행 한주기와 동일하게 설정

# Convex MPC 솔버 초기화
I_body_diag = [cfg.Ixx, cfg.Iyy, cfg.Izz]
mpc_solver = cvx_mpc.ConvexMPC(
    m=cfg.m,
    I_body=I_body_diag,
    gz=cfg.gz,
    dt=mpc_dt,
    horizon=horizon,
    Lweights = cvx_mpc.ConvexMPC.build_state_weight(cfg.L_w_th, cfg.L_w_z, cfg.L_w_yr, cfg.L_w_v),
    Kweights=cfg.K_w_f,
    mu=cfg.MU_FRICTION,
    fmin=cfg.fmin,
    fmax=cfg.fmax)

contact_sequence_horizon = np.zeros((4, horizon)) 

# Mujoco 
# 💡 커스텀 또는 Menagerie XML 경로 지정
xml_path = "/Users/icryu/Documents/code/quadruped-simulation/mujoco_test/mit_cheetah3/scene.xml"  

try:
    model = mujoco.MjModel.from_xml_path(xml_path)
except Exception as e:
    print(f"❌ MuJoCo 모델 로드 실패: {xml_path}")
    raise e

data = mujoco.MjData(model)

model.opt.timestep =sim_ts
foot_names = ['FR_site', 'FL_site', 'RR_site', 'RL_site']

# 1. 원하는 홈 포지션(기본 관절 각도) 입력
home_qpos = np.array([
    0.0, 0.0, 0.65,          # Trunk Pos (x, y, z) - 임시 높이
    1.0, 0.0, 0.0, 0.0,      # Trunk Quat (w, x, y, z)
    0.0, 0.9, -1.8,          # FR 다리 관절
    0.0, 0.9, -1.8,          # FL 다리 관절
    0.0, 0.9, -1.8,          # RR 다리 관절
    0.0, 0.9, -1.8           # RL 다리 관절
])
data.qpos[:] = home_qpos

# for compute J_dot in swing leg torque control
prev_J = np.zeros((4, 3, model.nv))

# 2. 지면 자동 정렬 함수 호출
site_names = ['FR_site', 'FL_site', 'RR_site', 'RL_site']
align_robot_to_ground(model, data, site_names, foot_radius=0.025)

# 시뮬레이션 변수
Rw_b = get_Rotation_Matrix_Body2World(model, data)
Rb_w = Rw_b.T


# 시각화 데이터 로깅 리스트
history_p, history_R, history_r_feet_wf, history_F_G = [], [], [], []
history_x, history_xref = [], []
history_Sa, history_p_feet_des_wf, history_p_feet_wf, history_r_feet_des_wf, history_p_feet_local = [], [], [], [], []
history_q = []
history_s = []
history_tau = []
history_p_feet_des_bf = []
history_p_feet_des_wf = []
history_p_feet_act_bf = []
history_p_feet_act_wf = []

# ----------------------------------------------------
# 카메라 및 렌더러 설정 (시뮬레이션 루프 진입 전)
# ----------------------------------------------------
framerate = 60
frames = []
renderer = mujoco.Renderer(model, height=480, width=640)

# 🌟 카메라 객체 생성 및 로봇 추적(Tracking) 설정
cam = mujoco.MjvCamera()
mujoco.mjv_defaultFreeCamera(model, cam)

trunk_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk")
cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING  # 카메라 모드를 추적 모드로 변경
cam.trackbodyid = trunk_id                      # 추적할 로봇 몸체 ID
cam.distance = 2.5                              # 로봇과의 거리 (m)
cam.elevation = -20.0                           # 위에서 아래로 내려다보는 각도 (deg)
cam.azimuth = 90.0                              # 바라보는 방향 각도 (90도: 측면 뷰)

next_render_time = 0.0
render_interval = 1.0 / framerate


# -----------------------------------------------------------------
# MuJoCo 수동 시각화 뷰어 및 루프 실행
# -----------------------------------------------------------------
print("🚀 MuJoCo 기반 4족보행 로봇 제어 시뮬레이션 시작!")

with mujoco.viewer.launch_passive(model, data) as viewer:
    # (선택 사항) 화면에 떠 있는 패시브 뷰어 창도 로봇을 따라가게 하려면 아래 3줄 추가
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    viewer.cam.trackbodyid = trunk_id
    viewer.cam.distance = 2.5
    while viewer.is_running() and data.time < sim_tf:  # 10초간 구동

        for sim_cnt in range(num_steps):
    
            current_time = sim_cnt * sim_ts
            X_current = get_13d_state_from_mujoco(model, data)

            # 🟦 상위 제어기 (MPC)
            if (sim_cnt % step_blue == 0):
                x_ref_traj = np.zeros((13 * horizon, 1))
                r_feet_traj = []
                yaw_traj = np.zeros(horizon)
                p_feet_raibert_wf = np.zeros((3,4))
                r_feet_raibert_wf = np.zeros((3,4))
                p_feet_raibert_del = np.array((3,1)) 

                # 회전 속도 (rad/s) 설정: 초당 약 30도 회전
                omega_z_des = 0.0 * DEG2RAD 
                
                # X, Y 이동 속도는 0으로 묶어둠 (제자리 유지)
                v_des = np.array([[0.5], [0.0], [0.0]]) 
                
                z_des = 0.65 # cfg.leg_length_straight*4/5
                # 기준점: 현재 위치(X,Y)와 현재 Yaw 각도
                current_x, current_y = X_current[3, 0], X_current[4, 0]
                p_com_cur_wf = X_current[3:6, 0:1]
                current_yaw = X_current[2, 0] # 상태 벡터의 [2]번 인덱스가 Yaw(ψ)
                
                for i in range(horizon):
                    # 위치(P): 현재 위치 그대로 고정 (이동 안 함)
                    x_ref_traj[i * 13 + 3, 0] = current_x + v_des[1,0] * (i * mpc_dt)
                    x_ref_traj[i * 13 + 4, 0] = current_y + v_des[2,0] * (i * mpc_dt)
                    x_ref_traj[i * 13 + 5, 0] = z_des # Z축 높이 고정
                    
                    # 속도(V): 선속도는 0으로 고정
                    x_ref_traj[i * 13 + 9 : i * 13 + 12, 0] = v_des.flatten()
                    
                    # [추가] 각속도(ω): Z축 방향(Yaw)으로만 각속도 부여
                    # (상태 벡터 [6:9]는 Roll, Pitch, Yaw의 각속도)
                    x_ref_traj[i * 13 + 6 : i * 13 + 8, 0] = 0.0 # Roll, Pitch 각속도 0
                    x_ref_traj[i * 13 + 8, 0] = omega_z_des      # Yaw 각속도 입력
                    
                    # [추가] 자세(Θ): Yaw 각도 예측값 누적
                    # (yaw_traj는 MPC 솔버 내부 회전행렬 계산을 위해 별도로 넘겨주는 배열)
                    predicted_yaw = current_yaw + omega_z_des * (i * mpc_dt)
                    yaw_traj[i] = predicted_yaw
                    
                    # 상태 벡터의 자세(Theta) 부분에도 Yaw 예측값 업데이트
                    x_ref_traj[i * 13 + 0, 0] = 0.0 # Roll
                    x_ref_traj[i * 13 + 1, 0] = 0.0 # Pitch
                    x_ref_traj[i * 13 + 2, 0] = predicted_yaw # Yaw
                    
                # ----------------------------------------------------
                # 2. ★ Raibert Heuristic 발판 위치 계산 ★
                # ----------------------------------------------------
                v_current = X_current[9:12, 0:1].copy() # 현재 CoM 선속도 (3x1)
                
                # CoM 기준 각 엉덩이(Hip) 관절의 위치 (3x4 행렬)
                hip_body_names = ["FR_hip", "FL_hip", "RR_hip", "RL_hip"]  # XML의 body 이름에 맞게 수정
                hip_location_wf = np.zeros((3, 4))

                for i, name in enumerate(hip_body_names):
                    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
                    hip_location_wf[:, i] = data.xpos[body_id].copy()  # World Frame 위치 (3,)

                # Raibert 공식 적용 (Numpy 브로드캐스팅으로 4다리 동시 계산!)
                p_feet_raibert_del = (T_stance / 2.0) * v_current

                # 미래에 착지할 상대 위치 (r = p_hip + p_foot_del)

                p_feet_raibert_wf = hip_location_wf + p_feet_raibert_del 
                p_feet_raibert_wf[2, :] = 0 # Z축은 지면을 향하므로 다리 길이로 고정
                r_feet_raibert_wf = p_feet_raibert_wf - p_com_cur_wf

                # ----------------------------------------------------
                # 3. MPC 미래 예측 궤적(Horizon) 구성
                # ----------------------------------------------------
                for k in range(horizon):
                    future_time = current_time + (k * mpc_dt)
                    future_phase = (future_time % gait_period) / gait_period
                    future_sa = get_contact_state(future_phase, gait_duty, gait_phase_offset)
                    
                    contact_sequence_horizon[:, k] = future_sa
                    
                    # 예측 구간 전체에 Raibert로 계산한 목표 착지점(p_foot_des_wf)을 복사해 넣습니다.
                    r_feet_traj.append(r_feet_raibert_wf.copy())

                # 4. MPC 풀이
                fmpc, umax = mpc_solver.solve(X_current, x_ref_traj, yaw_traj, r_feet_traj, contact_sequence_horizon)
                
                F_G = np.array(fmpc).reshape(4, 3).T

            # 🟥 상태 추정 및 스윙 제어기
            if (sim_cnt % step_red == 0):


                # Calculate Rotation Matrix in Red Loop
                Rw_b = get_Rotation_Matrix_Body2World(model, data)
                Rb_w = Rw_b.T

                # get foot position in world frame
                p_feet_wf, v_feet_wf, p_feet_bf, v_feet_bf = get_all_feet_kinematics(
                    model, data, foot_names, Rb_w
                )
                
                # p_feet_wf = gen_swing_traj(cur_time, gait_info, dt, p_feet_des_wf)
                # gait_info: gait_period, gait_duty, gait_phase_offset, T_swing
                
                # 현재 시간 기준의 정확한 위상(Phase) 계산
                current_phase = (current_time % gait_period) / gait_period
                # # 현재 다리의 접촉 상태 업데이트
                Sa_current = get_contact_state(current_phase, gait_duty, gait_phase_offset)

                # 2. 각 다리의 상태에 따른 위치 제어
                for i, f_name in enumerate(foot_names):
                    # --- [STANCE PHASE (지지기)] ---
                    if Sa_current[i] == 0: # ground
                        # 방금 전까지 스윙(공중)이었다가 처음 땅에 닿은 순간(Touchdown)
                        current_s[i] = 0.0
                        if swing_time_counter[i] > 0:
                            swing_time_counter[i] = 0.0 # 스윙 타이머 초기화
                        
                        # 💡 핵심: 지지기일 때는 r_feet_wf[:, i]의 값을 절대 갱신하지 않습니다!
                        # 발이 땅에 박혀있으므로 이전 월드 좌표를 그대로 유지합니다.
                        
                    # --- [SWING PHASE (스윙기)] ---
                    else:
                        # 방금 막 땅에서 떨어진 순간(Liftoff)
                        if swing_time_counter[i] == 0.0:
                            # 궤적의 출발점이 될 현재 발의 월드 좌표를 고정 저장
                            # 원래는 현재 상태를 받아와야함
                            swing_start_pos_wf[:, i] = p_feet_wf[:, i].copy()
                            # swing_start_pos_wf[2, i] = 0
                        
                        # 스윙 타이머 업데이트 및 진행률 s 계산
                        swing_time_counter[i] += dt_red
                        current_s[i] = (swing_time_counter[i] % T_swing) / T_swing
                        
                        # s = np.clip(s, 0.0, 1.0) # 0.0 ~ 1.0 사이로 강제 고정
                        # Raibert Heuristic을 이용한 목표 착지점 (P3) 예측
                        # 1) 현재 몸통 위치 기준 해당 다리 엉덩이(Hip)의 월드 좌표를 구함
                        # 2) 엉덩이 위치에서 로봇의 속도를 곱해 미래 착지점 투영
                        p3_target = p_feet_raibert_wf[:,i]
                        
                        # 베지에 곡선 제어점 세팅 (P0: 시작, P1: 이륙, P2: 착지 진입, P3: 목표)
                        p0_start = swing_start_pos_wf[:, i].flatten()                 # 발 떼기 (Start), 발 떼기 전 위치 넣기
                        p1 = p0_start + np.array([0, 0,  clearance_height])  # 위로 들어올림 (Control 1) 높이 설정하기
                        p2 = p3_target + np.array([0, 0, clearance_height]) # 앞으로 이동하며 고도 유지 (Control 2), 다음 발 디딤 위치 넣기

                        # 3차 베지에 곡선을 통해 현재 시점(s)의 스윙 발 위치 도출
                        p_feet_swing_traj_wf[:,i], v_feet_swing_traj_wf[:,i], a_feet_swing_traj_wf[:,i] = compute_bezier_with_kinematics(current_s[i], 
                                                                                                                                         [p0_start, p1, p2, p3_target], 
                                                                                                                                         T_swing=T_swing)
                        p_com_wf = X_current[3:6,0:1]
                        v_com_wf = X_current[9:12,0:1]
                        
                        # 5. 월드 목표 상태를 바디 좌표계(Body Frame) 목표 상태로 변환!
                        p_feet_swing_traj_bf[:, i] = Rb_w @ (p_feet_swing_traj_wf[:,i].flatten() - p_com_wf.flatten())
                        v_feet_swing_traj_bf[:, i] = Rb_w @ (v_feet_swing_traj_wf[:,i].flatten() - v_com_wf.flatten())
                        a_feet_swing_traj_bf[:, i] = Rb_w @ a_feet_swing_traj_wf[:,i].flatten()

            # =====================================================================
            # 🟩 Leg Position, Torque, Force Control (1kHz / 매 step_green 마다 실행)
            # =====================================================================
            if (sim_cnt % step_green == 0):
                Tm = np.zeros(12)  # 모터 지령 토크 (12,)
                
                # 1. 루프 시작 시 1회만 일괄 계산 (동역학, 회전행렬, 발 기구학, 자코비안)
                M, bias = get_dynamics_matrices(model, data)
                R_b2w = get_Rotation_Matrix_Body2World(model, data, "trunk")
                Rb_w = R_b2w.T # World -> Body 변환 행렬
                
                p_feet_wf, v_feet_wf, p_feet_bf, v_feet_bf = get_all_feet_kinematics(
                    model, data, foot_names, Rb_w
                )
                
                J_full_all, J_dot_all = get_all_feet_jacobians_and_derivatives(
                    model, data, foot_names, prev_J, dt_green
                )

                # 2. 4개 다리 순회 (0: FR, 1: FL, 2: RR, 3: RL)
                for i in range(4):
                    J_full_i = J_full_all[:, :, i]            # (3, 18)
                    J_dot_i = J_dot_all[:, :, i]              # (3, 18)
                    J_i = J_full_i[:, 6 + 3*i : 6 + 3*i + 3] # (3, 3) 조인트 자코비안

                    if Sa_current[i] == 0: # GROUND
                        # -------------------------------------------------------------
                        # [Stance Leg] MPC 지면 반발력(GRF) -> 모터 토크 변환
                        # -------------------------------------------------------------
                        f_mpc_i = -F_G[:,i]
                        tau_i = J_i.T @ f_mpc_i

                    else:
                        # -------------------------------------------------------------
                        # [Swing Leg] 동적 게인 스케줄링 & 베지에 궤적 추종
                        # -------------------------------------------------------------
                        p_foot_cur_bf = p_feet_bf[:, i]
                        v_foot_cur_bf = v_feet_bf[:, i]

                        # (1) 논문 Eq.(3) 적용 작업 공간 관성(Lambda) 및 동적 Kp, Kd 계산
                        Lambda_i = compute_operational_space_inertia(M, J_full_i)
                        Lambda_diag = np.diag(Lambda_i)
                        
                        Kp_dynamic = np.diag((omega_n ** 2) * Lambda_diag)
                        Kd_dynamic = np.diag(2.0 * zeta * omega_n * Lambda_diag)

                        # (2) 작업 공간 가상 피드백 힘 계산 (Body Frame)
                        f_task_bf = Kp_dynamic @ (p_feet_swing_traj_bf[:, i] - p_foot_cur_bf) + \
                                    Kd_dynamic @ (v_feet_swing_traj_bf[:, i] - v_foot_cur_bf)

                        # (3) 피드포워드 토크 계산
                        tau_ff_full = compute_swing_leg_feedforward_torque_clean(
                            M, bias, J_full_i, J_dot_i, a_feet_swing_traj_bf[:, i]
                        )
                        tau_ff_i = tau_ff_full[6 + 3*i : 6 + 3*i + 3] # 해당 다리 슬라이싱

                        # (4) 좌표계 맞춤: Body Frame 힘 -> World Frame 변환 후 자코비안 적용
                        f_task_wf = R_b2w @ f_task_bf
                        tau_i = J_i.T @ f_task_wf + tau_ff_i

                    # 최종 모터 토크 저장
                    Tm[3*i : 3*i + 3] = tau_i

                # 3. 다음 타임스텝을 위한 자코비안 버퍼 갱신 및 토크 인가
                prev_J = J_full_all.copy()
                data.ctrl[:] = Tm

            mujoco.mj_step(model, data)
            viewer.sync()
            
            if data.time >= next_render_time:
                renderer.update_scene(data, camera=cam)  # <- cam 객체 지정!
                pixels = renderer.render()
                frames.append(pixels.copy())
                next_render_time += render_interval
                    
            # ----------------------------------------------------
            # 데이터 로깅
            # ----------------------------------------------------
            # if (sim_cnt % step_vis == 0):
            history_xref.append(x_ref_traj[0:13].flatten())
            history_x.append(X_current.flatten())
            history_F_G.append(F_G.copy())
            history_Sa.append(Sa_current.copy())
            history_q.append(data.qpos[7:].copy())
            # history_p_feet_des_wf.append(p_feet_raibert_wf.copy())
            # history_r_feet_des_wf.append(r_feet_raibert_wf.copy())
            history_s.append(current_s.copy())
            history_tau.append(data.ctrl.copy())
            history_p_feet_des_wf.append(p_feet_swing_traj_wf.copy())
            history_p_feet_act_wf.append(p_feet_wf.copy())
            history_p_feet_des_bf.append(p_feet_swing_traj_bf.copy())
            history_p_feet_act_bf.append(p_feet_bf.copy())

            # if ((sim_cnt / sim_freq) % 1):
            #     print("simulation 1s 경과")


print("💾 시뮬레이션 데이터 저장 중...")

np.savez('sim_history.npz',
         x=np.array(history_x),
         xref=np.array(history_xref),
         F_G=np.array(history_F_G),
         Sa=np.array(history_Sa),
         q=np.array(history_q),
         p_feet_des_bf=np.array(history_p_feet_des_bf),
         p_feet_act_bf=np.array(history_p_feet_act_bf),
         p_feet_des_wf=np.array(history_p_feet_des_wf),
         p_feet_act_wf=np.array(history_p_feet_act_wf),
         s=np.array(history_s),
         tau=np.array(history_tau))

print(f"📊 로깅된 데이터 개수: {len(history_x)} 개, 시뮬레이션 시간: {sim_tf} 초", )

print("✅ 데이터 저장 완료! (sim_history.npz)")

print("동영상 렌더링 중...")
media.write_video('quadruped_sim_1x_speed.mp4', frames, fps=framerate)
print("저장 완료!")
