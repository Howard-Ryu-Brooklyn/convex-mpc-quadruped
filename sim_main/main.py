import numpy as np
import config as cfg
from config import get_q, get_r_feet_bf, get_13d_state, compute_bezier
from visualization import animate_quadruped, plot_state_tracking, plot_force_and_contact, plot_leg_angles, plot_foot_comparison, plot_foot_trajectory_3d, plot_r_feet_wf_over_time, plot_foot_trajectory_3d, plot_swing_progress
import dynamics as SRB_model
from dynamics import get_rotation_matrix  
import convex_mpc as cvx_mpc
from gait_planning import get_gait_parameters, get_contact_state
import math

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
sim_tf = 5 # [sec] 전체 시뮬레이션 시간
num_steps = round(sim_tf / sim_ts)

step_blue = round(sim_freq / loop_blue_hz)   
step_red = round(sim_freq / loop_red_hz)     
step_vis = round(sim_freq / 30) # 시각화는 30FPS

# ==========================================
# 2. 초기 상태 및 변수 선언
# ==========================================
# 다리가 구부러진 상태를 시뮬레이션 하기 위해 높이를 다리가 쫙펴진 높이의 절반으로 설정
target_height_com = cfg.leg_length_straight/2 #from com to ground
P0 = np.array([[0.0], [0.0], [target_height_com]]) 
V0 = np.zeros((3, 1))
ANG0 = np.array([[0.0*DEG2RAD], [0.0*DEG2RAD], [0.0*DEG2RAD]]) # Yaw 90도 시작 원하시면 [0, 0, 90*DEG2RAD]
ANGVEL0 = np.zeros((3, 1))
X0 = get_13d_state(ANG0, P0, ANGVEL0, V0)

g_vec = np.array([0,0,abs(cfg.m*cfg.gz/4)])
F_G0 = np.tile(g_vec.reshape(3,1),(1,4))

# 엉덩이 모터가 com가 동일 높이에 있다고 가능
# 초기 변수
pfoot_bf0 = np.array([
        [ 0, 0, -target_height_com], # FR
        [ 0, 0, -target_height_com], # FL
        [ 0, 0, -target_height_com], # RR
        [ 0, 0, -target_height_com]  # RL
    ]).T # hip의 바로 아래 발이 위치한다고 가정
Rw_b0 = get_rotation_matrix(ANG0[0,0],ANG0[1,0],ANG0[2,0])

Q0 = get_q(target_height_com)
QDOT0 = np.zeros((3,4)) 

# 시뮬레이터 객체 생성
inertia_diag_list = [cfg.Ixx, cfg.Iyy, cfg.Izz] 
inertia_leg_diag_list = [cfg.Ilink_hip, cfg.Ilink_upper, cfg.Ilink_lower]

link_info = [cfg.link_upper, cfg.link_lower, cfg.link_hip]

# 보행 스케줄러
gait_mode = ["standing", "trotting", "flying_trot", "bounding","galloping"]
gait_period, gait_duty, gait_phase_offset = get_gait_parameters(gait_mode[3])
Sa_current = np.zeros(4)
# 스윙 상태 추적을 위한 타이머 및 저장 공간
swing_time_counter = np.zeros(4)      # 각 다리가 공중에 떠 있는 시간을 기록 (0.0으로 초기화)
swing_start_pos_wf = np.zeros((3, 4)) # 발이 땅에서 떨어진 순간의 위치를 영구 저장

# Gait 파라미터 (예: Trot의 경우 보통 0.15초 ~ 0.25초)
T_swing = gait_period * (1-gait_duty)   # 발이 공중에 떠서 이동하는 목표 시간 (150ms)
T_stance = gait_period * gait_duty  # 발이 땅을 딛고 있는 목표 시간 (150ms)
clearance_height = 0.05 # 발을 들어올리는 최대 포물선 높이 (10cm)


mpc_dt = 1/loop_blue_hz
horizon = 10 # math.ceil(gait_period/mpc_dt) # 보행 한주기와 동일하게 설정
print('horizon', horizon)
# Ttrot = 0.33s  10 timestep =  (mpc주기에 맞춰 타임스텝을 설정)
# Thound = 0.5s  16 timestep
# MPC는 최소 보행 한주기 정도는 내다 볼 수 있게 설계해야함
# horizon * mpc_dt = Tgait
# MPC 계산할때는 위 주기에서 10~16스텝 쪼갠걸로 행렬 계산
L_weights = cvx_mpc.ConvexMPC.build_state_weight(cfg.L_w_th, cfg.L_w_z, cfg.L_w_yr, cfg.L_w_v)

p_feet_del = np.array((3,1)) 
x_ref_traj = np.zeros((13 * horizon, 1))
r_feet_traj = []

# 시각화 데이터 로깅 리스트
history_p, history_R, history_r_feet_wf, history_F_G = [], [], [], []
# 상태 플롯(Plot)용 로깅 리스트 추가
history_x, history_xref = [], []
history_Sa = []
history_q = []
history_s = []
history_p_feet_des_wf = []
history_p_feet_wf = []
history_r_feet_des_wf = []
history_p_feet_local = []
p_feet_local = np.zeros((3,4))
current_s = np.zeros(4)
# ==========================================
# 3. 메인 시뮬레이션 제어 루프
# ==========================================
print("🚀 4족보행 로봇 제어 시뮬레이션 시작...")

for sim_cnt in range(num_steps):
    
    if sim_cnt == 0:
        X_current = X0.copy()
        F_G = F_G0

        pfoot_bf = pfoot_bf0
        r_feet_bf = get_r_feet_bf(pfoot_bf) # vector from com to foot

        r_feet_wf = Rw_b0 @ r_feet_bf
        r_feet_des_wf = r_feet_wf
        hip_location_wf = P0 + Rw_b0 @ cfg.hip_location_bf

        p_feet_wf = P0 + r_feet_wf
        p_feet_des_wf = r_feet_wf.copy()
        robot = SRB_model.SRBDynamics(sim_ts, cfg.m, inertia_diag_list, cfg.gz, P0, V0, ANG0, ANGVEL0, link_info, Q0, QDOT0, cfg.hip_location_bf, r_feet_wf, inertia_leg_diag_list)

        mpc = cvx_mpc.ConvexMPC(cfg.m, inertia_diag_list, cfg.gz, mpc_dt, horizon, L_weights, cfg.K_w_f, cfg.μ, cfg.fmin, cfg.fmax)
    else:
        pass    
    
        current_time = sim_cnt * sim_ts
        X_current = get_13d_state(robot.ANG, robot.P, robot.ANGVEL, robot.V)
        
        if np.isnan(X_current).any() or np.isinf(X_current).any():
            print(f"🚨 [경고] 상태 벡터(X_current)에 NaN/Inf 발생! Step: {sim_cnt}")
            break
        if np.isnan(r_feet_traj).any() or np.isinf(r_feet_traj).any():
            print(f"🚨 [경고] 발 위치(r_feet_traj)에 NaN/Inf 발생! Step: {sim_cnt}")
            break

        # 🟦 상위 제어기 (MPC)
        if (sim_cnt % step_blue == 0):
            
            # 목표 궤적 산출
            # 1. 목표 궤적 및 속도 설정
            v_des = np.array([[1.5], [0.0], [0.0]]) # X축으로 0.5m/s 전진
            
            # 🚨 추가: 예측의 기준점이 될 '현재 로봇의 실제 위치'
            current_x = X_current[3, 0].copy()
            current_y = X_current[4, 0].copy()
            
            for i in range(horizon):
                # 🚨 핵심: 현재 위치에서 목표 속도 * 시간(i * mpc_dt) 만큼 적분하여 미래 위치 세팅
                x_ref_traj[i * 13 + 3, 0] = current_x + v_des[0, 0] * (i * mpc_dt) # 미래 Px
                x_ref_traj[i * 13 + 4, 0] = current_y + v_des[1, 0] * (i * mpc_dt) # 미래 Py
                x_ref_traj[i * 13 + 5, 0] = target_height_com #+ 0.5*cfg.leg_length_straight/2*np.sin(2*np.pi*0.1*current_time) # Z축은 고정 높이
                
                # 목표 선속도 세팅
                x_ref_traj[i * 13 + 9 : i * 13 + 12, 0] = v_des.flatten()
                
            yaw_traj = np.zeros(horizon)

            # ----------------------------------------------------
            # 2. ★ Raibert Heuristic 발판 위치 계산 ★
            # ----------------------------------------------------
            v_current = X_current[9:12, 0:1].copy() # 현재 CoM 선속도 (3x1)
            
            # CoM 기준 각 엉덩이(Hip) 관절의 위치 (3x4 행렬)
            hip_location_wf = robot.P + robot.RW_B @ cfg.hip_location_bf # (3x3) @ (3x4) = (3x4)
            
            # Raibert 공식 적용 (Numpy 브로드캐스팅으로 4다리 동시 계산!)
            p_feet_del = (T_stance / 2.0) * v_current

            # 미래에 착지할 상대 위치 (r = p_hip + p_foot_del)

            p_feet_des_wf = hip_location_wf + p_feet_del 
            p_feet_des_wf[2, :] = 0 #-x_ref_traj[i * 13 + 5, 0] # Z축은 지면을 향하므로 다리 길이로 고정
            r_feet_des_wf = p_feet_des_wf - robot.P

            # ----------------------------------------------------
            # 3. MPC 미래 예측 궤적(Horizon) 구성
            # ----------------------------------------------------
            
            contact_sequence = np.zeros((4, horizon)) 
            
            for k in range(horizon):
                future_time = current_time + (k * mpc_dt)
                future_phase = (future_time % gait_period) / gait_period
                future_sa = get_contact_state(future_phase, gait_duty, gait_phase_offset)
                
                contact_sequence[:, k] = future_sa
                
                # 예측 구간 전체에 Raibert로 계산한 목표 착지점(p_foot_des_wf)을 복사해 넣습니다.
                r_feet_traj.append(r_feet_des_wf.copy())

            # 4. MPC 풀이
            fmpc, umax = mpc.solve(X_current, x_ref_traj, yaw_traj, r_feet_traj, contact_sequence)
            
            F_G = np.array(fmpc).reshape(4, 3).T

        # 🟥 상태 추정 및 스윙 제어기
        if (sim_cnt % step_red == 0):
            # Red loop의 실제 dt (예: 1000Hz면 0.001)
            
            dt_red = sim_ts * step_red 

            # 현재 시간 기준의 정확한 위상(Phase) 계산
            current_phase = (current_time % gait_period) / gait_period
            # # 현재 다리의 접촉 상태 업데이트
            Sa_current = get_contact_state(current_phase, gait_duty, gait_phase_offset)

            # 2. 각 다리의 상태에 따른 위치 제어
            for i in range(4):
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
                    
                    # 스윙 타이머 업데이트 및 진행률 s 계산
                    swing_time_counter[i] += dt_red
                    s = (swing_time_counter[i] / T_swing)
                    s_clip = np.clip(s, 0.0, 1.0)
                    current_s[i] = s_clip
                    # s = np.clip(s, 0.0, 1.0) # 0.0 ~ 1.0 사이로 강제 고정
                    # Raibert Heuristic을 이용한 목표 착지점 (P3) 예측
                    # 1) 현재 몸통 위치 기준 해당 다리 엉덩이(Hip)의 월드 좌표를 구함
                    # 2) 엉덩이 위치에서 로봇의 속도를 곱해 미래 착지점 투영
                    p3_target = p_feet_des_wf[:,i].copy()                           # 발 딛기 (Target)
                    
                    # 베지에 곡선 제어점 세팅 (P0: 시작, P1: 이륙, P2: 착지 진입, P3: 목표)
                    p0_start = swing_start_pos_wf[:, i]                 # 발 떼기 (Start), 발 떼기 전 위치 넣기
                    p1 = p0_start + np.array([0, 0,  clearance_height])  # 위로 들어올림 (Control 1) 높이 설정하기
                    p2 = p3_target + np.array([0, 0, clearance_height]) # 앞으로 이동하며 고도 유지 (Control 2), 다음 발 디딤 위치 넣기
            
                    # 3차 베지에 곡선을 통해 현재 시점(s)의 스윙 발 위치 도출
                    p_feet_wf[:, i] = cfg.compute_bezier(s, [p0_start, p1, p2, p3_target])
        # 🟩 

        r_feet_wf = p_feet_wf - robot.P
        # 물리 엔진 스텝 업데이트 (Single Rigid Body Dynamics)
        p_feet_local = robot.step(F_G, r_feet_wf)

    # ----------------------------------------------------
    # 데이터 로깅
    # ----------------------------------------------------
    if (sim_cnt % step_vis == 0):
        history_R.append(robot.RW_B.copy())
        history_r_feet_wf.append(r_feet_wf.copy())
        history_F_G.append(F_G.copy())
        history_x.append(X_current.flatten())
        history_xref.append(x_ref_traj[0:13].flatten())
        history_Sa.append(Sa_current.copy())
        history_q.append(robot.Q.copy())
        history_p_feet_des_wf.append(p_feet_des_wf.copy())
        history_r_feet_des_wf.append(r_feet_des_wf.copy())
        history_p_feet_local.append(p_feet_local.copy())
        history_p_feet_wf.append(p_feet_wf.copy())
        history_s.append(current_s.copy())

print("✅ 연산 완료! 애니메이션 렌더링 시작...")

print("📊 상태 추종 그래프를 생성합니다...")
# 시각화 저장 주기(step_vis)에 맞춰서 dt를 넘겨줍니다. (예: 30Hz 로깅이면 dt=0.033)
plot_state_tracking(history_x, history_xref, dt=sim_ts * step_vis)

plot_force_and_contact(history_F_G, history_Sa, sim_tf)

plot_leg_angles(history_q, sim_ts)

plot_foot_comparison(history_r_feet_des_wf, history_r_feet_wf, dt=0.033, leg_idx=0)

plot_foot_trajectory_3d(history_p_feet_wf, history_Sa=history_Sa, history_x=history_x, history_p_feet_des_wf=history_p_feet_des_wf)
plot_r_feet_wf_over_time(history_r_feet_wf, dt=0.033)

# 스윙 위상(s) 그래프 출력
plot_swing_progress(history_s, dt=sim_ts * step_vis)

# ==========================================
# 4. 애니메이션 생성
# ==========================================
anim_html = animate_quadruped(
        np.array(history_x), 
        np.array(history_R), 
        np.array(history_r_feet_wf),
        np.array(history_F_G),
        cfg.body_length,
        cfg.body_width,
        dt=0.033,
        save=True,
        history_q=np.array(history_q),
        link_lengths=(cfg.link_hip, cfg.link_upper, cfg.link_lower)
    )

anim_html