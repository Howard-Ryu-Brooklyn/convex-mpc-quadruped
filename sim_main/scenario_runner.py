"""회귀 기준선 캡처용 테스트 하네스.

⚠️ 이 파일은 main.ipynb의 셀을 **로직 변경 없이** 함수로 감싼 것입니다.
   구조 개선은 Step 1에서 합니다. 여기서 로직을 손대면 기준선의 의미가 사라집니다.
"""
import numpy as np
import config as cfg
from bezier import bezier
from config import get_13d_state
from kinematics import get_q, get_r_feet_bf
from planning import (
    build_contact_schedule,
    build_horizon_plan,
    build_reference_trajectory,
    raibert_footholds,
)
import dynamics as SRB_model
from rotations import rpy_to_matrix
import convex_mpc as cvx_mpc
from gait_planning import get_gait_parameters, get_contact_state

DEG2RAD = np.pi / 180

def run_scenario(
    gait_name: str = "trotting",
    v_des_x: float = 0.0,          # [m/s] 월드 X 방향 목표 속도
    v_des_y: float = 0.0,          # [m/s]
    omega_z_deg_s: float = 0.0,    # [deg/s] 목표 yaw 각속도
    duration_s: float = 3.0,
    horizon: int = 10,
    mpc_hz: int = 30,
    red_hz: int = 1000,
    sim_hz: int = 9000,
    log_hz: int = 30,
    clearance_height: float = 0.05,
) -> dict[str, np.ndarray]:
    """시나리오 하나를 끝까지 돌리고 로깅 배열을 딕셔너리로 반환합니다.

    main.ipynb의 셀과 **수치적으로 동일한 결과**를 내야 합니다.
    (v_des_x = v_des_y = 0, omega_z_deg_s = 20 일 때 노트북과 일치)
    """
    # --- 여기부터 노트북 셀을 그대로 옮긴 부분 ---
    #     (파라미터 선언 → 초기화 → for 루프 → 로깅)
    #     플로팅/애니메이션 코드만 제거합니다.

    # utils
    DEG2RAD = np.pi / 180

    # ==========================================
    # 1. 시뮬레이션 파라미터 및 제어 주기 설정
    # ==========================================
    loop_blue_hz = mpc_hz    # [Hz] MPC 연산 주기 25~50
    loop_red_hz = red_hz   # [Hz] 상태 추정/스윙 제어 주기
    loop_green_hz = 4500 # [Hz] 저수준 제어 주기
    loop_sim_hz = sim_hz   # [Hz] 물리 엔진(동역학) 주기

    sim_freq = loop_sim_hz
    sim_ts = 1 / sim_freq
    sim_tf = duration_s # [sec] 전체 시뮬레이션 시간
    
    num_steps = round(sim_tf / sim_ts)

    step_blue = round(sim_freq / loop_blue_hz)   
    step_red = round(sim_freq / loop_red_hz)     
    step_vis = round(sim_freq / log_hz) # 시각화는 30FPS

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
    Rw_b0 = rpy_to_matrix(ANG0[0,0],ANG0[1,0],ANG0[2,0])

    Q0 = get_q(target_height_com)
    QDOT0 = np.zeros((3,4)) 

    # 시뮬레이터 객체 생성
    inertia_diag_list = [cfg.Ixx, cfg.Iyy, cfg.Izz] 
    inertia_leg_diag_list = [cfg.Ilink_hip, cfg.Ilink_upper, cfg.Ilink_lower]

    link_info = [cfg.link_upper, cfg.link_lower, cfg.link_hip]

    # 보행 스케줄러
    gait_period, gait_duty, gait_phase_offset = get_gait_parameters(gait_name)

    Sa_current = np.zeros(4)
    # 스윙 상태 추적을 위한 타이머 및 저장 공간
    swing_time_counter = np.zeros(4)      # 각 다리가 공중에 떠 있는 시간을 기록 (0.0으로 초기화)
    swing_start_pos_wf = np.zeros((3, 4)) # 발이 땅에서 떨어진 순간의 위치를 영구 저장

    # Gait 파라미터 (예: Trot의 경우 보통 0.15초 ~ 0.25초)
    T_swing = gait_period * (1-gait_duty)   # 발이 공중에 떠서 이동하는 목표 시간 (150ms)
    T_stance = gait_period * gait_duty  # 발이 땅을 딛고 있는 목표 시간 (150ms)


    mpc_dt = 1/loop_blue_hz
    # Ttrot = 0.33s  10 timestep =  (mpc주기에 맞춰 타임스텝을 설정)
    # Thound = 0.5s  16 timestep
    # MPC는 최소 보행 한주기 정도는 내다 볼 수 있게 설계해야함
    # horizon * mpc_dt = Tgait
    # MPC 계산할때는 위 주기에서 10~16스텝 쪼갠걸로 행렬 계산
    L_weights = cvx_mpc.ConvexMPC.build_state_weight(cfg.L_w_th, cfg.L_w_z, cfg.L_w_yr, cfg.L_w_v)

    # 첫 MPC 틱(sim_cnt = step_blue) 이전에도 로깅이 x_ref_traj 를 읽으므로
    # 사전 초기화가 필요하다. MPC 브랜치에서 매 틱 새 배열로 교체된다.
    x_ref_traj = np.zeros((13 * horizon, 1))
    yaw_traj = np.zeros(horizon)

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
    diverged = False
    max_s = 0
    # ==========================================
    # 3. 메인 시뮬레이션 제어 루프
    # ==========================================

    for sim_cnt in range(num_steps):
        
        if sim_cnt == 0:
            X_current = X0.copy()
            # TODO(Step 1-6): 첫 MPC 틱(sim_cnt = step_blue) 이전 300 스텝은
            #   MPC 해가 없어 중력 보상값을 하드코딩해 쓴다. 스윙 다리에도
            #   힘이 실리므로 물리적으로 맞지 않다. Runner 에서는 sim_cnt=0 에
            #   MPC 를 한 번 풀고 시작하도록 정리한다.
            F_G = F_G0

            pfoot_bf = pfoot_bf0
            r_feet_bf = get_r_feet_bf(pfoot_bf) # vector from com to foot

            r_feet_wf = Rw_b0 @ r_feet_bf
            r_feet_des_wf = r_feet_wf
            hip_location_wf = P0 + Rw_b0 @ cfg.hip_location_bf

            p_feet_wf = P0 + r_feet_wf
            # 결함 5 — p_feet_des_wf 는 '절대 위치'다.
            #   r_feet_wf(CoM 기준 상대 벡터)를 넣으면 z 가 -0.34 가 되어,
            #   첫 MPC 틱 이전(33ms) 스윙 궤적의 목표점이 지면 34cm 아래를
            #   향한다. 단위가 같아서(m) 대입이 조용히 성립한 사고다.
            #   1-1 에서 FootState 가 pos_W 와 rel_com_W 를 별도 필드로
            #   나눈 이유가 이것이며, 그 타입을 쓰면 애초에 불가능하다.
            p_feet_des_wf = p_feet_wf.copy()
            robot = SRB_model.SRBDynamics(sim_ts, cfg.m, inertia_diag_list, cfg.gz, P0, V0, ANG0, ANGVEL0, link_info, Q0, QDOT0, cfg.hip_location_bf, r_feet_wf, inertia_leg_diag_list)

            mpc = cvx_mpc.ConvexMPC(cfg.m, inertia_diag_list, cfg.gz, mpc_dt, horizon, L_weights, cfg.K_w_f, cfg.MU_FRICTION, cfg.fmin, cfg.fmax)
        else:
            pass    
        
            current_time = sim_cnt * sim_ts
            X_current = get_13d_state(robot.ANG, robot.P, robot.ANGVEL, robot.V)
            
            if np.isnan(X_current).any() or np.isinf(X_current).any():
                diverged = True
                break
            # r_feet_traj는 이제 MPC 틱마다 새로 만들어지므로 이 시점에 존재하지 않는다.
            # 애초에 검사해야 할 것은 파생 리스트가 아니라 그 원천인 r_feet_des_wf다.
            if not np.isfinite(r_feet_des_wf).all():
                diverged = True
                break

            # 🟦 상위 제어기 (MPC)
            if (sim_cnt % step_blue == 0):
                
                omega_z_des = omega_z_deg_s * DEG2RAD
                v_des = np.array([[v_des_x], [v_des_y], [0.0]])

                # 1) 참조 상태 궤적 (속도 추종 모드 — planning.py 의 설계 메모 참조)
                x_ref_traj, yaw_traj = build_reference_trajectory(
                    X_current, v_des, omega_z_des, target_height_com, horizon, mpc_dt
                )

                # 2) Raibert 발판 계획.
                #    p_feet_des_wf 는 red 루프의 스윙 궤적 목표점으로도 쓰인다.
                p_feet_des_wf, r_feet_des_wf = raibert_footholds(
                    robot.P, robot.RW_B, X_current[9:12, 0:1].copy(), T_stance
                )

                # 3) horizon 각 스텝의 접촉 스케줄 예측
                is_stance_schedule = build_contact_schedule(
                    current_time, gait_period, gait_duty, gait_phase_offset,
                    horizon, mpc_dt,
                )

                # 4) horizon 조립.
                #    반환값이므로 '이전 호출의 값이 남아 있다'가 존재할 수 없다.
                #    결함 2 를 규율이 아니라 구조로 막는 지점이다.
                plan = build_horizon_plan(
                    x_ref=x_ref_traj,
                    yaw_ref=yaw_traj,
                    is_stance_schedule=is_stance_schedule,
                    p_feet_now_W=p_feet_wf,
                    p_feet_des_W=p_feet_des_wf,
                    is_stance_now=(np.asarray(Sa_current) == 0),
                )

                fmpc, umax = mpc.solve(
                    X_current, plan.x_ref, plan.yaw_ref, plan.r_feet_W, plan.contact_legacy
                )

                # 결함 7 — 접촉 마스크.
                # OSQP 는 수치 솔버라 fz in [0,0] 제약을 허용오차 안에서만 만족시킨다.
                # 그 결과 스윙 다리에 ~1e-4 N (때로는 음수) 의 잔류력이 남고,
                # 마스킹하지 않으면 그대로 플랜트의 tau = r x f 에 들어간다.
                # 시뮬레이션에서는 미세하지만 실기에서는 공중에 뜬 다리에 토크
                # 지령이 새는 것이고, 원인이 '솔버 수렴 오차'라 로그만 봐서는
                # 절대 찾을 수 없다.
                #
                # 물리적으로 반드시 성립해야 하는 조건은 솔버에 맡기지 않고
                # 출력단에서 강제한다. 마스크를 plan 에서 가져오므로 접촉 상태의
                # 출처가 하나로 유지된다 (별도 변수를 쓰면 갈라진다).
                F_G = np.array(fmpc).reshape(4, 3).T * plan.is_stance[:, 0]

            # 🟥 상태 추정 및 스윙 제어기
            if (sim_cnt % step_red == 0):
                # Red loop의 실제 dt (예: 1000Hz면 0.001)
                
                dt_red = sim_ts * step_red 

                # p_feet_wf = gen_swing_traj(cur_time, gait_info, dt, p_feet_des_wf)
                # gait_info: gait_period, gait_duty, gait_phase_offset, T_swing
                
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
                        max_s = max(max_s, s)
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
                        p_feet_wf[:, i] = bezier(s_clip, [p0_start, p1, p2, p3_target])
            # 🟩 

            r_feet_wf = p_feet_wf - robot.P
            # 물리 엔진 스텝 업데이트 (Single Rigid Body Dynamics)
            p_feet_local = robot.step(F_G, r_feet_wf)
            # r_feet_wf, X, p_feet_wf
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
        n_log = len(history_x)

    return {
        "completed":    not diverged,               # 끝까지 돌았는가
        "diverged_at_s": None if not diverged else sim_cnt * sim_ts,
        "t":        np.arange(n_log)*(step_vis*sim_ts),
        "com_pos":  np.asarray(history_x)[:, 3:6],
        "com_vel":  np.asarray(history_x)[:, 9:12],
        "rpy":      np.asarray(history_x)[:, 0:3],
        "omega":    np.asarray(history_x)[:, 6:9],
        "grf":      np.asarray(history_F_G),
        "feet_W":   np.asarray(history_p_feet_wf),
        "contact":  np.asarray(history_Sa),
        "q":        np.asarray(history_q),
        "max_s":    np.asarray(max_s),
    }