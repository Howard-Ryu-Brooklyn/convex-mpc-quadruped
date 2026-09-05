"""회귀 기준선 캡처용 테스트 하네스.

⚠️ 이 파일은 main.ipynb의 셀을 **로직 변경 없이** 함수로 감싼 것입니다.
   구조 개선은 Step 1에서 합니다. 여기서 로직을 손대면 기준선의 의미가 사라집니다.
"""
import numpy as np
import config as cfg
from bezier import bezier
from clock import MultiRateClock
from history import HistoryLogger
from robot_types import ControlCommand
from kinematics import get_q, get_r_feet_bf
from swing import SwingTrajectoryGenerator
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
    # 1. 제어 주기
    # ==========================================
    clock = MultiRateClock(
        sim_hz=sim_hz, mpc_hz=mpc_hz, swing_hz=red_hz,
        log_hz=log_hz, duration_s=duration_s,
    )
    mpc_dt = clock.mpc_dt

    # ==========================================
    # 2. 초기 상태 및 변수 선언
    # ==========================================
    # 다리가 구부러진 상태를 시뮬레이션 하기 위해 높이를 다리가 쫙펴진 높이의 절반으로 설정
    target_height_com = cfg.leg_length_straight/2 #from com to ground
    P0 = np.array([[0.0], [0.0], [target_height_com]]) 
    V0 = np.zeros((3, 1))
    ANG0 = np.array([[0.0*DEG2RAD], [0.0*DEG2RAD], [0.0*DEG2RAD]]) # Yaw 90도 시작 원하시면 [0, 0, 90*DEG2RAD]
    ANGVEL0 = np.zeros((3, 1))
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

    # 초기 접촉 상태도 스케줄러가 답이다. zeros(4) 는 '전부 스탠스'라는 거짓말이다.
    Sa_current = get_contact_state(0.0, gait_duty, gait_phase_offset)

    # Gait 파라미터 (예: Trot의 경우 보통 0.15초 ~ 0.25초)
    T_swing = gait_period * (1-gait_duty)   # 발이 공중에 떠서 이동하는 목표 시간 (150ms)
    T_stance = gait_period * gait_duty  # 발이 땅을 딛고 있는 목표 시간 (150ms)

    # 스윙 궤적 생성기가 이지 시점 위치와 경과 시간을 소유한다.
    swing = SwingTrajectoryGenerator(T_swing, clearance_height)


    # Ttrot = 0.33s  10 timestep =  (mpc주기에 맞춰 타임스텝을 설정)
    # Thound = 0.5s  16 timestep
    # MPC는 최소 보행 한주기 정도는 내다 볼 수 있게 설계해야함
    # horizon * mpc_dt = Tgait
    # MPC 계산할때는 위 주기에서 10~16스텝 쪼갠걸로 행렬 계산
    L_weights = cvx_mpc.ConvexMPC.build_state_weight(cfg.L_w_th, cfg.L_w_z, cfg.L_w_yr, cfg.L_w_v)

    # x_ref_traj / p_feet_local / current_s 의 사전 초기화는 사라졌다.
    # sim_cnt = 0 이 모든 주기(MPC/스윙/로깅)의 틱이므로 로깅 시점에는
    # 반드시 실제 값이 들어와 있다. '로깅이 크래시 나지 않게 0 을 채워둔다'는
    # 노트북 시절의 임시방편이었고, 그 0 이 로그 첫 줄에 그대로 찍히고 있었다.
    log = HistoryLogger()
    diverged = False
    # 첫 스텝은 반드시 풀어야 하므로 '어떤 접촉 상태와도 다른' 값으로 시작한다.
    stance_mask_at_solve = None
    n_event_solves = 0        # 진단: 주기 밖에서 추가로 푼 횟수
    # ==========================================
    # 3. 초기 발 위치와 플랜트/제어기 생성
    #    (sim_cnt 에 의존하지 않는 값이므로 루프 밖에 둔다)
    # ==========================================

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
    robot = SRB_model.SRBDynamics(clock.dt, cfg.m, inertia_diag_list, cfg.gz, P0, V0, ANG0, ANGVEL0, link_info, Q0, QDOT0, cfg.hip_location_bf, r_feet_wf, inertia_leg_diag_list)

    mpc = cvx_mpc.ConvexMPC(cfg.m, inertia_diag_list, cfg.gz, mpc_dt, horizon, L_weights, cfg.K_w_f, cfg.MU_FRICTION, cfg.fmin, cfg.fmax)

    # ==========================================
    # 4. 메인 시뮬레이션 제어 루프
    # ==========================================
    # sim_cnt = 0 도 MPC 틱(0 % mpc_every == 0)이다. 따라서 첫 스텝부터
    # 실제 QP 해로 구동된다. 예전에는 첫 300 스텝(33 ms)을 mg/4 하드코딩
    # 값으로 채웠는데, 그 값은 공중에 뜬 스윙 다리에도 105 N 을 싣는
    # 물리적으로 성립하지 않는 입력이었다.
    for sim_cnt in range(clock.n_steps):
        current_time = clock.time_at(sim_cnt)

        # ── 접촉 상태는 매 물리 스텝 평가한다 (결함 10) ─────────────────
        # 예전에는 스윙 루프(1kHz) 안에서만 갱신했다. get_contact_state 는
        # 비교 네 번이라 9kHz 로 돌려도 비용이 없고, 대신 '지금 이 순간
        # 어느 발이 땅에 있는가'가 항상 정확해진다. 접촉은 제어 주기의
        # 사정을 봐주지 않는다.
        current_phase = (current_time % gait_period) / gait_period
        Sa_current = get_contact_state(current_phase, gait_duty, gait_phase_offset)
        is_stance_now = np.asarray(Sa_current) == 0
        # 살아있는 참조 4개를 꺼내 조립하는 대신 불변 스냅샷 하나를 받는다.
        # test_observe_to_mpc_vector_matches_legacy_layout 가 두 경로의
        # 비트 단위 동등성을 증명해두었으므로 이 교체는 기계적이다.
        #
        # (3,1) 로 reshape 하는 이유 — 형상 부채
        #   이 루프는 전부 (3,1) 열벡터 규약으로 쓰여 있는데 RobotState 는
        #   numpy 관용에 맞춰 (3,) 을 쓴다. 섞으면 조용한 브로드캐스팅
        #   버그가 난다: (3,) + (3,4) 는 에러 없이 '틀린 축으로' 퍼진다.
        #   루프 전체를 (3,) 로 옮기는 것은 별도 단계로 다룬다.
        X_current = robot.observe().to_mpc_vector().reshape(13, 1)
        
        if np.isnan(X_current).any() or np.isinf(X_current).any():
            diverged = True
            break
        # r_feet_traj는 이제 MPC 틱마다 새로 만들어지므로 이 시점에 존재하지 않는다.
        # 애초에 검사해야 할 것은 파생 리스트가 아니라 그 원천인 r_feet_des_wf다.
        if not np.isfinite(r_feet_des_wf).all():
            diverged = True
            break

        # 🟦 상위 제어기 (MPC)
        #
        # 결함 10 해결 — 주기 구동에 이벤트 구동을 더한다.
        # ────────────────────────────────────────────────────────────
        # MPC 를 33.3ms 격자에서만 풀면, 그 사이에 일어난 접촉 전이는 다음
        # 틱까지 반영되지 않는다. 이지한 다리에 힘이 실린 채 최대 16.7ms 가
        # 적분되고(S1 3초에서 FR/RL 각 0.1초), 그 결함의 가시성은 보행 주기와
        # MPC 주기의 우연한 정수비에 좌우된다.
        #
        # Sa_current 로 힘을 다시 마스킹하는 것은 개악이다. 이지 순간 그
        # 다리들이 체중 전부를 지지하므로, 0 으로 만들면 지면 반력이 통째로
        # 사라져 자유낙하한다. 힘을 지우는 게 아니라 **다시 풀어야** 한다.
        #
        # 접촉은 이산 사건이다. 이산 사건을 고정 주기로 표본화하면 지연은
        # 반드시 남고, 주기를 올리는 것은 지연을 줄일 뿐 없애지 못한다.
        # 사건이 일어난 그 순간에 반응하는 것이 유일한 근본 해법이다.
        contact_changed = not np.array_equal(is_stance_now, stance_mask_at_solve)
        if contact_changed and not clock.is_mpc_tick(sim_cnt):
            n_event_solves += 1          # 주기 틱과 겹치지 않은 추가 풀이만 센다
        if clock.is_mpc_tick(sim_cnt) or contact_changed:
            
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
                is_stance_now=is_stance_now,
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
            # 마스크의 출처는 '지금 이 순간의 접촉 상태' 하나다.
            # plan.is_stance[:, 0] 은 같은 시각·같은 공식이라 값이 같지만,
            # 출처를 둘로 두면 언젠가 갈라진다.
            stance_mask_at_solve = is_stance_now
            F_G = np.array(fmpc).reshape(4, 3).T * stance_mask_at_solve

        # 🟥 상태 추정 및 스윙 제어기
        if clock.is_swing_tick(sim_cnt):
            # 접촉 상태는 루프 최상단에서 이미 갱신됐다. 스윙 궤적은 여전히
            # 1kHz 로만 전진시킨다 - 여기서 바뀐 것은 '언제 보는가'가 아니라
            # '누가 그 사실의 주인인가'다.
            p_feet_wf = swing.update(
                is_stance=is_stance_now,
                p_feet_W=p_feet_wf,
                p_feet_target_W=p_feet_des_wf,
                dt=clock.swing_dt,
            )
            current_s = swing.phase
        # 🟩 

        r_feet_wf = p_feet_wf - robot.P

        # 지령을 하나의 값으로 묶어 넘긴다.
        #
        # 결함 10 을 고친 뒤에야 아래 is_stance 인자가 의미를 갖는다. 이제
        # stance_mask_at_solve 와 is_stance_now 는 항상 같다 - 달라지는 순간
        # 위에서 MPC 를 다시 풀기 때문이다. 따라서 ControlCommand 의
        # '스윙 다리 힘 = 0' 검사는 형식 검사가 아니라 실제 물리 조건의
        # 강제가 되고, 매 물리 스텝(27000회) 실행된다. 별도의 assert 를 두지
        # 않는 이유가 이것이다 - 검사 지점은 하나여야 한다.
        command = ControlCommand(
            forces_W=F_G,
            r_feet_W=r_feet_wf,
            is_stance=is_stance_now,
            # SRBD 는 이 둘을 무시한다(이상적 플랜트라 저수준 추종이 완벽하다는
            # 가정). 그래도 싣는 이유는 지령의 '모양'이 Plant 종류와 무관해야
            # 하기 때문이다 - 같은 Controller 가 MuJoCo 에도 그대로 꽂힌다.
            v_feet_W=swing.velocity_W,
            a_feet_W=swing.acceleration_W,
        )
        # 물리 엔진 스텝 업데이트 (Single Rigid Body Dynamics)
        robot.step(command)
        p_feet_local = robot.feet_local_B
        # ----------------------------------------------------
        # 데이터 로깅
        # ----------------------------------------------------
        if clock.is_log_tick(sim_cnt):
            log.record(
                R=robot.RW_B,
                r_feet_wf=r_feet_wf,
                F_G=F_G,
                x=X_current.flatten(),
                xref=x_ref_traj[0:13].flatten(),
                Sa=Sa_current,
                q=robot.Q,
                p_feet_des_wf=p_feet_des_wf,
                r_feet_des_wf=r_feet_des_wf,
                p_feet_local=p_feet_local,
                p_feet_wf=p_feet_wf,
                s=current_s,
            )

    h = log.arrays()
    n_log = len(log)

    return {
        "completed":     not diverged,
        "diverged_at_s": None if not diverged else sim_cnt * clock.dt,
        "t":             np.arange(n_log) * (clock.log_every * clock.dt),
        "com_pos":       h["x"][:, 3:6],
        "com_vel":       h["x"][:, 9:12],
        "rpy":           h["x"][:, 0:3],
        "omega":         h["x"][:, 6:9],
        "grf":           h["F_G"],
        "feet_W":        h["p_feet_wf"],
        "contact":       h["Sa"],
        "q":             h["q"],
        "max_s":         np.asarray(swing.max_phase_seen),
        # 진단: 접촉 전이 때문에 주기 밖에서 추가로 푼 MPC 횟수.
        # 0 이면 이벤트 구동이 동작하지 않는 것이고, 과도하게 크면 접촉
        # 판정이 떨리고 있다는 뜻이다(채터링).
        "n_event_solves": np.asarray(n_event_solves),

        # ── 시각화/진단용 신호 ──────────────────────────────────────────
        # 위쪽은 '물리적으로 의미가 고정된' 신호라 회귀 판정(TOLERANCES)의
        # 대상이다. 아래쪽은 그림을 그리는 데 필요한 내부 상태이고, 판정
        # 대상이 아니다. 한 딕셔너리에 담되 주석으로 층을 나눠 둔다 —
        # 여기서 층이 흐려지면 '그림이 바뀌었으니 회귀'라는 잘못된 경보가 난다.
        "R_W_B":          h["R"],              # (T,3,3) 월드←바디 회전행렬
        "state_13":       h["x"],              # (T,13) RobotState.to_mpc_vector() 레이아웃
        "state_ref_13":   h["xref"],           # (T,13) horizon 첫 스텝의 참조
        "feet_rel_W":     h["r_feet_wf"],      # (T,3,4) CoM→발 (월드 정렬)
        "feet_des_W":     h["p_feet_des_wf"],  # (T,3,4) Raibert 목표 착지점 (절대)
        "feet_des_rel_W": h["r_feet_des_wf"],  # (T,3,4) 같은 목표점의 CoM 상대
        "feet_local_B":   h["p_feet_local"],   # (T,3,4) 바디 기준 발 위치 (IK 입력)
        "swing_phase":    h["s"],              # (T,4) 스윙 진행률 0..1
    }
