"""MuJoCo 플랜트로 시나리오를 실행한다.

이 파일에 **제어 로직은 한 줄도 없다.** 전부 scenario_runner.py 와 같은
모듈을 부른다 - planning, swing, clock, convex_mpc, robot_types.
다른 것은 Plant 한 줄뿐이다.

    scenario_runner :  robot = SRBDynamics(...)
    mujoco_runner   :  robot = MuJoCoPlant(...)

그것이 Step 1 의 목표였다. 예전에는 이 자리에 630 줄짜리 mujoco_main_ref.py
가 있었고, 결함 1~12 중 어느 것도 반영되어 있지 않았다.

왜 두 runner 를 합치지 않는가
    합칠 수도 있다. 하지만 두 실험의 **질문이 다르다**(model_audit.py 참조).
    SRB 는 알고리즘 자체를, MuJoCo 는 선형화 가정의 강인성을 묻는다.
    질문이 다르면 보고 싶은 진단값도 다르고(여기서는 관절 토크, 접촉력,
    yaw unwrap 감시값), 발산 판정 기준도 다르다. 공유해야 하는 것은
    '제어 로직'이지 '실험 진행 방식'이 아니다.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

import config as cfg
import convex_mpc as cvx_mpc
from clock import MultiRateClock
from gait_planning import get_contact_state, get_gait_parameters
from history import HistoryLogger
from mujoco_plant import MuJoCoPlant
from planning import (
    build_contact_schedule,
    build_horizon_plan,
    build_reference_trajectory,
    raibert_footholds,
)
from robot_types import ControlCommand
from swing import SwingTrajectoryGenerator

DEG2RAD = np.pi / 180

DEFAULT_XML = (Path(__file__).resolve().parent.parent
               / "mujoco_test" / "mit_cheetah3" / "scene.xml")

#: MuJoCo 모델의 홈 포지션. trunk z 는 align_to_ground 가 다시 잡는다.
HOME_QPOS = np.array([
    0.0, 0.0, 0.65,          # trunk pos
    1.0, 0.0, 0.0, 0.0,      # trunk quat (w,x,y,z)
    0.0, 0.9, -1.8,          # FR
    0.0, 0.9, -1.8,          # FL
    0.0, 0.9, -1.8,          # RR
    0.0, 0.9, -1.8,          # RL
])


def run_scenario_mujoco(
    gait_name: str = "trotting",
    v_des_x: float = 0.0,
    v_des_y: float = 0.0,
    omega_z_deg_s: float = 0.0,
    duration_s: float = 3.0,
    horizon: int = 10,
    mpc_hz: int = 30,
    red_hz: int = 1000,
    sim_hz: int = 9000,
    log_hz: int = 30,
    clearance_height: float = 0.05,
    xml_path: str | Path = DEFAULT_XML,
    swing_omega_n: float = 240.0,
    verbose: bool = True,
) -> dict[str, np.ndarray]:
    """시나리오 하나를 MuJoCo 로 끝까지 돌리고 로깅 배열을 돌려준다.

    scenario_runner.run_scenario 와 **같은 인자**를 받는다. 같은 시나리오
    정의(scenarios.SCENARIOS)를 두 Plant 에 그대로 먹일 수 있어야 두 결과를
    비교할 수 있기 때문이다.
    """
    clock = MultiRateClock(sim_hz=sim_hz, mpc_hz=mpc_hz, swing_hz=red_hz,
                           log_hz=log_hz, duration_s=duration_s)
    mpc_dt = clock.mpc_dt

    gait_period, gait_duty, gait_phase_offset = get_gait_parameters(gait_name)
    T_swing = gait_period * (1 - gait_duty)
    T_stance = gait_period * gait_duty

    plant = MuJoCoPlant(xml_path, dt=clock.dt, home_qpos=HOME_QPOS,
                        swing_omega_n=swing_omega_n, verbose=verbose)
    swing = SwingTrajectoryGenerator(T_swing, clearance_height)

    L_weights = cvx_mpc.ConvexMPC.build_state_weight(
        cfg.L_w_th, cfg.L_w_z, cfg.L_w_yr, cfg.L_w_v)
    mpc = cvx_mpc.ConvexMPC(cfg.m, [cfg.Ixx, cfg.Iyy, cfg.Izz], cfg.gz, mpc_dt,
                            horizon, L_weights, cfg.K_w_f, cfg.MU_FRICTION,
                            cfg.fmin, cfg.fmax)

    target_height_com = float(plant.observe().p_com_W[2])
    p_feet_wf = plant.feet_positions_W()
    p_feet_des_wf = p_feet_wf.copy()
    r_feet_des_wf = p_feet_wf - plant.observe().p_com_W.reshape(3, 1)

    log = HistoryLogger()
    diverged = False
    stance_mask_at_solve = None
    n_event_solves = 0
    F_G = np.zeros((3, 4))

    # ── 결함 13 진단: 스윙 지령 위치가 계단식으로 튀는가 ────────────────
    # 토크가 omega_n^2 로 커진다는 것은 '고정된 위치 오차 x 강성'이라는 뜻이고,
    # 그렇다면 어딘가에 계단 입력이 있어야 한다. 합성 상황이 아니라 실제 보행
    # 중에 재야 한다 - 지령이 한 물리 스텝(0.11ms) 사이에 얼마나 튀는가.
    prev_cmd_feet = None
    max_cmd_jump_m = 0.0            # 스윙 다리 지령 위치의 스텝간 최대 점프
    phase_at_max_jump = 0.0
    max_target_jump_m = 0.0         # Raibert 목표의 MPC 틱간 최대 점프(스윙 다리)
    prev_target = None

    for sim_cnt in range(clock.n_steps):
        current_time = clock.time_at(sim_cnt)

        # 접촉 상태는 매 물리 스텝 평가한다 (결함 10)
        phase = (current_time % gait_period) / gait_period
        Sa_current = get_contact_state(phase, gait_duty, gait_phase_offset)
        is_stance_now = np.asarray(Sa_current) == 0

        state = plant.observe()
        X_current = state.to_mpc_vector().reshape(13, 1)
        p_com = state.p_com_W.reshape(3, 1)

        # ── 발판 계획은 스윙 루프 주기로 갱신한다 (결함 13) ─────────────
        # 예전에는 MPC 브랜치 안에 있었다. 그래서 목표가 33.3ms 마다 계단식으로
        # 튀었고, 스윙 후반(위상 0.94, ∂B/∂p3 ≈ 0.99)에서 그 점프가 거의 그대로
        # 지령 위치의 점프가 됐다.
        #
        # 실측(MuJoCo, trot 0.5m/s):
        #   Raibert 목표 MPC틱간 점프   19.7 mm
        #   지령 위치 스텝간 점프       17.0 mm   <- 0.11ms 사이에!
        #                                          정상 스윙 속도로는 0.16mm
        #   스윙 토크  omega_n=120: 70 Nm / 240: 274 Nm  (∝ omega_n^2)
        # 토크가 omega_n^2 로 커진다는 것이 '고정 오차 x 강성'의 서명이었고,
        # 그 고정 오차가 이 계단이었다.
        #
        # 착지 직전에 발이 17mm 옆으로 튀는 것은 토크 초과보다 나쁘다 -
        # 발을 옆으로 던지면서 착지시키는 것이기 때문이다.
        #
        # 발판 계획이 MPC 주기를 물려받을 이유가 없다. Raibert 는 힙 위치와
        # 속도의 곱 하나이므로 1kHz 로 돌려도 비용이 없고, 계단이 33배 작아져
        # 연속적 드리프트가 된다.
        #
        # 힙 위치는 Plant 가 준다. MuJoCo 는 엔진이 이미 알고 있다(data.xpos).
        if clock.is_swing_tick(sim_cnt):
            p_feet_des_wf, r_feet_des_wf = raibert_footholds(
                plant.hip_positions_W(), p_com,
                X_current[9:12, 0:1].copy(), T_stance)

            # 진단: 공중에 있는 다리의 목표가 갱신 사이에 얼마나 움직였는가.
            # 결함 13 수정 전에는 이 간격이 MPC 주기(33.3ms)라 19.7mm 였다.
            if prev_target is not None and (~is_stance_now).any():
                moved = np.abs(p_feet_des_wf - prev_target)[:, ~is_stance_now]
                max_target_jump_m = max(max_target_jump_m, float(moved.max()))
            prev_target = p_feet_des_wf.copy()

        if not np.isfinite(X_current).all():
            diverged = True
            break
        # 자세가 무너지면 그 뒤 숫자는 의미가 없다. SRBD 와 달리 MuJoCo 는
        # 넘어져도 NaN 이 되지 않으므로 물리적 기준이 필요하다.
        if abs(state.rpy_W[0]) > np.pi / 3 or abs(state.rpy_W[1]) > np.pi / 3:
            diverged = True
            break

        # 🟦 상위 제어기 (MPC) — 주기 + 접촉 전이 이벤트 구동
        contact_changed = not np.array_equal(is_stance_now, stance_mask_at_solve)
        if contact_changed and not clock.is_mpc_tick(sim_cnt):
            n_event_solves += 1
        if clock.is_mpc_tick(sim_cnt) or contact_changed:
            v_des = np.array([[v_des_x], [v_des_y], [0.0]])
            x_ref_traj, yaw_traj = build_reference_trajectory(
                X_current, v_des, omega_z_deg_s * DEG2RAD,
                target_height_com, horizon, mpc_dt)

            is_stance_schedule = build_contact_schedule(
                current_time, gait_period, gait_duty, gait_phase_offset,
                horizon, mpc_dt)
            plan = build_horizon_plan(
                x_ref=x_ref_traj, yaw_ref=yaw_traj,
                is_stance_schedule=is_stance_schedule,
                p_feet_now_W=p_feet_wf, p_feet_des_W=p_feet_des_wf,
                is_stance_now=is_stance_now)

            fmpc, _ = mpc.solve(X_current, plan.x_ref, plan.yaw_ref,
                                plan.r_feet_W, plan.contact_legacy)
            stance_mask_at_solve = is_stance_now
            F_G = np.array(fmpc).reshape(4, 3).T * stance_mask_at_solve


        # 🟥 스윙 궤적
        if clock.is_swing_tick(sim_cnt):
            p_feet_wf = swing.update(
                is_stance=is_stance_now, p_feet_W=p_feet_wf,
                p_feet_target_W=p_feet_des_wf, dt=clock.swing_dt)

        # 지령 위치의 스텝간 점프 (스윙 다리만). 이것이 PD 에 들어가는 입력이다.
        if prev_cmd_feet is not None and (~is_stance_now).any():
            jump = np.abs(p_feet_wf - prev_cmd_feet)[:, ~is_stance_now].max()
            if jump > max_cmd_jump_m:
                max_cmd_jump_m = float(jump)
                phase_at_max_jump = float(swing.phase[~is_stance_now].max())
        prev_cmd_feet = p_feet_wf.copy()

        # 🟩 지령 -> 관절 토크 -> 엔진 (Plant 의 책임)
        plant.step(ControlCommand(
            forces_W=F_G,
            r_feet_W=p_feet_wf - p_com,
            is_stance=is_stance_now,
            v_feet_W=swing.velocity_W,
            a_feet_W=swing.acceleration_W,
        ))

        # 접지 다리의 '실제' 위치를 엔진에서 되읽는다.
        # SRBD 는 발 위치를 운동학적으로 강제하므로 지령 = 실제였지만,
        # MuJoCo 에서는 발이 미끄러지고 잠긴다. 그 차이가 곧 이 실험이
        # 재려는 것이므로, 다음 스텝의 토크암은 실측값으로 계산해야 한다.
        p_feet_measured = plant.feet_positions_W()
        p_feet_wf = np.where(is_stance_now[None, :], p_feet_measured, p_feet_wf)

        if clock.is_log_tick(sim_cnt):
            log.record(
                x=X_current.flatten(),
                xref=x_ref_traj[0:13].flatten(),
                F_G=F_G, Sa=Sa_current,
                q=plant.joint_angles,
                R=np.asarray(plant._data.xmat[plant._trunk_id]).reshape(3, 3),
                p_feet_wf=p_feet_measured,
                p_feet_des_wf=p_feet_des_wf,
                r_feet_des_wf=r_feet_des_wf,
                r_feet_wf=p_feet_measured - p_com,
            )

    h = log.arrays()
    n_log = len(log)
    return {
        "completed":      not diverged,
        "diverged_at_s":  None if not diverged else sim_cnt * clock.dt,
        "t":              np.arange(n_log) * (clock.log_every * clock.dt),
        "com_pos":        h["x"][:, 3:6],
        "com_vel":        h["x"][:, 9:12],
        "rpy":            h["x"][:, 0:3],
        "omega":          h["x"][:, 6:9],
        "grf":            h["F_G"],
        "feet_W":         h["p_feet_wf"],
        "contact":        h["Sa"],
        "q":              h["q"],
        "max_s":          np.asarray(swing.max_phase_seen),
        "n_event_solves": np.asarray(n_event_solves),
        # ── MuJoCo 전용 진단 ────────────────────────────────────────
        "max_torque_nm":        np.asarray(plant.max_abs_torque_nm),
        "max_torque_stance_nm": np.asarray(plant.max_torque_stance_nm),
        "max_torque_swing_nm":  np.asarray(plant.max_torque_swing_nm),
        "t_at_max_torque_s":    np.asarray(plant.time_at_max_torque_s),
        # 결함 13 진단
        "max_cmd_jump_m":       np.asarray(max_cmd_jump_m),
        "phase_at_max_jump":    np.asarray(phase_at_max_jump),
        "max_target_jump_m":    np.asarray(max_target_jump_m),
        "max_yaw_step":   np.asarray(plant.max_yaw_step_rad),
        # ── 시각화용 ────────────────────────────────────────────────
        "R_W_B":          h["R"],
        "state_13":       h["x"],
        "state_ref_13":   h["xref"],
        "feet_rel_W":     h["r_feet_wf"],
        "feet_des_W":     h["p_feet_des_wf"],
        "feet_des_rel_W": h["r_feet_des_wf"],
    }
