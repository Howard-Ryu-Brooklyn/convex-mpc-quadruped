"""계획 계층 — 제어기에게 '무엇을 할지'를 넘긴다.

분리 전에는 아래 네 가지가 메인 루프의 MPC 브랜치 안에 뒤섞여 있었다.
    1) 참조 상태 궤적 생성
    2) Raibert heuristic 발판 계획
    3) 접촉 스케줄 예측
    4) horizon 조립

이들을 함수로 분리하면 Plant 없이 단위 테스트가 가능해지고, 무엇보다
horizon 조립 결과가 '반환값'이 되어 결함 2 가 구조적으로 불가능해진다.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

import config as cfg
from gait_planning import get_contact_state
from robot_types import HorizonPlan


def build_reference_trajectory(
    x_current: NDArray[np.float64],
    v_des_W: NDArray[np.float64],
    omega_z_des: float,
    target_height_m: float,
    horizon: int,
    mpc_dt: float,
):
    """MPC 참조 궤적 (13*k, 1) 과 스텝별 예측 yaw (k,) 를 만든다.

    Args:
        x_current: (13,1) 현재 상태 [Theta, p, omega, v, 1]
        v_des_W: (3,1) 목표 선속도 [m/s], world frame
        omega_z_des: 목표 yaw 각속도 [rad/s]
        target_height_m: 목표 CoM 높이 [m]

    설계 메모 — 이것은 '속도 추종' 모드다
        위치와 yaw 의 기준점을 매 틱 현재값으로 다시 잡는다. 따라서 절대
        위치/방위 피드백이 없고 드리프트가 보정되지 않는다 (S1 직진 3초에
        yaw 가 1.14도 밀린다). 보행 제어기로서는 합리적인 선택이지만
        코드 어디에도 명시되어 있지 않았다. 실기에서는 상위 계층(주행 계획,
        SLAM)이 절대 위치를 잡아줘야 한다.
    """
    x_ref = np.zeros((13 * horizon, 1))
    yaw_ref = np.zeros(horizon)

    current_yaw = x_current[2, 0]
    current_x = x_current[3, 0]
    current_y = x_current[4, 0]

    for i in range(horizon):
        t_i = i * mpc_dt
        x_ref[i * 13 + 0, 0] = 0.0                              # roll
        x_ref[i * 13 + 1, 0] = 0.0                              # pitch
        x_ref[i * 13 + 2, 0] = current_yaw + omega_z_des * t_i  # yaw
        x_ref[i * 13 + 3, 0] = current_x + v_des_W[0, 0] * t_i
        x_ref[i * 13 + 4, 0] = current_y + v_des_W[1, 0] * t_i
        x_ref[i * 13 + 5, 0] = target_height_m
        x_ref[i * 13 + 6 : i * 13 + 8, 0] = 0.0                 # roll/pitch rate
        x_ref[i * 13 + 8, 0] = omega_z_des                      # yaw rate
        x_ref[i * 13 + 9 : i * 13 + 12, 0] = v_des_W.flatten()
        # index 12 (중력 상수항) 는 0 으로 둔다. L 가중치가 0 이라 무관하다.
        yaw_ref[i] = current_yaw + omega_z_des * t_i

    return x_ref, yaw_ref


def raibert_footholds(
    p_com_W: NDArray[np.float64],
    R_W_B: NDArray[np.float64],
    v_com_W: NDArray[np.float64],
    T_stance: float,
    hip_location_bf: NDArray[np.float64] | None = None,
):
    """Raibert heuristic 으로 다음 착지점을 계산한다.

        p_foot = p_hip + (T_stance / 2) * v_com

    직관: 지지 구간의 절반만큼 앞에 발을 두면, 몸이 그 위를 지나갈 때
    발이 몸통 아래 중앙에 오게 되어 속도가 유지된다.

    Returns:
        (p_feet_des_W, r_feet_des_W) — 절대 착지점과 CoM 기준 상대 벡터.
        두 값을 모두 돌려주는 이유는 스윙 궤적은 절대 위치를, MPC 토크암은
        상대 벡터를 필요로 하기 때문이다. 하나에서 다른 하나를 매번 유도하면
        어느 쪽을 넘겼는지 헷갈린다 (결함 5 가 그 사고였다).
    """
    if hip_location_bf is None:
        hip_location_bf = cfg.hip_location_bf

    hip_W = p_com_W + R_W_B @ hip_location_bf          # (3,1) + (3,3)@(3,4)
    p_feet_des_W = hip_W + (T_stance / 2.0) * v_com_W
    p_feet_des_W[2, :] = 0.0                            # 지면 높이로 투영
    return p_feet_des_W, p_feet_des_W - p_com_W


def build_contact_schedule(
    current_time: float,
    gait_period: float,
    gait_duty: float,
    gait_phase_offset,
    horizon: int,
    mpc_dt: float,
) -> NDArray[np.bool_]:
    """horizon 각 스텝의 접촉 상태 (4, k) 를 예측한다. True = 접지."""
    is_stance = np.zeros((4, horizon), dtype=bool)
    for k in range(horizon):
        future_time = current_time + (k * mpc_dt)
        future_phase = (future_time % gait_period) / gait_period
        sa = get_contact_state(future_phase, gait_duty, gait_phase_offset)
        is_stance[:, k] = np.asarray(sa) == 0            # GROUND(0) -> True
    return is_stance


def build_horizon_plan(
    *,
    x_ref: NDArray[np.float64],
    yaw_ref: NDArray[np.float64],
    is_stance_schedule: NDArray[np.bool_],
    r_feet_now_W: NDArray[np.float64],
    r_feet_des_W: NDArray[np.float64],
    is_stance_now: NDArray[np.bool_],
) -> HorizonPlan:
    """MPC 한 번의 풀이에 필요한 모든 것을 하나의 불변 객체로 조립한다.

    토크암 선택
        MPC 의 B 행렬에 들어가는 r_i 는 '힘이 실제로 작용하는 지점'이어야 한다.
          - 지지 중인 다리: 접지 순간에 고정된 실제 위치 (r_feet_now_W)
          - 스윙 중인 다리: 아직 닿지 않았으므로 Raibert 목표점 (r_feet_des_W)

    TODO(Step 1-4b): 지금은 '현재 시점' 값을 horizon 전 구간에 복사하는 근사다.
        스텝 k 에서 접지 상태가 바뀌는 다리는 그 시점의 예측 위치를 써야 하고,
        r 은 각 스텝의 예측 CoM 기준이어야 한다. is_stance_schedule 이
        이미 스텝별 정보를 갖고 있으므로 그 자리는 마련되어 있다.
    """
    horizon = is_stance_schedule.shape[1]
    r_feet_mpc = np.where(is_stance_now[None, :], r_feet_now_W, r_feet_des_W)

    return HorizonPlan(
        x_ref=x_ref,
        yaw_ref=yaw_ref,
        r_feet_W=[r_feet_mpc.copy() for _ in range(horizon)],
        is_stance=is_stance_schedule,
    )
