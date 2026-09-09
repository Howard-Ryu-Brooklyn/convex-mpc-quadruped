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

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from quadruped_mpc.control.gait_planning import get_contact_state
from quadruped_mpc.core.robot_types import HorizonPlan


def build_reference_trajectory(
    x_current: NDArray[np.float64],
    v_des_W: NDArray[np.float64],
    omega_z_des: float,
    target_height_m: float,
    horizon: int,
    mpc_dt: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
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
    hip_positions_W: NDArray[np.float64],
    p_com_W: NDArray[np.float64],
    v_com_W: NDArray[np.float64],
    T_stance: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Raibert heuristic 으로 다음 착지점을 계산한다.

        p_foot = p_hip + (T_stance / 2) * v_com

    직관: 지지 구간의 절반만큼 앞에 발을 두면, 몸이 그 위를 지나갈 때
    발이 몸통 아래 중앙에 오게 되어 속도가 유지된다.

    ★ 힙 위치를 (p_com, R_W_B, hip_location_bf) 에서 유도하지 않고 인자로
      받는 이유 — Plant 중립성
      예전 시그니처는 자세 행렬과 config 의 힙 배치로부터 힙 위치를 **계산**
      했다. 그것은 "이 로봇의 힙은 몸통에 강체로 붙어 있고 그 오프셋은 config
      에 있다"는 가정을 계획 계층에 박아 넣는 것이다.

      MuJoCoPlant 는 힙 위치를 물리 엔진이 이미 알고 있다(data.xpos). 실기
      에서는 관절각과 실제 링크 치수로부터 나온다. 그 지식의 주인은 Plant 다.
      계획 계층은 "힙이 여기 있다"만 받으면 되고, 그러면 같은 함수가 SRBD /
      MuJoCo / 실기 어디에나 그대로 꽂힌다.

      이 인자 하나를 바꾸는 것이 'Controller 가 Plant 종류를 모른다'를
      규율이 아니라 시그니처로 강제하는 지점이다.

    Args:
        hip_positions_W: (3,4) 네 고관절의 월드 좌표. Plant 가 준다.
        p_com_W: (3,1) CoM 월드 좌표.
        v_com_W: (3,1) CoM 월드 속도.
        T_stance: 지지 구간 길이 [s].

    Returns:
        (p_feet_des_W, r_feet_des_W) — 절대 착지점과 CoM 기준 상대 벡터.
        두 값을 모두 돌려주는 이유는 스윙 궤적은 절대 위치를, MPC 토크암은
        상대 벡터를 필요로 하기 때문이다. 하나에서 다른 하나를 매번 유도하면
        어느 쪽을 넘겼는지 헷갈린다 (결함 5 가 그 사고였다).
    """
    p_feet_des_W = hip_positions_W + (T_stance / 2.0) * v_com_W
    p_feet_des_W = p_feet_des_W.copy()
    p_feet_des_W[2, :] = 0.0                            # 지면 높이로 투영
    return p_feet_des_W, p_feet_des_W - p_com_W


def build_contact_schedule(
    current_time: float,
    gait_period: float,
    gait_duty: float,
    gait_phase_offset: Sequence[float] | NDArray[np.float64],
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
    p_feet_now_W: NDArray[np.float64],
    p_feet_des_W: NDArray[np.float64],
    is_stance_now: NDArray[np.bool_],
) -> HorizonPlan:
    """MPC 한 번의 풀이에 필요한 모든 것을 하나의 불변 객체로 조립한다.

    Args:
        x_ref: (13k, 1) 참조 상태 궤적
        yaw_ref: (k,) 스텝별 예측 yaw
        is_stance_schedule: (4, k) 스텝별 접촉 예측
        p_feet_now_W: (3,4) 현재 발의 **절대** 위치 [m]
        p_feet_des_W: (3,4) Raibert 목표 착지점의 **절대** 위치 [m]
        is_stance_now: (4,) 현재 접촉 상태

    ── 토크암 r_i 를 어떻게 정하는가 (결함 8) ──────────────────────────────

    MPC 의 B 행렬에 들어가는 r_i 는 스텝 k 에서 **힘이 실제로 작용하는 지점을
    그 시점의 CoM 기준으로** 본 것이어야 한다. 두 축 모두 시간에 따라 변한다.

    (1) 발 위치
        지금 접지 중이고 스텝 k 까지 계속 접지라면, 그 발은 움직이지 않는다.
        따라서 현재 실제 위치를 그대로 쓴다.
        그 외(지금 스윙 중이거나, 도중에 이지했다가 다시 착지하는 경우)는
        아직 그 자리에 없으므로 Raibert 목표 착지점을 쓴다.

    (2) CoM 위치
        참조 궤적의 스텝 k 위치를 쓴다. MPC 는 자신이 참조를 추종한다고
        가정하고 그 궤적 주변에서 선형화하므로 일관된 선택이다.

        대안으로 p_com_now + v_com_now * t 도 가능하지만, trot 에서
        v_com_now 는 보행 주기에 맞춰 심하게 진동한다. 그러면 토크암이 매
        MPC 틱마다 요동쳐 QP 에 노이즈를 주입하게 된다. v_des 는 매끄럽다.

    ── 이전 근사가 왜 문제였는가 ──────────────────────────────────────────

    직전까지는 '현재 시점' 값 하나를 horizon 전 구간에 복사했다. v=1 m/s 에서
    horizon 끝(k=9)의 CoM 은 실제로 0.3 m 앞에 있는데 그것을 무시한 것이다.
    네 발의 토크암이 일제히 뒤로 치우친 것과 같고, 존재하지 않는 피치 모멘트를
    MPC 가 보게 된다. 결함 8 의 절반만 고친 상태였다.
    """
    horizon = is_stance_schedule.shape[1]
    r_feet_W = []

    for k in range(horizon):
        # 스텝 k 까지 한 번도 발을 떼지 않은 다리만 '그 자리에 그대로' 있다.
        stays_planted = is_stance_now & is_stance_schedule[:, : k + 1].all(axis=1)

        p_feet_k = np.where(stays_planted[None, :], p_feet_now_W, p_feet_des_W)
        p_com_k = x_ref[k * 13 + 3 : k * 13 + 6, 0:1]      # (3,1)

        r_feet_W.append(p_feet_k - p_com_k)

    return HorizonPlan(
        x_ref=x_ref,
        yaw_ref=yaw_ref,
        r_feet_W=r_feet_W,
        is_stance=is_stance_schedule,
    )
