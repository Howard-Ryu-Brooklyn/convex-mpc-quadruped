"""계획 계층 검증.

분리 전에는 이 네 가지가 메인 루프 안에 뒤섞여 있어 Plant 없이는 아무것도
테스트할 수 없었다. 분리 자체보다 '단위 테스트가 가능해진 것'이 이득이다.
"""
import numpy as np
import numpy.testing as npt
import pytest

import config as cfg
from gait_planning import get_gait_parameters
from planning import (
    build_contact_schedule,
    build_horizon_plan,
    build_reference_trajectory,
    raibert_footholds,
)
from robot_types import HorizonPlan, Leg

HORIZON = 10
MPC_DT = 1.0 / 30
HEIGHT = cfg.leg_length_straight / 2


def _state(yaw=0.0, x=0.0, y=0.0, vx=0.0):
    s = np.zeros((13, 1))
    s[2, 0], s[3, 0], s[4, 0], s[9, 0], s[12, 0] = yaw, x, y, vx, 1.0
    s[5, 0] = HEIGHT
    return s


# ── 참조 궤적 ────────────────────────────────────────────────────────────
def test_reference_trajectory_layout_and_progression():
    x0 = _state(yaw=0.3, x=1.0, y=-2.0)
    v_des = np.array([[1.0], [0.5], [0.0]])
    omega_z = 0.4

    x_ref, yaw_ref = build_reference_trajectory(x0, v_des, omega_z, HEIGHT, HORIZON, MPC_DT)

    assert x_ref.shape == (13 * HORIZON, 1)
    assert yaw_ref.shape == (HORIZON,)

    for i in range(HORIZON):
        t = i * MPC_DT
        b = i * 13
        npt.assert_allclose(x_ref[b + 0 : b + 2, 0], [0.0, 0.0])          # roll/pitch 0
        npt.assert_allclose(x_ref[b + 2, 0], 0.3 + omega_z * t)           # yaw
        npt.assert_allclose(x_ref[b + 3, 0], 1.0 + 1.0 * t)               # x
        npt.assert_allclose(x_ref[b + 4, 0], -2.0 + 0.5 * t)              # y
        npt.assert_allclose(x_ref[b + 5, 0], HEIGHT)                      # z
        npt.assert_allclose(x_ref[b + 6 : b + 8, 0], [0.0, 0.0])          # roll/pitch rate
        npt.assert_allclose(x_ref[b + 8, 0], omega_z)                     # yaw rate
        npt.assert_allclose(x_ref[b + 9 : b + 12, 0], v_des.flatten())    # 선속도
        assert x_ref[b + 12, 0] == 0.0                                    # 중력 상수항
        npt.assert_allclose(yaw_ref[i], 0.3 + omega_z * t)


def test_reference_is_velocity_tracking_not_position_tracking():
    """위치 기준점이 매 틱 '현재값'으로 리셋된다.

    설계 선택이며 코드 어디에도 명시되어 있지 않았다. 결과로 절대 위치/방위
    드리프트가 보정되지 않는다 (S1 직진 3초에 yaw 1.14도).
    """
    v = np.zeros((3, 1))
    ref_a, _ = build_reference_trajectory(_state(x=0.0), v, 0.0, HEIGHT, HORIZON, MPC_DT)
    ref_b, _ = build_reference_trajectory(_state(x=5.0), v, 0.0, HEIGHT, HORIZON, MPC_DT)
    assert ref_a[3, 0] == 0.0 and ref_b[3, 0] == 5.0   # 오차가 아니라 기준점이 따라간다


# ── Raibert 발판 ─────────────────────────────────────────────────────────
def test_raibert_projects_footholds_to_ground():
    p_com = np.array([[1.0], [2.0], [HEIGHT]])
    p_des, r_des = raibert_footholds(p_com, np.eye(3), np.zeros((3, 1)), 0.25)
    npt.assert_allclose(p_des[2, :], 0.0)
    npt.assert_allclose(r_des[2, :], -HEIGHT)


def test_raibert_offset_is_half_stance_times_velocity():
    """v != 0 이면 발판이 (T_stance/2)*v 만큼 앞으로 나간다."""
    p_com = np.array([[1.0], [2.0], [HEIGHT]])
    T_stance, vx = 0.25, 1.0

    still, _ = raibert_footholds(p_com, np.eye(3), np.zeros((3, 1)), T_stance)
    moving, _ = raibert_footholds(p_com, np.eye(3), np.array([[vx], [0.0], [0.0]]), T_stance)

    npt.assert_allclose(moving[0, :] - still[0, :], T_stance / 2 * vx)
    npt.assert_allclose(moving[1, :] - still[1, :], 0.0)


def test_raibert_places_feet_under_hips_when_standing():
    p_com = np.array([[0.0], [0.0], [HEIGHT]])
    p_des, _ = raibert_footholds(p_com, np.eye(3), np.zeros((3, 1)), 0.25)
    npt.assert_allclose(p_des[:2, :], cfg.hip_location_bf[:2, :], atol=1e-15)


# ── 접촉 스케줄 ──────────────────────────────────────────────────────────
def test_standing_gait_keeps_all_feet_on_ground():
    period, duty, offset = get_gait_parameters("standing")
    sched = build_contact_schedule(0.0, period, duty, offset, HORIZON, MPC_DT)
    assert sched.shape == (4, HORIZON)
    assert sched.all()


def test_trot_keeps_exactly_two_diagonal_legs_in_stance():
    """trot 은 대각 쌍이 교대한다. 언제나 정확히 두 다리가 접지다."""
    period, duty, offset = get_gait_parameters("trotting")
    sched = build_contact_schedule(0.0, period, duty, offset, 30, MPC_DT)

    npt.assert_array_equal(sched.sum(axis=0), np.full(30, 2))
    for k in range(30):
        col = sched[:, k]
        assert col[Leg.FR] == col[Leg.RL], "FR 과 RL 은 같은 쌍이다"
        assert col[Leg.FL] == col[Leg.RR], "FL 과 RR 은 같은 쌍이다"


# ── ★ HorizonPlan — 결함 2 가 구조적으로 불가능해진 지점 ──────────────────
def _plan(is_stance_now=None):
    period, duty, offset = get_gait_parameters("trotting")
    x_ref, yaw_ref = build_reference_trajectory(
        _state(), np.zeros((3, 1)), 0.0, HEIGHT, HORIZON, MPC_DT)
    if is_stance_now is None:
        is_stance_now = np.array([True, False, False, True])
    return build_horizon_plan(
        x_ref=x_ref,
        yaw_ref=yaw_ref,
        is_stance_schedule=build_contact_schedule(0.0, period, duty, offset, HORIZON, MPC_DT),
        r_feet_now_W=np.full((3, 4), 1.0),
        r_feet_des_W=np.full((3, 4), 2.0),
        is_stance_now=is_stance_now,
    )


def test_repeated_calls_never_accumulate():
    """결함 2 재현 시도.

    이전에는 r_feet_traj 가 루프 밖 리스트라 호출할 때마다 horizon 개씩
    자랐고, solve() 가 읽는 앞의 k 개는 영원히 첫 호출의 값이었다.
    반환값에는 '이전 호출의 값이 남아 있다'가 존재할 수 없다.
    """
    for _ in range(5):
        plan = _plan()
        assert plan.horizon == HORIZON
        assert len(plan.r_feet_W) == HORIZON


def test_each_call_returns_independent_objects():
    a, b = _plan(), _plan()
    assert a is not b
    assert a.r_feet_W is not b.r_feet_W
    assert a.r_feet_W[0] is not b.r_feet_W[0]

    a.r_feet_W[0][:] = 99.0                       # 한쪽을 오염시켜도
    assert not np.any(b.r_feet_W[0] == 99.0)      # 다른 쪽은 무사하다


def test_plan_is_immutable():
    with pytest.raises(Exception):
        _plan().x_ref = np.zeros((130, 1))


def test_torque_arm_uses_actual_position_for_stance_and_target_for_swing():
    """결함 8 의 핵심 규칙.

    MPC 의 r_i 는 힘이 실제로 작용하는 지점이어야 한다. 지지 다리는 접지
    위치를, 스윙 다리는 아직 닿지 않았으므로 목표 착지점을 쓴다.
    """
    plan = _plan(is_stance_now=np.array([True, False, False, True]))
    r = plan.r_feet_W[0]
    npt.assert_allclose(r[:, Leg.FR], 1.0)   # stance -> 현재 실제 위치
    npt.assert_allclose(r[:, Leg.RL], 1.0)
    npt.assert_allclose(r[:, Leg.FL], 2.0)   # swing  -> Raibert 목표점
    npt.assert_allclose(r[:, Leg.RR], 2.0)


def test_contact_legacy_inverts_the_convention():
    """MPC 가 아직 받는 구 규약(AIR=1, GROUND=0) 변환이 맞는가.

    TODO(Step 1-4b): MPC 가 is_stance 를 직접 받게 되면 이 테스트도 사라진다.
    """
    plan = _plan()
    npt.assert_array_equal(plan.contact_legacy, 1.0 - plan.is_stance.astype(float))
    assert plan.contact_legacy[plan.is_stance].max() == 0.0    # 접지 -> 0
    assert plan.contact_legacy[~plan.is_stance].min() == 1.0   # 체공 -> 1


@pytest.mark.parametrize("field,bad", [
    ("x_ref", np.zeros((13, 1))),
    ("yaw_ref", np.zeros(3)),
    ("is_stance", np.zeros((4, 3), dtype=bool)),
])
def test_horizon_plan_rejects_inconsistent_shapes(field, bad):
    kwargs = dict(
        x_ref=np.zeros((13 * HORIZON, 1)),
        yaw_ref=np.zeros(HORIZON),
        r_feet_W=[np.zeros((3, 4)) for _ in range(HORIZON)],
        is_stance=np.zeros((4, HORIZON), dtype=bool),
    )
    kwargs[field] = bad
    with pytest.raises(ValueError, match=field):
        HorizonPlan(**kwargs)
