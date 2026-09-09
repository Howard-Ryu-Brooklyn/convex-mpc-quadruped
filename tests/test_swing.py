"""스윙 궤적 생성기 검증.

분리 전에는 이지/착지 전환 로직을 골든 테스트로만 간접 확인할 수 있었다.
'발을 뗀 순간의 위치를 고정한다', '지지 다리는 절대 움직이지 않는다' 같은
불변식은 시뮬레이션 3초를 돌려야만 건드려볼 수 있었다.
"""
import numpy as np
import numpy.testing as npt
import pytest

from quadruped_mpc.core.bezier import bezier_with_derivatives
from quadruped_mpc.control.swing import SwingTrajectoryGenerator

T_SWING = 0.25
CLEARANCE = 0.05
DT = 0.001

ALL_STANCE = np.ones(4, dtype=bool)
ALL_SWING = np.zeros(4, dtype=bool)


def make_gen():
    return SwingTrajectoryGenerator(T_SWING, CLEARANCE)


def feet(x=0.0, z=0.0):
    return np.tile(np.array([[x], [0.0], [z]]), (1, 4))


# ── 핵심 불변식 ──────────────────────────────────────────────────────────
def test_stance_feet_never_move():
    """지지 다리는 땅에 박혀 있다. 위치를 절대 갱신하지 않는다."""
    gen = make_gen()
    start = feet(x=1.0)
    out = gen.update(ALL_STANCE, start, feet(x=99.0), DT)
    npt.assert_array_equal(out, start)


def test_update_does_not_mutate_its_input():
    """복사본을 돌려준다. 호출부의 배열을 제자리 수정하지 않는다."""
    gen = make_gen()
    start = feet(x=1.0)
    original = start.copy()
    gen.update(ALL_SWING, start, feet(x=2.0), DT)
    npt.assert_array_equal(start, original)


def test_liftoff_position_is_frozen_at_the_first_swing_tick():
    """이지 순간의 위치가 궤적 시작점으로 고정된다.

    이후 호출에서 p_feet_W 가 달라져도 시작점은 그대로여야 한다 —
    이미 발이 공중에 떴으므로 '현재 위치'는 궤적 자신이 만든 값이다.
    """
    gen = make_gen()
    first = gen.update(ALL_SWING, feet(x=1.0), feet(x=2.0), DT)
    second = gen.update(ALL_SWING, feet(x=999.0), feet(x=2.0), DT)

    # 999 는 무시되고, 궤적은 x=1.0 에서 x=2.0 으로 계속 간다
    assert 1.0 <= second[0, 0] <= 2.0
    assert second[0, 0] > first[0, 0]


# ── 위상 ─────────────────────────────────────────────────────────────────
def test_phase_advances_linearly_and_clips_at_one():
    gen = make_gen()
    for step in range(1, 401):                    # 0.4s > T_swing=0.25s
        gen.update(ALL_SWING, feet(), feet(x=1.0), DT)
        expected = min(step * DT / T_SWING, 1.0)
        npt.assert_allclose(gen.phase, expected, atol=1e-12)


def test_touchdown_resets_the_timer():
    gen = make_gen()
    for _ in range(100):
        gen.update(ALL_SWING, feet(), feet(x=1.0), DT)
    assert gen.phase[0] > 0.3

    gen.update(ALL_STANCE, feet(), feet(x=1.0), DT)
    npt.assert_array_equal(gen.phase, np.zeros(4))

    gen.update(ALL_SWING, feet(), feet(x=1.0), DT)   # 다시 이지
    npt.assert_allclose(gen.phase[0], DT / T_SWING)


def test_max_phase_seen_records_overrun_before_clipping():
    """T_swing 보다 오래 스윙하면 위상이 1 을 넘는다.

    클립 덕분에 궤적은 안전하지만(결함 6), 넘쳤다는 사실 자체가 스윙 시간과
    위상 기반 접촉 구간이 어긋났다는 신호이므로 기록해 둔다.
    """
    gen = make_gen()
    for _ in range(400):                          # 0.4s
        gen.update(ALL_SWING, feet(), feet(x=1.0), DT)
    assert gen.max_phase_seen == pytest.approx(0.4 / T_SWING, rel=1e-9)
    npt.assert_array_equal(gen.phase, np.ones(4))  # 궤적은 클립됨


# ── 궤적 형상 ────────────────────────────────────────────────────────────
def test_trajectory_starts_at_liftoff_and_reaches_the_target():
    gen = make_gen()
    start, target = feet(x=0.0), feet(x=0.3)

    p = gen.update(ALL_SWING, start, target, 0.0)   # phase = 0
    npt.assert_allclose(p, start, atol=1e-12)

    gen = make_gen()
    gen.update(ALL_SWING, start, target, T_SWING)   # phase = 1
    p = gen.update(ALL_SWING, start, target, 0.0)
    npt.assert_allclose(p, target, atol=1e-12)


def test_foot_lifts_above_both_endpoints_mid_swing():
    """clearance 가 실제로 발을 들어올리는가."""
    gen = make_gen()
    start, target = feet(x=0.0, z=0.0), feet(x=0.3, z=0.0)

    heights = []
    for _ in range(int(T_SWING / DT)):
        heights.append(gen.update(ALL_SWING, start, target, DT)[2, 0])

    assert max(heights) > 0.0, "발이 전혀 올라가지 않았다"
    assert max(heights) <= CLEARANCE + 1e-12, "제어점 높이를 넘을 수 없다"


def test_legs_are_independent():
    """한 다리의 상태가 다른 다리에 영향을 주지 않는가."""
    gen = make_gen()
    mixed = np.array([True, False, True, False])   # FR/RR 접지, FL/RL 스윙
    start = feet(x=0.0)

    for _ in range(50):
        out = gen.update(mixed, start, feet(x=1.0), DT)

    npt.assert_array_equal(out[:, 0], start[:, 0])   # 지지 다리는 그대로
    npt.assert_array_equal(out[:, 2], start[:, 2])
    assert out[0, 1] > 0.0 and out[0, 3] > 0.0       # 스윙 다리는 전진
    npt.assert_array_equal(gen.phase[[0, 2]], [0.0, 0.0])
    assert gen.phase[1] > 0.0 and gen.phase[3] > 0.0


# ── 1-7d-2a: 토크 구동 Plant 를 위한 속도·가속도 ────────────────────────
def _run_swing(gen, n, dt, target=None):
    """다리 0 만 스윙시키며 (위치, 속도, 가속도) 궤적을 모은다."""
    is_stance = np.array([False, True, True, True])
    p = np.zeros((3, 4))
    tgt = np.zeros((3, 4)) if target is None else target
    P, V, A = [], [], []
    for _ in range(n):
        p = gen.update(is_stance=is_stance, p_feet_W=p, p_feet_target_W=tgt, dt=dt)
        P.append(p[:, 0].copy())
        V.append(gen.velocity_W[:, 0].copy())
        A.append(gen.acceleration_W[:, 0].copy())
    return np.array(P), np.array(V), np.array(A)


def test_velocity_matches_the_numerical_derivative_of_the_position():
    """속도가 '그 위치 궤적의' 미분인가.

    위치와 미분을 따로 계산하면 언젠가 서로 맞지 않는 지령이 나간다. 토크
    구동에서는 그것이 곧 발이 목표를 향해 가면서 목표에서 멀어지라는 지령이
    된다. 두 값이 같은 곡선에서 나온다는 것을 수치로 확인한다.
    """
    dt = 1e-4
    gen = SwingTrajectoryGenerator(swing_duration_s=0.15, clearance_height_m=0.05)
    target = np.zeros((3, 4))
    target[:, 0] = [0.3, 0.1, 0.0]
    P, V, _ = _run_swing(gen, 1200, dt, target)

    v_num = np.gradient(P, dt, axis=0)
    inner = slice(50, -50)                    # 끝단은 gradient 정확도가 떨어진다
    npt.assert_allclose(V[inner], v_num[inner], atol=2e-3)


def test_acceleration_matches_the_numerical_derivative_of_the_velocity():
    dt = 1e-4
    gen = SwingTrajectoryGenerator(swing_duration_s=0.15, clearance_height_m=0.05)
    target = np.zeros((3, 4))
    target[:, 0] = [0.3, 0.1, 0.0]
    _, V, A = _run_swing(gen, 1200, dt, target)

    a_num = np.gradient(V, dt, axis=0)
    inner = slice(50, -50)
    npt.assert_allclose(A[inner], a_num[inner], rtol=2e-2, atol=1e-1)


def test_stance_legs_have_zero_velocity_and_acceleration():
    """땅에 박힌 발은 움직이지 않는다 - 위치뿐 아니라 미분도 0이어야 한다."""
    gen = SwingTrajectoryGenerator(swing_duration_s=0.15, clearance_height_m=0.05)
    gen.update(is_stance=np.array([False, True, True, True]),
               p_feet_W=np.zeros((3, 4)), p_feet_target_W=np.zeros((3, 4)), dt=1e-3)
    npt.assert_array_equal(gen.velocity_W[:, 1:], np.zeros((3, 3)))
    npt.assert_array_equal(gen.acceleration_W[:, 1:], np.zeros((3, 3)))


def test_touchdown_clears_the_derivatives():
    """스윙 중 값이 착지 후에도 남아 있으면, 지지 다리에 속도 지령이 샌다."""
    gen = SwingTrajectoryGenerator(swing_duration_s=0.15, clearance_height_m=0.05)
    tgt = np.zeros((3, 4)); tgt[:, 0] = [0.3, 0.0, 0.0]
    p = np.zeros((3, 4))
    for _ in range(70):                        # 스윙 중간까지
        p = gen.update(is_stance=np.array([False, True, True, True]),
                       p_feet_W=p, p_feet_target_W=tgt, dt=1e-3)
    assert np.linalg.norm(gen.velocity_W[:, 0]) > 0.1

    gen.update(is_stance=np.ones(4, dtype=bool),
               p_feet_W=p, p_feet_target_W=tgt, dt=1e-3)
    npt.assert_array_equal(gen.velocity_W[:, 0], np.zeros(3))
    npt.assert_array_equal(gen.acceleration_W[:, 0], np.zeros(3))


def test_liftoff_and_touchdown_velocities_are_vertical_on_the_curve():
    """제어점이 p0, p0+lift, p3+lift, p3 이므로 곡선의 양 끝 속도는 수직이다.

    B'(0) = 3(P1-P0) = 3*lift (수직 +z),  B'(1) = 3(P3-P2) = -3*lift (수직 -z).
    즉 발이 지면에 수직으로 내려앉는다 - 착지 순간 수평 속도가 0 이어야
    미끄러지지 않는다. 이 성질은 **곡선의 성질**이므로 곡선에 직접 묻는다.
    """
    p0 = np.array([0.0, 0.0, 0.0])
    p3 = np.array([0.3, 0.1, 0.0])
    lift = np.array([0.0, 0.0, 0.05])
    ctrl = [p0, p0 + lift, p3 + lift, p3]

    _, v0, _ = bezier_with_derivatives(0.0, ctrl, duration_s=0.15)
    _, v1, _ = bezier_with_derivatives(1.0, ctrl, duration_s=0.15)
    npt.assert_allclose(v0, 3 * lift / 0.15, atol=1e-12)
    npt.assert_allclose(v1, -3 * lift / 0.15, atol=1e-12)


def test_generator_leaves_the_ground_almost_vertically():
    """생성기의 첫 표본은 s=0 이 아니라 s=dt/T 다 - update() 가 먼저 시간을
    흘린 뒤 평가하기 때문이다. 따라서 '정확히 0' 이 아니라 '수직이 지배적'을
    묻는다. 무엇을 단언할 수 있는지는 구현의 시점 규약이 정한다.
    """
    dt = 1e-4
    gen = SwingTrajectoryGenerator(swing_duration_s=0.15, clearance_height_m=0.05)
    tgt = np.zeros((3, 4)); tgt[:, 0] = [0.3, 0.1, 0.0]
    _, V, _ = _run_swing(gen, 1500, dt, tgt)

    assert V[0, 2] > 0.5                                        # 위로 뜬다
    assert np.abs(V[0, :2]).max() < 0.02 * V[0, 2]              # 수평은 무시할 수준


def test_derivatives_scale_with_swing_duration():
    """같은 궤적을 절반 시간에 그리면 속도는 2배, 가속도는 4배다."""
    dt = 1e-5
    tgt = np.zeros((3, 4)); tgt[:, 0] = [0.3, 0.0, 0.0]
    out = {}
    for T in (0.2, 0.1):
        gen = SwingTrajectoryGenerator(swing_duration_s=T, clearance_height_m=0.05)
        n = int(round(0.5 * T / dt))            # 위상 0.5 지점
        _, V, A = _run_swing(gen, n, dt, tgt)
        out[T] = (V[-1], A[-1])
    # atol 이 필요한 이유: z 성분은 위상 0.5 에서 해석적으로 0 이라 값이
    # 1e-13 수준이다. 0 근처에서 상대 오차를 요구하면 반드시 실패한다 -
    # golden 테스트에서 신호별 atol 을 넣었던 것과 같은 문제다.
    npt.assert_allclose(out[0.1][0], 2.0 * out[0.2][0], rtol=1e-3, atol=1e-9)
    npt.assert_allclose(out[0.1][1], 4.0 * out[0.2][1], rtol=1e-3, atol=1e-6)


def test_derivatives_go_to_zero_once_the_phase_is_clipped():
    """결함 12 — 궤적이 끝나면 위치가 상수이므로 미분도 0 이어야 한다.

    3차 베지에의 끝점 속도는 0 이 아니라 -3*lift/T (여기서는 -1.0 m/s) 다.
    위상만 클립하고 미분을 그대로 두면 "위치는 여기 고정, 속도는 초당 1m 로
    내려가라"는 모순된 지령이 나간다. SRBD 는 발을 운동학적으로 강제하므로
    아무 일도 없지만, 토크 구동에서는 이미 착지한 발을 계속 땅으로 밀어넣는다.

    이 결함은 스윙 시간(T_swing)과 접촉 스케줄이 어긋나 위상이 1 을 넘는
    구간에서만 나타난다. 현재 게이트에서는 max_phase_seen 이 1+7e-16 이라
    사실상 닫혀 있지만, 게이트를 바꾸는 순간 열린다.
    """
    dt = 1e-3
    T = 0.15
    gen = SwingTrajectoryGenerator(swing_duration_s=T, clearance_height_m=0.05)
    tgt = np.zeros((3, 4)); tgt[:, 0] = [0.3, 0.1, 0.0]
    n = int(round(1.5 * T / dt))                 # 위상 1.5 까지 (50% 초과)
    P, V, A = _run_swing(gen, n, dt, tgt)

    assert gen.max_phase_seen > 1.4               # 실제로 오버런했다
    npt.assert_allclose(P[-1], [0.3, 0.1, 0.0], atol=1e-12)   # 위치는 목표에 고정
    npt.assert_array_equal(V[-1], np.zeros(3))                # 속도 0
    npt.assert_array_equal(A[-1], np.zeros(3))                # 가속도 0

    # 위치가 상수인 구간에서 수치 미분도 0 - 위치와 미분이 일관된다
    clipped = P[-20:]
    npt.assert_allclose(np.diff(clipped, axis=0), 0.0, atol=1e-12)
