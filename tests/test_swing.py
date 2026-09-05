"""스윙 궤적 생성기 검증.

분리 전에는 이지/착지 전환 로직을 골든 테스트로만 간접 확인할 수 있었다.
'발을 뗀 순간의 위치를 고정한다', '지지 다리는 절대 움직이지 않는다' 같은
불변식은 시뮬레이션 3초를 돌려야만 건드려볼 수 있었다.
"""
import numpy as np
import numpy.testing as npt
import pytest

from swing import SwingTrajectoryGenerator

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
