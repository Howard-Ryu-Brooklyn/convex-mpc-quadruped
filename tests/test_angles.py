"""AngleUnwrapper 테스트.

이 모듈은 MuJoCo 를 import 하지 않는다. 그래서 MuJoCo 가 없는 환경에서도 돌고,
실패했을 때 원인이 '물리 엔진'인지 '각도 처리'인지 헷갈릴 일이 없다.
경계에서 생기는 문제는 경계 코드만 따로 떼어 시험할 수 있어야 한다.
"""

import numpy as np
import numpy.testing as npt
import pytest

from quadruped_mpc.core.angles import TWO_PI, AngleUnwrapper, wrap_to_pi


# ── wrap_to_pi ────────────────────────────────────────────────────────
@pytest.mark.parametrize("angle", [0.0, 0.5, np.pi - 1e-9, -np.pi, 3.5, -3.5,
                                   10.0, -10.0, 100.0])
def test_wrap_to_pi_lands_in_range(angle):
    w = float(wrap_to_pi(angle))
    assert -np.pi <= w < np.pi


@pytest.mark.parametrize("angle", [0.0, 0.5, -0.5, 3.0, -3.0])
def test_wrap_to_pi_preserves_angle_modulo_two_pi(angle):
    """접어도 같은 방향을 가리켜야 한다 — 회전행렬로 비교한다."""
    w = float(wrap_to_pi(angle))
    npt.assert_allclose([np.cos(w), np.sin(w)],
                        [np.cos(angle), np.sin(angle)], atol=1e-12)


def test_wrap_to_pi_is_vectorized():
    out = wrap_to_pi(np.array([0.0, 4.0, -4.0]))
    assert out.shape == (3,)


# ── 기본 동작 ─────────────────────────────────────────────────────────
def test_first_update_seeds_from_the_observation():
    u = AngleUnwrapper()
    assert u.update(1.234) == pytest.approx(1.234)
    assert u.value == pytest.approx(1.234)


def test_value_before_any_update_raises():
    with pytest.raises(ValueError):
        _ = AngleUnwrapper().value


def test_explicit_initial_angle_is_the_starting_continuous_value():
    """yaw=90deg 자세로 출발하는 시나리오를 위한 경로."""
    u = AngleUnwrapper(initial_angle_rad=np.pi / 2)
    assert u.value == pytest.approx(np.pi / 2)
    assert u.update(np.pi / 2 + 0.01) == pytest.approx(np.pi / 2 + 0.01)


def test_update_is_idempotent():
    """같은 관측을 두 번 넣어도 값이 움직이지 않아야 한다.

    observe() 를 한 스텝에 두 번 부르는 일이 실제로 생긴다
    (로깅 한 번, 제어 한 번). 그때 자세가 조용히 미끄러지면 안 된다.
    """
    u = AngleUnwrapper()
    u.update(3.0)
    first = u.update(3.05)
    assert u.update(3.05) == pytest.approx(first)
    assert u.update(3.05) == pytest.approx(first)


# ── 핵심: pi 경계를 넘는 회전 ──────────────────────────────────────────
def test_recovers_a_continuous_ramp_through_pi():
    """이 테스트 하나가 '4초 뒤 발산'의 MuJoCo 판을 막는다."""
    u = AngleUnwrapper()
    truth = np.linspace(0.0, 3 * TWO_PI, 2000)   # 3 바퀴 연속 회전
    got = np.array([u.update(float(wrap_to_pi(t))) for t in truth])
    npt.assert_allclose(got, truth, atol=1e-9)


@pytest.mark.parametrize("omega_z", [+20.0, -20.0, +180.0, -180.0])
def test_recovers_both_rotation_directions(omega_z):
    """양·음 방향 모두. 부호를 한쪽만 시험하면 부호 버그를 통째로 놓친다."""
    dt, n = 1e-3, 5000
    u = AngleUnwrapper()
    truth = np.deg2rad(omega_z) * dt * np.arange(n)
    got = np.array([u.update(float(wrap_to_pi(t))) for t in truth])
    npt.assert_allclose(got, truth, atol=1e-9)


def test_the_jump_itself_is_never_seen_by_the_consumer():
    """접힌 열에는 6.28 짜리 점프가 있고, 편 열에는 없어야 한다."""
    truth = np.linspace(0.0, TWO_PI * 2, 1000)
    wrapped = wrap_to_pi(truth)
    assert np.abs(np.diff(wrapped)).max() > 6.0        # 점프가 실재한다

    u = AngleUnwrapper()
    got = np.array([u.update(float(w)) for w in wrapped])
    assert np.abs(np.diff(got)).max() < 0.05           # 편 열에는 없다


def test_survives_a_random_walk():
    """단조 회전만이 아니라 방향이 바뀌는 실제 궤적에서도 성립해야 한다."""
    rng = np.random.default_rng(0)
    steps = rng.uniform(-0.3, 0.3, size=3000)          # |step| < pi
    truth = np.cumsum(steps)

    u = AngleUnwrapper()
    got = np.array([u.update(float(wrap_to_pi(t))) for t in truth])
    npt.assert_allclose(got, truth, atol=1e-9)


# ── 진단값 ────────────────────────────────────────────────────────────
def test_max_abs_step_records_the_largest_increment():
    u = AngleUnwrapper()
    for a in [0.0, 0.1, 0.4, 0.45]:                    # 최대 증분 0.3
        u.update(a)
    assert u.max_abs_step_rad == pytest.approx(0.3)


def test_max_abs_step_measures_the_unwrapped_increment_not_the_raw_one():
    """+pi -> -pi 점프는 '작은 한 걸음'으로 기록되어야 한다."""
    u = AngleUnwrapper()
    u.update(np.pi - 0.01)
    u.update(-np.pi + 0.01)
    assert u.max_abs_step_rad == pytest.approx(0.02, abs=1e-9)


# ── 한계를 테스트로 문서화한다 ─────────────────────────────────────────
def test_a_step_larger_than_pi_is_aliased_and_cannot_be_recovered():
    """음성 대조군(negative control).

    unwrap 은 '관측 간격 사이 변화가 pi 미만'이라는 전제 위에서만 성립한다.
    이 전제가 깨지면 조용히 틀린 답을 낸다 — 예외가 나지 않는다는 것이
    바로 위험한 지점이라, 그 사실 자체를 테스트로 못박아 둔다.

    실제로 이 전제가 깨지려면 1kHz 관측에서 180000 deg/s 가 필요하다.
    깨졌는지 여부는 max_abs_step_rad 가 pi 에 근접하는지로 감시한다.
    """
    u = AngleUnwrapper()
    u.update(0.0)
    got = u.update(float(wrap_to_pi(4.0)))   # 진짜 변화량은 +4.0 rad (> pi)

    assert got == pytest.approx(4.0 - TWO_PI)          # 최단 경로로 오해한다
    assert got != pytest.approx(4.0)
    assert u.max_abs_step_rad > 2.0                    # 감시값은 경고를 남긴다


def test_reset_clears_both_state_and_diagnostics():
    u = AngleUnwrapper()
    u.update(0.0)
    u.update(0.5)
    u.reset()
    assert u.max_abs_step_rad == 0.0
    assert u.update(9.9) == pytest.approx(9.9)
