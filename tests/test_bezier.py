"""베지에 곡선 검증.

compute_bezier_with_kinematics 의 속도/가속도는 손으로 유도한 해석 미분이고,
지금까지 한 번도 검증된 적이 없다. MuJoCo 쪽 토크 제어기가 이 가속도를
피드포워드로 쓰고 있으므로(compute_swing_leg_feedforward_torque_clean),
틀렸다면 스윙 다리 토크가 통째로 어긋난다.

여기서는 해석 미분을 수치 미분과 대조한다. 정답을 직접 알 필요 없이
'해석해와 수치해가 일치해야 한다'만으로 검증되는 유형이다.
"""
import numpy as np
import numpy.testing as npt
import pytest

from quadruped_mpc.core.bezier import bezier, bezier_with_derivatives

# 실제 스윙 궤적과 같은 모양의 3차 베지에
P0 = np.array([0.30, -0.13, 0.00])
P1 = P0 + np.array([0.00, 0.00, 0.05])
P3 = np.array([0.42, -0.13, 0.00])
P2 = P3 + np.array([0.00, 0.00, 0.05])
SWING_PTS = [P0, P1, P2, P3]
T = 0.25

S_SAMPLES = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]


# ── 두 구현의 위치가 같은가 (결함 6 재발 방지) ────────────────────────────
@pytest.mark.parametrize("s", S_SAMPLES)
def test_position_agrees_between_implementations(s):
    """미분까지 내는 버전의 위치 성분이 위치 전용 버전과 같은가.

    분리 전에는 같은 곡선을 두 파일이 각자 구현했고 안전성이 달랐다.
    지금은 bezier_with_derivatives 가 bezier 에 위임하므로 구조적으로
    같지만, 누가 최적화한다고 다시 갈라놓는 것을 이 테스트가 막는다.
    """
    p_only = bezier(s, SWING_PTS)
    p_with, _, _ = bezier_with_derivatives(s, SWING_PTS, T)
    npt.assert_array_equal(p_only, p_with)


# ── 베지에의 기본 성질 ───────────────────────────────────────────────────
def test_endpoints_interpolate_control_points():
    """B(0) = P0, B(1) = Pn. 베지에는 양 끝점만 지난다."""
    npt.assert_allclose(bezier(0.0, SWING_PTS), P0, atol=1e-15)
    npt.assert_allclose(bezier(1.0, SWING_PTS), P3, atol=1e-15)


@pytest.mark.parametrize("s", S_SAMPLES)
def test_stays_inside_control_point_bounding_box(s):
    """볼록껍질 성질. 곡선은 제어점들의 바운딩 박스를 벗어나지 않는다."""
    p = bezier(s, SWING_PTS)
    pts = np.array(SWING_PTS)
    assert np.all(p >= pts.min(axis=0) - 1e-12)
    assert np.all(p <= pts.max(axis=0) + 1e-12)


@pytest.mark.parametrize("s_out", [-0.5, -1e-9, 1.0 + 1e-9, 1.5, 3.0])
def test_phase_is_clipped(s_out):
    """결함 6 회귀 감시.

    s > 1 에서 3차 베지에는 p3 + 3(s-1)(p3-p2) 로 외삽한다. 스윙 궤적에서
    p3-p2 = -clearance*z_hat 이므로 발이 지면 아래로 파고든다.
    클립이 함수 내부에 있어야 호출부가 잊어도 안전하다.
    """
    expected = bezier(np.clip(s_out, 0.0, 1.0), SWING_PTS)
    npt.assert_array_equal(bezier(s_out, SWING_PTS), expected)


# ── ★ 해석 미분 검증 (수치 미분과 대조) ──────────────────────────────────
@pytest.mark.parametrize("s", [0.1, 0.25, 0.5, 0.75, 0.9])
def test_analytic_velocity_matches_numerical(s):
    """v = dp/dt 가 중심차분과 일치하는가."""
    h = 1e-6
    _, v, _ = bezier_with_derivatives(s, SWING_PTS, T)

    p_fwd = bezier(s + h, SWING_PTS)
    p_bwd = bezier(s - h, SWING_PTS)
    v_num = (p_fwd - p_bwd) / (2 * h) / T      # ds -> dt 환산

    npt.assert_allclose(v, v_num, rtol=1e-6, atol=1e-8,
                        err_msg=f"s={s}: 해석 속도가 수치 미분과 다르다")


@pytest.mark.parametrize("s", [0.2, 0.35, 0.5, 0.65, 0.8])
def test_analytic_acceleration_matches_numerical(s):
    """a = d2p/dt2 가 2차 중심차분과 일치하는가.

    MuJoCo 스윙 토크 피드포워드가 이 값을 그대로 쓴다.
    """
    h = 1e-4
    _, _, a = bezier_with_derivatives(s, SWING_PTS, T)

    p_f = bezier(s + h, SWING_PTS)
    p_0 = bezier(s, SWING_PTS)
    p_b = bezier(s - h, SWING_PTS)
    a_num = (p_f - 2 * p_0 + p_b) / (h ** 2) / (T ** 2)

    npt.assert_allclose(a, a_num, rtol=1e-5, atol=1e-6,
                        err_msg=f"s={s}: 해석 가속도가 수치 미분과 다르다")


# ── 시간 스케일링 ────────────────────────────────────────────────────────
def test_time_scaling_of_derivatives():
    """duration 을 절반으로 줄이면 속도는 2배, 가속도는 4배가 된다."""
    _, v1, a1 = bezier_with_derivatives(0.4, SWING_PTS, T)
    _, v2, a2 = bezier_with_derivatives(0.4, SWING_PTS, T / 2)
    npt.assert_allclose(v2, 2 * v1, rtol=1e-12)
    npt.assert_allclose(a2, 4 * a1, rtol=1e-12)


def test_linear_segment_has_constant_velocity_and_zero_acceleration():
    """제어점이 2개면 직선이다. 속도 일정, 가속도 0."""
    a_pt, b_pt = np.zeros(3), np.array([1.0, 2.0, 3.0])
    for s in S_SAMPLES:
        p, v, acc = bezier_with_derivatives(s, [a_pt, b_pt], 1.0)
        npt.assert_allclose(p, a_pt + s * (b_pt - a_pt), atol=1e-15)
        npt.assert_allclose(v, b_pt - a_pt, atol=1e-15)
        npt.assert_allclose(acc, np.zeros(3), atol=1e-15)
