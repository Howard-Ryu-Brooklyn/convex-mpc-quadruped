"""다리 기구학 검증.

compute_leg_ik 는 지금까지 한 번도 검증된 적이 없다. 손으로 유도한 삼각함수
식이고 부호 규약이 여러 군데 얽혀 있어, 틀렸어도 시뮬레이션은 그럴듯하게
돌아간다 (IK 결과는 시각화와 관절 로깅에만 쓰이기 때문이다).

순기구학(FK)이 없어 IK/FK 왕복 테스트는 아직 쓸 수 없다. 대신 FK 없이도
성립해야 하는 필요조건 — 코사인 법칙 — 으로 부분 검증한다.
완전한 왕복 검증은 Step 5 에서 FK 를 추가하며 다룬다.
"""
import numpy as np
import numpy.testing as npt
import pytest

import config as cfg
from kinematics import (
    clip_q,
    compute_leg_ik,
    get_interior_angle,
    get_q,
    get_r_feet_bf,
)

L_HIP, L1, L2 = cfg.link_hip, cfg.link_upper, cfg.link_lower

# 관절 제한에 걸리지 않는(= clip 이 발동하지 않는) 목표 위치들.
# clip 이 걸리면 IK 는 더 이상 그 목표를 만족하지 않으므로 아래 검증이 성립하지 않는다.
REACHABLE = [
    np.array([0.00, 0.00, -0.34]),
    np.array([0.05, 0.00, -0.36]),
    np.array([-0.05, 0.02, -0.38]),
    np.array([0.03, -0.03, -0.40]),
]


# ── 관절 제한 ────────────────────────────────────────────────────────────
@pytest.mark.parametrize("q1,q2,q3", [
    (10.0, 10.0, 10.0),
    (-10.0, -10.0, -10.0),
    (0.0, 0.0, 0.0),
])
def test_clip_q_respects_joint_limits(q1, q2, q3):
    q = clip_q(q1, q2, q3)
    assert cfg.min_q1 <= q[0] <= cfg.max_q1
    assert cfg.min_q2 <= q[1] <= cfg.max_q2
    assert cfg.min_q3 <= q[2] <= cfg.max_q3


# ── 코사인 법칙 ──────────────────────────────────────────────────────────
@pytest.mark.parametrize("a,b,c", [(0.34, 0.34, 0.4), (1.0, 1.0, 1.0), (3.0, 4.0, 5.0)])
def test_get_interior_angle_matches_law_of_cosines(a, b, c):
    """c^2 = a^2 + b^2 - 2ab cos(theta) 를 만족하는 theta 를 돌려주는가."""
    theta = get_interior_angle(a, b, c)
    npt.assert_allclose(c**2, a**2 + b**2 - 2 * a * b * np.cos(theta), rtol=1e-12)


@pytest.mark.parametrize("p_foot", REACHABLE)
def test_ik_knee_angle_satisfies_law_of_cosines(p_foot):
    """IK 가 낸 q3 가 다리 2링크 기하와 일치하는가.

    FK 없이 확인할 수 있는 필요조건이다. pitch 평면에서 고관절-발끝 거리 L 은
        L^2 = l1^2 + l2^2 + 2*l1*l2*cos(q3)
    를 만족해야 한다 (q3 는 뒤로 꺾이므로 음수, cos(q3) = cos(-q3)).
    """
    q = compute_leg_ik(p_foot, L_HIP, L1, L2)
    assert np.all(np.isfinite(q)), "도달 가능한 목표인데 NaN 이 나왔다"

    x, y, z = p_foot
    z_pitch_sq = (y**2 + z**2) - L_HIP**2          # roll 을 푼 뒤의 pitch 평면 깊이
    L_sq = x**2 + z_pitch_sq                        # 고관절 -> 발끝 거리 제곱

    npt.assert_allclose(L_sq, L1**2 + L2**2 + 2 * L1 * L2 * np.cos(q[2]), rtol=1e-9)


@pytest.mark.parametrize("p_foot", REACHABLE)
def test_ik_knee_is_backward(p_foot):
    """무릎은 뒤로 꺾인다 (치타3 규약). q3 는 항상 음수여야 한다."""
    q = compute_leg_ik(p_foot, L_HIP, L1, L2)
    assert q[2] <= 0.0, f"q3 = {q[2]:.4f} — 무릎이 앞으로 꺾였다"


def test_ik_returns_nan_when_target_is_inside_hip_radius():
    """현재 동작의 특성화(characterization).

    목표가 고관절 오프셋 반경 안쪽이면 IK 는 NaN 배열을 반환한다. 호출부는
    이를 검사하지 않으므로 실패가 조용히 전파되고, clip_q 가 그것을 은폐한다.

    TODO(Step 3): KinematicsError 예외로 바꾼다. 그때 이 테스트는
    pytest.raises 로 바뀌며, 그 변경 자체가 의도된 동작 변경의 기록이 된다.
    """
    too_close = np.array([0.0, 0.0, -L_HIP / 2])
    q = compute_leg_ik(too_close, L_HIP, L1, L2)
    assert np.all(np.isnan(q))


# ── 좌표 변환 ────────────────────────────────────────────────────────────
def test_get_r_feet_bf_matches_legacy_loop():
    """벡터화가 기존 for 루프와 비트 단위로 같은 결과를 내는가."""
    rng = np.random.default_rng(0)
    p_foot_bf = rng.normal(size=(3, 4))

    legacy = np.zeros((3, 4))
    for i in range(4):
        legacy[:, i] = p_foot_bf[:, i] + cfg.hip_location_bf[:, i]

    npt.assert_array_equal(get_r_feet_bf(p_foot_bf), legacy)


def test_get_r_feet_bf_accepts_explicit_hip_layout():
    """전역 hip_location_bf 에 대한 암묵 의존이 인자로 드러났는가."""
    p = np.zeros((3, 4))
    custom = np.ones((3, 4)) * 7.0
    npt.assert_array_equal(get_r_feet_bf(p, hip_location_bf=custom), custom)


# ── 초기 자세 ────────────────────────────────────────────────────────────
def test_get_q_returns_same_angles_for_all_legs():
    """get_q 는 네 다리가 동일한 자세라고 가정한다."""
    q = get_q(0.34)
    assert q.shape == (3, 4)
    for leg in range(1, 4):
        npt.assert_array_equal(q[:, leg], q[:, 0])
