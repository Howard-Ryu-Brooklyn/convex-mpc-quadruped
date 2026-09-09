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

from quadruped_mpc import config as cfg
from quadruped_mpc.core.kinematics import (
    clip_q,
    compute_leg_ik,
    leg_forward_kinematics,
    leg_link_positions,
    LEG_HIP_SIGN,
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


#: 아래 단독 IK 테스트들이 쓰는 기준 다리. FL(+y) 을 쓴다.
LEG = 1


@pytest.mark.parametrize("p_foot", REACHABLE)
def test_ik_knee_angle_satisfies_law_of_cosines(p_foot):
    """IK 가 낸 q3 가 다리 2링크 기하와 일치하는가.

    FK 없이 확인할 수 있는 필요조건이다. pitch 평면에서 고관절-발끝 거리 L 은
        L^2 = l1^2 + l2^2 + 2*l1*l2*cos(q3)
    를 만족해야 한다 (q3 는 뒤로 꺾이므로 음수, cos(q3) = cos(-q3)).
    """
    q = compute_leg_ik(p_foot, LEG, L_HIP, L1, L2)
    assert np.all(np.isfinite(q)), "도달 가능한 목표인데 NaN 이 나왔다"

    x, y, z = p_foot
    z_pitch_sq = (y**2 + z**2) - L_HIP**2          # roll 을 푼 뒤의 pitch 평면 깊이
    L_sq = x**2 + z_pitch_sq                        # 고관절 -> 발끝 거리 제곱

    npt.assert_allclose(L_sq, L1**2 + L2**2 + 2 * L1 * L2 * np.cos(q[2]), rtol=1e-9)


@pytest.mark.parametrize("p_foot", REACHABLE)
def test_ik_knee_is_backward(p_foot):
    """무릎은 뒤로 꺾인다 (치타3 규약). q3 는 항상 음수여야 한다."""
    q = compute_leg_ik(p_foot, LEG, L_HIP, L1, L2)
    assert q[2] <= 0.0, f"q3 = {q[2]:.4f} — 무릎이 앞으로 꺾였다"


def test_ik_returns_nan_when_target_is_inside_hip_radius():
    """현재 동작의 특성화(characterization).

    목표가 고관절 오프셋 반경 안쪽이면 IK 는 NaN 배열을 반환한다. 호출부는
    이를 검사하지 않으므로 실패가 조용히 전파되고, clip_q 가 그것을 은폐한다.

    TODO(Step 3): KinematicsError 예외로 바꾼다. 그때 이 테스트는
    pytest.raises 로 바뀌며, 그 변경 자체가 의도된 동작 변경의 기록이 된다.
    """
    too_close = np.array([0.0, 0.0, -L_HIP / 2])
    q = compute_leg_ik(too_close, LEG, L_HIP, L1, L2)
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
def test_get_q_is_mirror_symmetric_not_identical():
    """get_q 는 네 다리가 '같은' 자세가 아니라 '거울 대칭' 자세다.

    예전 테스트는 네 다리가 완전히 동일하다고 단언했고, 그것이 통과한 이유는
    구현이 q1 = 0 을 그대로 복사했기 때문이다. 그런데 abad 링크가 y 로
    8cm 뻗어 있으므로 q1 = 0 이면 발은 고관절 바로 아래가 아니다.
    테스트가 구현의 잘못된 가정을 그대로 굳히고 있었다.
    """
    q = get_q(0.34)
    npt.assert_allclose(q[0], [-q[0][1], q[0][1], -q[0][1], q[0][1]], atol=1e-12)
    assert abs(q[0][0]) > 0.1                      # q1 = 0 이 아니다
    npt.assert_allclose(q[1], np.full(4, q[1][0]), atol=1e-12)   # pitch 는 동일
    npt.assert_allclose(q[2], np.full(4, q[2][0]), atol=1e-12)


def test_get_q_actually_places_the_foot_under_the_hip():
    """이름이 약속하는 자세를 실제로 만드는가 — FK 로 확인한다."""
    q = get_q(0.34)
    for leg in range(4):
        npt.assert_allclose(leg_forward_kinematics(q[:, leg], leg),
                            [0.0, 0.0, -0.34], atol=1e-12)


# ── 결함 11: IK 와 FK 가 서로의 역함수인가 ─────────────────────────────
@pytest.mark.parametrize("leg", [0, 1, 2, 3])
def test_ik_and_fk_are_inverses(leg):
    """무작위 목표 600개에 대해 FK(IK(p)) == p.

    이 성질이 깨져 있던 것이 결함 11 이다. 두 가지가 겹쳐 있었다.
      1) compute_leg_ik 의 q1 에서 arctan2 항 부호가 반대였다.
      2) 호출부가 네 다리 모두에 같은 부호의 l_hip 을 넘겼다.
    둘 다 y = 0 인 정지 자세에서는 오차가 0 이라 드러나지 않는다. 결함 1이
    psi = 0 에서 숨어 있던 것과 같은 구조다 - 대칭점에서만 보면 못 잡는다.

    clip=False 로 보는 이유: 관절 한계는 기구학이 아니라 포화다. 섞으면
    'IK 가 틀렸는가'와 '한계에 걸렸는가'를 구별할 수 없다.
    """
    rng = np.random.default_rng(leg)
    ok = 0
    for _ in range(600):
        p = np.array([rng.uniform(-0.12, 0.12), rng.uniform(-0.12, 0.12),
                      rng.uniform(-0.45, -0.22)])
        q = compute_leg_ik(p, leg, clip=False)
        if np.isnan(q).any():
            continue
        npt.assert_allclose(leg_forward_kinematics(q, leg), p, atol=1e-12)
        ok += 1
    assert ok > 500, f"유효 목표가 너무 적다 ({ok}) — 시험 자체가 무의미해진다"


def test_left_and_right_legs_are_mirrored():
    """y 를 뒤집은 목표는 abad 각도만 부호가 뒤집혀야 한다 (등변성 시험)."""
    p = np.array([0.04, 0.05, -0.33])
    q_left = compute_leg_ik(p, 1, clip=False)                     # FL (+y)
    q_right = compute_leg_ik(p * np.array([1, -1, 1]), 0, clip=False)  # FR (-y)
    npt.assert_allclose(q_left[0], -q_right[0], atol=1e-12)
    npt.assert_allclose(q_left[1:], q_right[1:], atol=1e-12)


def test_hip_sign_is_derived_from_the_hip_layout():
    """부호를 하드코딩하지 않고 cfg 에서 파생한다 — 한 사실은 한 곳에."""
    npt.assert_array_equal(LEG_HIP_SIGN, np.sign(cfg.hip_location_bf[1, :]))
    npt.assert_array_equal(LEG_HIP_SIGN, [-1, 1, -1, 1])          # FR FL RR RL


def test_link_positions_chain_is_connected_with_correct_lengths():
    """그리기용 체인이 실제 링크 길이를 지키는가."""
    q = compute_leg_ik(np.array([0.03, -0.02, -0.31]), 0, clip=False)
    c = leg_link_positions(q, 0)
    assert c.shape == (3, 4)
    npt.assert_allclose(c[:, 0], 0.0, atol=1e-12)                 # 고관절 원점
    npt.assert_allclose(np.linalg.norm(c[:, 1] - c[:, 0]), cfg.link_hip, atol=1e-12)
    npt.assert_allclose(np.linalg.norm(c[:, 2] - c[:, 1]), cfg.link_upper, atol=1e-12)
    npt.assert_allclose(np.linalg.norm(c[:, 3] - c[:, 2]), cfg.link_lower, atol=1e-12)


def test_clip_is_what_breaks_the_round_trip_not_the_kinematics():
    """한계에 걸리는 목표에서는 왕복이 깨지고, 그 원인이 포화임을 고정한다."""
    far = np.array([0.0, 0.40, -0.20])       # q1 = 53.1도, abad 한계 ±45도 밖
    q_raw = compute_leg_ik(far, 1, clip=False)
    q_clipped = compute_leg_ik(far, 1, clip=True)

    assert np.rad2deg(q_raw[0]) > np.rad2deg(cfg.max_q1)           # 정말 한계 밖
    assert not np.allclose(q_raw, q_clipped)                       # 실제로 잘렸다
    npt.assert_allclose(leg_forward_kinematics(q_raw, 1), far, atol=1e-12)
    assert np.linalg.norm(leg_forward_kinematics(q_clipped, 1) - far) > 0.05


def test_legacy_get_q_signature_still_returns_3x4():
    assert get_q(0.34).shape == (3, 4)
