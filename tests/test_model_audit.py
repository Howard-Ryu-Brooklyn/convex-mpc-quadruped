"""model_audit 테스트.

MuJoCo 를 import 하지 않는다. 판정 로직(audit)과 엔진 경계
(extract_model_facts)를 나눠 둔 이유가 이것이다 — MuJoCo 가 없는 CI 에서도
"무엇을 의도한 차이로 볼 것인가" 라는 정책이 회귀 없이 지켜지는지 검사한다.
"""

import numpy as np
import pytest

import config as cfg
from model_audit import (
    Mismatch,
    ModelFacts,
    UnexpectedModelMismatch,
    audit,
    format_report,
    pyramid_effective_mu,
)


def make_facts(**overrides) -> ModelFacts:
    """cfg 와 '완전히 일치하는' 가상의 모델. 여기서 하나씩 어긋뜨려 시험한다.

    기준값이 통과하는지를 먼저 보장해야, 이후 실패가 '내가 어긋뜨린 그것'
    때문임을 알 수 있다.
    """
    base = dict(
        total_mass_kg=cfg.m,
        trunk_mass_kg=cfg.m,
        inertia_about_com_diag=(cfg.Ixx, cfg.Iyy, cfg.Izz),
        hip_positions_B=cfg.hip_location_bf.copy(),
        foot_sphere_radius_m=0.025,
        floor_friction_mu=cfg.MU_FRICTION,
        cone_type="pyramidal",
        torque_limit_nm=cfg.TAU_MAX,
        joint_damping=0.0,
        joint_frictionloss=0.0,
        n_actuators=12,
    )
    base.update(overrides)
    return ModelFacts(**base)


# ── 기준선 ────────────────────────────────────────────────────────────
def test_a_perfectly_matching_model_reports_nothing():
    assert audit(make_facts()) == []


def test_report_of_no_mismatch_is_still_a_sentence():
    assert "불일치 없음" in format_report([])


# ── A. 의도된 불확실성: 보고하되 막지 않는다 ────────────────────────────
def test_mass_mismatch_is_intentional_and_does_not_raise():
    """실제 XML 값(45.84 kg)을 그대로 쓴다."""
    out = audit(make_facts(total_mass_kg=45.84))
    assert len(out) == 1
    assert out[0].name == "질량"
    assert out[0].intentional
    assert out[0].rel_delta == pytest.approx((45.84 - 43) / 43)


def test_inertia_mismatch_is_intentional():
    out = audit(make_facts(inertia_about_com_diag=(1.1, 2.4, 2.4)))
    assert {m.name for m in out} == {"Ixx", "Iyy", "Izz"}
    assert all(m.intentional for m in out)


def test_elliptic_cone_is_intentional_and_names_the_effective_mu():
    out = audit(make_facts(cone_type="elliptic"))
    assert len(out) == 1 and out[0].intentional
    assert "0.849" in out[0].note          # sqrt(2) * 0.6


def test_joint_dissipation_is_intentional():
    out = audit(make_facts(joint_damping=5.0, joint_frictionloss=0.5))
    assert len(out) == 1 and out[0].intentional


def test_several_intentional_mismatches_coexist_without_raising():
    """실제 mit_cheetah3 모델의 상태 — 전부 의도된 것이어야 한다."""
    out = audit(make_facts(
        total_mass_kg=45.84,
        inertia_about_com_diag=(1.1, 2.4, 2.4),
        cone_type="elliptic",
        joint_damping=5.0,
        joint_frictionloss=0.5,
    ))
    assert len(out) == 6
    assert all(m.intentional for m in out)


# ── B. 미선언 불일치: 반드시 막는다 ─────────────────────────────────────
@pytest.mark.parametrize("override, expected_name", [
    (dict(n_actuators=8), "액추에이터 수"),
    (dict(floor_friction_mu=0.8), "마찰계수 mu"),
    (dict(torque_limit_nm=300.0), "토크 한계"),
    (dict(foot_sphere_radius_m=0.0), "발 반지름"),
])
def test_undeclared_mismatch_raises(override, expected_name):
    with pytest.raises(UnexpectedModelMismatch):
        audit(make_facts(**override))

    out = audit(make_facts(**override), strict=False)
    assert [m.name for m in out] == [expected_name]
    assert not out[0].intentional


def test_shifted_hip_layout_raises():
    hips = cfg.hip_location_bf.copy()
    hips[0, 0] += 0.05
    with pytest.raises(UnexpectedModelMismatch):
        audit(make_facts(hip_positions_B=hips))


def test_swapped_leg_order_raises():
    """FR 과 FL 이 뒤바뀐 모델. 값은 전부 '있는' 값이라 눈으로는 안 보인다.

    이런 종류의 사고는 로그를 아무리 봐도 안 잡힌다 — 로봇이 그냥 이상하게
    걸을 뿐이다. 그래서 배치 검사를 실행 시점에 강제한다.
    """
    hips = cfg.hip_location_bf.copy()
    hips[:, [0, 1]] = hips[:, [1, 0]]
    with pytest.raises(UnexpectedModelMismatch):
        audit(make_facts(hip_positions_B=hips))


def test_hip_tolerance_allows_sub_millimeter_noise():
    hips = cfg.hip_location_bf + 1e-5
    assert audit(make_facts(hip_positions_B=hips)) == []


def test_strict_false_never_raises_but_still_flags():
    out = audit(make_facts(n_actuators=8, total_mass_kg=45.84), strict=False)
    assert {m.intentional for m in out} == {True, False}


# ── 유효 마찰계수 ──────────────────────────────────────────────────────
def test_pyramid_effective_mu_is_sqrt_two_times_mu():
    assert pyramid_effective_mu(0.6) == pytest.approx(0.8485, abs=1e-4)


def test_pyramid_corner_force_exceeds_the_true_cone():
    """제약식 수준에서 직접 확인한다 — 이 한 줄이 미끄러짐의 원인이다."""
    mu, fz = cfg.MU_FRICTION, 100.0
    f_corner = np.array([mu * fz, mu * fz])      # MPC 제약을 만족하는 모서리 해
    assert np.abs(f_corner).max() <= mu * fz + 1e-12          # 피라미드: 통과
    assert np.linalg.norm(f_corner) > mu * fz                 # 원뿔: 위반
    assert np.linalg.norm(f_corner) == pytest.approx(pyramid_effective_mu(mu) * fz)


def test_axis_aligned_force_is_safe_in_both_models():
    """축 방향으로는 절대 미끄러지지 않는다 — 미끄러짐이 방향에 의존하는 이유."""
    mu, fz = cfg.MU_FRICTION, 100.0
    f_axis = np.array([mu * fz, 0.0])
    assert np.linalg.norm(f_axis) == pytest.approx(mu * fz)


# ── 보고서 ────────────────────────────────────────────────────────────
def test_report_marks_undeclared_mismatches_distinctly():
    text = format_report(audit(make_facts(total_mass_kg=45.84, n_actuators=8),
                               strict=False))
    assert "의도됨" in text
    assert "미선언" in text


def test_raised_error_message_contains_only_the_undeclared_ones():
    with pytest.raises(UnexpectedModelMismatch) as e:
        audit(make_facts(total_mass_kg=45.84, n_actuators=8))
    assert "액추에이터 수" in str(e.value)
    assert "질량" not in str(e.value)
