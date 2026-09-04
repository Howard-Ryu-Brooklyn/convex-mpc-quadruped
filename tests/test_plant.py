"""Plant 인터페이스와 SRBD 시뮬레이터 검증.

여기서 처음으로 적분기 자체를 검증한다. 지금까지 골든 테스트는 '결과가
변하지 않았다'만 말했지, '물리가 맞다'는 말한 적이 없다.
"""
import numpy as np
import numpy.testing as npt
import pytest

import config as cfg
from config import get_13d_state
from dynamics import SRBDynamics
from kinematics import get_q, get_r_feet_bf
from plant_base import PlantBase
from robot_types import RobotState

HEIGHT = cfg.leg_length_straight / 2      # 0.34 m


def make_plant(dt=1.0 / 9000, height=HEIGHT):
    """scenario_runner 의 초기화와 같은 조건으로 Plant 하나를 만든다."""
    p0 = np.array([[0.0], [0.0], [height]])
    zeros31 = np.zeros((3, 1))
    p_foot_bf = np.tile(np.array([[0.0], [0.0], [-height]]), (1, 4))
    r_feet_wf = get_r_feet_bf(p_foot_bf)          # ANG0 = 0 이므로 R = I

    return SRBDynamics(
        dt, cfg.m, [cfg.Ixx, cfg.Iyy, cfg.Izz], cfg.gz,
        p0, zeros31, zeros31, zeros31,
        [cfg.link_upper, cfg.link_lower, cfg.link_hip],
        get_q(height), np.zeros((3, 4)),
        cfg.hip_location_bf, r_feet_wf,
        [cfg.Ilink_hip, cfg.Ilink_upper, cfg.Ilink_lower],
    ), r_feet_wf


# ── 인터페이스 ───────────────────────────────────────────────────────────
def test_plant_base_cannot_be_instantiated():
    with pytest.raises(TypeError):
        PlantBase()


def test_srbdynamics_implements_plant_base():
    plant, _ = make_plant()
    assert isinstance(plant, PlantBase)


def test_dt_is_exposed_through_the_interface():
    plant, _ = make_plant(dt=1.0 / 9000)
    assert plant.dt == pytest.approx(1.0 / 9000)


# ── observe() 특성화 (1-5 의 교체를 안전하게 만든다) ──────────────────────
def test_observe_matches_legacy_properties():
    plant, r_feet = make_plant()
    forces = np.tile(np.array([[0.0], [0.0], [abs(cfg.m * cfg.gz) / 4]]), (1, 4))
    for _ in range(50):
        plant.step(forces, r_feet)

    s = plant.observe()
    assert isinstance(s, RobotState)
    npt.assert_array_equal(s.p_com_W, plant.P.flatten())
    npt.assert_array_equal(s.v_com_W, plant.V.flatten())
    npt.assert_array_equal(s.rpy_W, plant.ANG.flatten())
    npt.assert_array_equal(s.omega_W, plant.ANGVEL.flatten())


def test_observe_to_mpc_vector_matches_get_13d_state():
    """RobotState 경로가 기존 get_13d_state 경로와 비트 단위로 같은가.

    1-5 에서 메인 루프를 observe() 로 갈아끼울 때 이 동등성이 그 교체를
    기계적 치환으로 만든다.
    """
    plant, r_feet = make_plant()
    forces = np.tile(np.array([[0.0], [0.0], [abs(cfg.m * cfg.gz) / 4]]), (1, 4))
    for _ in range(50):
        plant.step(forces, r_feet)

    legacy = get_13d_state(plant.ANG, plant.P, plant.ANGVEL, plant.V).flatten()
    npt.assert_array_equal(plant.observe().to_mpc_vector(), legacy)


def test_observe_returns_a_snapshot_not_a_live_reference():
    """observe() 는 복사본을 준다. 이후 step 해도 이전 스냅샷은 변하지 않는다."""
    plant, r_feet = make_plant()
    before = plant.observe()
    p_recorded = before.p_com_W.copy()

    for _ in range(10):
        plant.step(np.zeros((3, 4)), r_feet)

    npt.assert_array_equal(before.p_com_W, p_recorded)
    assert not np.array_equal(plant.observe().p_com_W, p_recorded)


def test_legacy_property_is_a_live_reference():
    """대비 기록: 기존 프로퍼티는 내부 배열을 그대로 노출한다.

    호출자가 robot.P 를 들고 있으면 시뮬레이터가 진행할 때 그 값이 몰래
    바뀌고, robot.P[:] = x 로 내부를 오염시킬 수도 있다. observe() 로
    옮기는 이유이며, 이 테스트는 그 차이를 명시적으로 남긴다.
    TODO(Step 1-5): 레거시 프로퍼티 제거 시 이 테스트도 함께 삭제한다.
    """
    plant, r_feet = make_plant()
    live = plant.P
    z_before = float(live[2, 0])
    for _ in range(10):
        plant.step(np.zeros((3, 4)), r_feet)
    assert float(live[2, 0]) != z_before      # 들고 있던 배열이 저절로 바뀐다


# ── ★ 적분기 물리 검증 (해석해 대조) ─────────────────────────────────────
def test_free_fall_matches_discrete_integrator_exactly():
    """GRF = 0 이면 자유낙하한다. 적분 방식은 semi-implicit(symplectic) Euler:

        v_{n+1} = v_n + a*dt
        x_{n+1} = x_n + v_{n+1}*dt

    따라서 N 스텝 뒤 정확히  z = z0 + a*dt^2 * N(N+1)/2  이다.
    """
    dt, n_steps = 1.0 / 9000, 900          # 0.1 s
    plant, r_feet = make_plant(dt=dt)
    z0 = float(plant.P[2, 0])

    for _ in range(n_steps):
        plant.step(np.zeros((3, 4)), r_feet)

    expected = z0 + cfg.gz * dt**2 * n_steps * (n_steps + 1) / 2
    npt.assert_allclose(plant.observe().p_com_W[2], expected, rtol=1e-12)


def test_free_fall_approaches_analytic_solution():
    """연속 해 z(t) = z0 + a*t^2/2 에 1차 정확도로 접근하는가.

    semi-implicit Euler 의 오차는 a*dt*T/2 이며 dt 에 선형으로 줄어든다.
    이 테스트는 적분 방식이 실제로 1차임을 못 박는다 — 나중에 RK4 등으로
    바꾸면 오차가 급감하므로 이 테스트가 그 변경을 알려준다.
    """
    T = 0.1
    errors = {}
    for dt in (1.0 / 4500, 1.0 / 9000, 1.0 / 18000):
        plant, r_feet = make_plant(dt=dt)
        z0 = float(plant.P[2, 0])
        for _ in range(round(T / dt)):
            plant.step(np.zeros((3, 4)), r_feet)
        analytic = z0 + 0.5 * cfg.gz * T**2
        errors[dt] = abs(float(plant.observe().p_com_W[2]) - analytic)

    # 이론 오차: |a|*dt*T/2
    for dt, err in errors.items():
        npt.assert_allclose(err, abs(cfg.gz) * dt * T / 2, rtol=1e-6)

    # dt 를 두 배로 키우면 오차도 두 배 (1차 정확도).
    # 위 루프가 이미 이를 함의하지만, 수렴 차수를 읽기 쉬운 형태로 남긴다.
    dts = sorted(errors)                       # 오름차순: dt 가 작은 것부터
    for small, large in zip(dts, dts[1:]):     # large == small * 2
        npt.assert_allclose(errors[large] / errors[small], 2.0, rtol=1e-4)


def test_static_stance_holds_height():
    """네 다리가 mg/4 씩 지지하면 높이가 유지되는가 (힘 균형)."""
    plant, r_feet = make_plant()
    forces = np.tile(np.array([[0.0], [0.0], [abs(cfg.m * cfg.gz) / 4]]), (1, 4))
    z0 = float(plant.P[2, 0])

    for _ in range(900):
        plant.step(forces, r_feet)

    npt.assert_allclose(plant.observe().p_com_W[2], z0, atol=1e-12)
    npt.assert_allclose(plant.observe().v_com_W, np.zeros(3), atol=1e-12)
