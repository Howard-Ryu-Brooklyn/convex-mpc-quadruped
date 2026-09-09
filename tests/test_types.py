"""types.py 의 characterization test.

1-1 은 순수 추가라 아무도 types.py 를 import 하지 않는다. 따라서 골든 테스트는
이 파일에 대해 아무것도 검증하지 못한다. 대신 '새 타입이 기존 함수와 같은 값을
낸다'를 먼저 못 박아두면, 나중에 기존 코드를 이 타입으로 갈아끼우는 작업이
기계적인 치환이 된다. 교체 중 깨지면 이 테스트가 어느 쪽이 틀렸는지 알려준다.
"""
import numpy as np
import numpy.testing as npt
import pytest

from quadruped_mpc import config as cfg
from quadruped_mpc.core.robot_types import ControlOutput, FootState, Leg, RobotState, SolverStatus


# ── RobotState ──────────────────────────────────────────────────────────
def test_to_mpc_vector_matches_legacy_layout():
    """13차원 변환이 기존 get_13d_state 의 배치와 비트 단위로 같은가.

    get_13d_state 는 1-5b 에서 제거됐다. 그 수식을 여기 인라인해 두어
    배치 보장이 코드에서 사라지지 않게 한다.

    이 동등성이 성립해야 나중의 교체가 안전하다.
    """
    rng = np.random.default_rng(0)
    for _ in range(200):
        rpy, p, omega, v = (rng.normal(size=3) for _ in range(4))

        legacy = np.vstack(
            (rpy.reshape(3, 1), p.reshape(3, 1),
             omega.reshape(3, 1), v.reshape(3, 1), [[1.0]])
        ).flatten()
        new = RobotState(
            p_com_W=p, v_com_W=v, rpy_W=rpy, omega_W=omega
        ).to_mpc_vector()

        npt.assert_array_equal(new, legacy)


def test_mpc_vector_layout():
    """13차원 벡터의 배치가 [Theta(3), p(3), omega(3), v(3), 1.0] 인가."""
    s = RobotState(
        p_com_W=np.array([1.0, 2.0, 3.0]),
        v_com_W=np.array([4.0, 5.0, 6.0]),
        rpy_W=np.array([0.1, 0.2, 0.3]),
        omega_W=np.array([7.0, 8.0, 9.0]),
    )
    x = s.to_mpc_vector()
    assert x.shape == (13,)
    npt.assert_array_equal(x[0:3], [0.1, 0.2, 0.3])
    npt.assert_array_equal(x[3:6], [1.0, 2.0, 3.0])
    npt.assert_array_equal(x[6:9], [7.0, 8.0, 9.0])
    npt.assert_array_equal(x[9:12], [4.0, 5.0, 6.0])
    assert x[12] == 1.0


def test_rpy_accessors():
    s = RobotState(
        p_com_W=np.zeros(3),
        v_com_W=np.zeros(3),
        rpy_W=np.array([0.1, 0.2, 0.3]),
        omega_W=np.zeros(3),
    )
    assert (s.roll, s.pitch, s.yaw) == (0.1, 0.2, 0.3)


@pytest.mark.parametrize("field", ["p_com_W", "v_com_W", "rpy_W", "omega_W"])
def test_robot_state_rejects_bad_shape(field):
    kwargs = {
        "p_com_W": np.zeros(3),
        "v_com_W": np.zeros(3),
        "rpy_W": np.zeros(3),
        "omega_W": np.zeros(3),
    }
    kwargs[field] = np.zeros(4)
    with pytest.raises(ValueError, match=field):
        RobotState(**kwargs)


def test_robot_state_is_immutable():
    """frozen=True.

    현재 SRBDynamics.P 는 내부 배열의 살아있는 참조를 반환해, 호출자가
    robot.P += x 로 시뮬레이터 내부 상태를 오염시킬 수 있다. 결함 2 가
    공유 가변 상태 사고였음을 생각하면 이 제약은 예방접종이다.
    """
    s = RobotState(np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3))
    with pytest.raises(Exception):
        s.p_com_W = np.ones(3)


# ── FootState ───────────────────────────────────────────────────────────
def test_foot_state_shapes():
    f = FootState(
        pos_W=np.zeros((3, 4)),
        rel_com_W=np.zeros((3, 4)),
        is_stance=np.array([True, False, False, True]),
    )
    assert f.is_stance[Leg.FR] and f.is_stance[Leg.RL]
    assert not f.is_stance[Leg.FL]


@pytest.mark.parametrize(
    "field,bad",
    [("pos_W", np.zeros((4, 3))), ("rel_com_W", np.zeros(3)), ("is_stance", np.zeros(3, dtype=bool))],
)
def test_foot_state_rejects_bad_shape(field, bad):
    kwargs = {
        "pos_W": np.zeros((3, 4)),
        "rel_com_W": np.zeros((3, 4)),
        "is_stance": np.zeros(4, dtype=bool),
    }
    kwargs[field] = bad
    with pytest.raises(ValueError, match=field):
        FootState(**kwargs)


# ── Leg / ControlOutput ─────────────────────────────────────────────────
def test_leg_order_matches_hip_layout():
    """Leg enum 의 순서가 config.hip_location_bf 의 열 순서(FR,FL,RR,RL)와 같은가.

    지금까지 주석에만 있던 규약이다. x 부호가 앞다리/뒷다리를,
    y 부호가 오른쪽/왼쪽을 가른다.
    """
    hips = cfg.hip_location_bf  # (3, 4)
    assert hips[0, Leg.FR] > 0 and hips[1, Leg.FR] < 0   # 앞, 오른쪽
    assert hips[0, Leg.FL] > 0 and hips[1, Leg.FL] > 0   # 앞, 왼쪽
    assert hips[0, Leg.RR] < 0 and hips[1, Leg.RR] < 0   # 뒤, 오른쪽
    assert hips[0, Leg.RL] < 0 and hips[1, Leg.RL] > 0   # 뒤, 왼쪽


def test_control_output_validity():
    ok = ControlOutput(np.zeros((3, 4)), SolverStatus.SOLVED, 0.001, 12)
    ng = ControlOutput(np.zeros((3, 4)), SolverStatus.MAX_ITER_REACHED, 0.033, 4000)
    assert ok.is_valid
    assert not ng.is_valid
