"""ControlCommand 테스트.

이 타입의 존재 이유는 '값을 담는 것'이 아니라 **물리적으로 성립할 수 없는
지령이 만들어지지 못하게 막는 것**이다. 그래서 테스트의 절반이 거부 사례다.
"""

import numpy as np
import pytest

from quadruped_mpc.core.robot_types import ControlCommand

ALL_STANCE = np.ones(4, dtype=bool)
TROT_A = np.array([True, False, False, True])   # FR, RL 접지


def make(**over):
    base = dict(forces_W=np.zeros((3, 4)),
                r_feet_W=np.zeros((3, 4)),
                is_stance=ALL_STANCE.copy())
    base.update(over)
    return ControlCommand(**base)


# ── 형상 검사 ─────────────────────────────────────────────────────────
@pytest.mark.parametrize("field, bad", [
    ("forces_W", np.zeros((4, 3))),
    ("forces_W", np.zeros(12)),
    ("r_feet_W", np.zeros((3, 3))),
    ("is_stance", np.ones(3, dtype=bool)),
    ("v_feet_W", np.zeros((3, 3))),
    ("a_feet_W", np.zeros(4)),
])
def test_rejects_wrong_shapes(field, bad):
    with pytest.raises(ValueError, match=field):
        make(**{field: bad})


def test_accepts_optional_swing_fields():
    c = make(v_feet_W=np.ones((3, 4)), a_feet_W=np.ones((3, 4)))
    assert c.v_feet_W is not None and c.a_feet_W is not None


def test_optional_fields_default_to_none_not_zero():
    """None 과 0 은 다른 사실이다 — '안 줬다' 와 '0 을 지령했다'.

    토크 구동 Plant 가 None 을 0 으로 조용히 대체하면, 스윙 피드포워드를
    빠뜨린 실수가 '가속도 0 을 의도했다' 로 위장된다.
    """
    c = make()
    assert c.v_feet_W is None
    assert c.a_feet_W is None


# ── 핵심 불변량: 공중에 뜬 다리는 힘을 낼 수 없다 (결함 7) ──────────────
def test_rejects_force_on_a_swing_leg():
    f = np.zeros((3, 4))
    f[2, 1] = 105.0                       # FL 은 스윙인데 힘이 실렸다
    with pytest.raises(ValueError, match="결함 7"):
        make(forces_W=f, is_stance=TROT_A)


def test_rejects_even_solver_residual_sized_force():
    """OSQP 잔류력 수준(1e-4 N)도 거부한다.

    '작으니까 괜찮다' 로 열어 두면 마스킹을 잊었을 때 아무 일도 일어나지
    않고, 실기에서 공중에 뜬 다리에 토크 지령이 새는 형태로 나타난다.
    허용치를 두는 순간 이 타입은 아무것도 보장하지 않게 된다.
    """
    f = np.zeros((3, 4))
    f[2, 1] = 8.339e-05
    with pytest.raises(ValueError, match="결함 7"):
        make(forces_W=f, is_stance=TROT_A)


def test_rejects_negative_residual_too():
    """잔류력은 음수로도 남는다. 절댓값으로 판정해야 한다."""
    f = np.zeros((3, 4))
    f[2, 2] = -1e-9
    with pytest.raises(ValueError, match="결함 7"):
        make(forces_W=f, is_stance=TROT_A)


def test_error_message_names_the_worst_offender_magnitude():
    """진단은 '틀렸다'가 아니라 '얼마나 틀렸다'여야 고칠 수 있다."""
    f = np.zeros((3, 4))
    f[0, 1] = 3.5
    with pytest.raises(ValueError) as e:
        make(forces_W=f, is_stance=TROT_A)
    assert "3.500e+00" in str(e.value)


def test_masked_force_passes():
    """올바른 사용법 — MPC 해에 접촉 마스크를 곱한 형태."""
    raw = np.random.default_rng(0).normal(size=(3, 4)) * 100
    c = make(forces_W=raw * TROT_A, is_stance=TROT_A)
    assert np.all(c.forces_W[:, ~TROT_A] == 0.0)


def test_all_swing_is_allowed_when_all_forces_are_zero():
    """비행 구간(flying trot/gallop)은 물리적으로 정당하다."""
    c = make(is_stance=np.zeros(4, dtype=bool))
    assert c.n_stance == 0


def test_stance_legs_may_carry_any_force():
    f = np.zeros((3, 4))
    f[2, :] = 1e6
    assert make(forces_W=f).n_stance == 4


# ── 불변성 ────────────────────────────────────────────────────────────
def test_command_is_frozen():
    c = make()
    with pytest.raises(Exception):
        c.forces_W = np.ones((3, 4))


def test_n_stance_counts_contacts():
    assert make(is_stance=TROT_A).n_stance == 2
