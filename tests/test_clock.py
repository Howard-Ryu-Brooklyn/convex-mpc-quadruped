"""제어 주기 검증.

분리 전에는 이 나눗셈들이 메인 함수 앞부분에 흩어져 있어 단위 테스트가
불가능했다. 어떤 주기가 어떤 루프에 속하는지도 변수 이름의 색깔 주석에만
있었다.
"""
import pytest

from quadruped_mpc.core.clock import MultiRateClock


def test_default_rates_match_the_notebook():
    c = MultiRateClock()
    assert c.sim_hz == 9000 and c.mpc_hz == 30 and c.swing_hz == 1000

    assert c.dt == pytest.approx(1 / 9000)
    assert c.mpc_every == 300      # 9000 / 30
    assert c.swing_every == 9      # 9000 / 1000
    assert c.log_every == 300      # 9000 / 30
    assert c.n_steps == 27000      # 3초


def test_tick_predicates():
    c = MultiRateClock()
    assert c.is_mpc_tick(0) and c.is_mpc_tick(300) and not c.is_mpc_tick(299)
    assert c.is_swing_tick(0) and c.is_swing_tick(9) and not c.is_swing_tick(8)
    assert c.is_log_tick(0) and c.is_log_tick(600)


def test_time_at_is_linear():
    c = MultiRateClock()
    assert c.time_at(0) == 0.0
    assert c.time_at(9000) == pytest.approx(1.0)
    assert c.time_at(300) == pytest.approx(1 / 30)


def test_swing_dt_is_the_interval_actually_experienced():
    """스윙 위상은 '실제로 흐른 시간'으로 적분해야 한다."""
    c = MultiRateClock()
    assert c.swing_dt == pytest.approx(c.dt * c.swing_every)
    assert c.swing_dt == pytest.approx(1 / 1000)


def test_mpc_dt_equals_actual_interval_when_rates_divide_evenly():
    """현재 설정에서는 모델의 dt 와 실제 호출 간격이 일치한다."""
    c = MultiRateClock()
    assert c.mpc_dt == pytest.approx(c.dt * c.mpc_every)


@pytest.mark.parametrize("kwargs", [
    dict(sim_hz=1000, mpc_hz=30),      # 1000/30 = 33.33 -> 나누어떨어지지 않음
    dict(sim_hz=9000, swing_hz=700),
    dict(sim_hz=9000, log_hz=16),   # 9000/16 = 562.5
    # 주의: 9000/24 = 375 로 딱 떨어진다. 반례를 고를 때 실제로 확인할 것.
])
def test_rejects_rates_that_do_not_divide_evenly(kwargs):
    """정수배가 아니면 조용한 모델 오차가 생긴다.

    sim_hz=1000, mpc_hz=30 이면 MPC 는 실제로 33ms 마다 호출되는데 모델은
    1/30 = 33.33ms 로 이산화한다. 1% 의 시간 오차가 Ad = I + Ac*dt 에
    그대로 들어가고, 아무도 알아채지 못한다.

    이런 종류의 오차가 결함 1(전치)과 결함 8(토크암)의 공통 성질이다 —
    모델이 플랜트와 미묘하게 다르지만 아무 데서도 에러가 나지 않는다.
    """
    with pytest.raises(ValueError, match="정수배"):
        MultiRateClock(**kwargs)
