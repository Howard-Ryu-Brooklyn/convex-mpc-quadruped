"""시계열 기록 검증."""
import numpy as np
import numpy.testing as npt

from quadruped_mpc.core.history import HistoryLogger


def test_record_stores_a_copy_not_a_reference():
    """호출부가 넘긴 배열을 이후에 수정해도 기록이 오염되지 않는가.

    메인 루프는 p_feet_wf 나 robot.RW_B 처럼 **매 스텝 제자리 수정되는**
    배열을 그대로 넘긴다. 복사하지 않으면 모든 기록이 마지막 값으로
    덮인다.

    공유 가변 상태가 무너지는 방식은 언제나 같다 — 결함 2(r_feet_traj),
    SRBDynamics.P 의 살아있는 참조, 세션 스코프 fixture 에 이어 네 번째다.
    """
    log = HistoryLogger()
    live = np.zeros(3)

    log.record(pos=live)
    live[:] = 99.0                      # 호출부가 제자리 수정
    log.record(pos=live)

    npt.assert_array_equal(log.arrays()["pos"], [[0, 0, 0], [99, 99, 99]])


def test_arrays_stacks_time_on_the_first_axis():
    log = HistoryLogger()
    for k in range(5):
        log.record(scalar=k, vec=np.full(3, k), mat=np.full((3, 4), k))

    a = log.arrays()
    assert a["scalar"].shape == (5,)
    assert a["vec"].shape == (5, 3)
    assert a["mat"].shape == (5, 3, 4)
    npt.assert_array_equal(a["vec"][3], [3, 3, 3])


def test_len_is_the_sample_count():
    log = HistoryLogger()
    assert len(log) == 0
    for _ in range(7):
        log.record(x=1.0)
    assert len(log) == 7


def test_empty_logger_returns_empty_dict():
    assert HistoryLogger().arrays() == {}


def test_accepts_python_lists():
    """접촉 상태는 get_contact_state 가 파이썬 리스트로 돌려준다."""
    log = HistoryLogger()
    log.record(contact=[0, 1, 1, 0])
    npt.assert_array_equal(log.arrays()["contact"], [[0, 1, 1, 0]])
