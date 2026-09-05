"""시계열 기록.

파일명이 logging.py 가 아닌 이유: 표준 라이브러리의 logging 모듈을 가린다.
sim_main 이 sys.path 에 직접 얹혀 있어 robot_types.py 와 같은 문제가 생긴다.

분리 전에는 12 개의 리스트를 함수 앞에서 선언하고, 루프 안에서 12 줄로
append 하고, 마지막에 12 줄로 조립했다. 무엇을 기록하는지가 세 곳에 흩어져
있어 하나를 추가하려면 세 군데를 고쳐야 했고, 실제로 history_p 는 선언만
되고 아무것도 담기지 않은 채 남아 있었다.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class HistoryLogger:
    """이름 붙은 시계열을 모은다.

    record() 는 값을 항상 복사한다. 호출부가 넘긴 배열이 이후 스텝에서
    제자리 수정되어도 기록이 오염되지 않는다 — 실제로 p_feet_wf 나
    robot.RW_B 처럼 매 스텝 갱신되는 배열을 그대로 넘기고 있다.
    """

    def __init__(self) -> None:
        self._series: dict[str, list] = {}

    def record(self, **values) -> None:
        for name, value in values.items():
            self._series.setdefault(name, []).append(np.array(value, copy=True))

    def __len__(self) -> int:
        """기록된 샘플 수."""
        return len(next(iter(self._series.values()))) if self._series else 0

    def arrays(self) -> dict[str, NDArray]:
        """각 시계열을 첫 축이 시간인 배열로 쌓아 반환한다."""
        return {name: np.asarray(values) for name, values in self._series.items()}
