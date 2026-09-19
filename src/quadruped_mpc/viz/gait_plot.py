"""보행 접촉 패턴 그림.

control/gait_planning.py 에서 옮겨 왔다. 제어 모듈이 matplotlib 을 끌고 오면
"core <- control <- plants <- experiments <- viz" 의존 방향이 뒤집힌다.
그림은 아무도 의존하지 않는 쪽에 있어야 한다.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from numpy.typing import ArrayLike

from quadruped_mpc.control.gait_planning import GROUND

LEG_NAMES = ("FR", "FL", "RR", "RL")


def plot_gait_pattern(history_Sa: ArrayLike, dt: float = 1.0) -> Axes:
    """접촉 상태 이력을 간트 차트처럼 그린다.

    Args:
        history_Sa: (T, 4) — 각 행이 [FR, FL, RR, RL] 의 GROUND/AIR.
        dt: 한 스텝의 시간 [s]. 1.0 이면 x 축이 스텝 인덱스가 된다.

    Returns:
        그려진 Axes. 호출부가 제목·저장 등을 더 손볼 수 있게 돌려준다
        (show() 를 함수 안에서 부르면 노트북 밖에서 쓸 수 없다).
    """
    sa = np.asarray(history_Sa)
    if sa.ndim != 2 or sa.shape[1] != 4:
        raise ValueError(f"history_Sa 는 (T,4) 여야 한다. 받은 형상: {sa.shape}")

    time = np.arange(len(sa)) * dt
    _, ax = plt.subplots(figsize=(10, 4))

    for leg in range(4):
        is_stance = sa[:, leg] == GROUND
        ax.fill_between(time, leg - 0.3, leg + 0.3, where=is_stance,
                        color=f"C{leg}", alpha=0.8,
                        label=LEG_NAMES[leg] if is_stance.any() else "")

    ax.set_yticks(range(4))
    ax.set_yticklabels(LEG_NAMES)
    ax.invert_yaxis()                      # FR 이 맨 위 — 직관적인 순서
    ax.set_xlabel("Time (s)" if dt != 1.0 else "Simulation step")
    ax.set_ylabel("Leg")
    ax.set_title("Gait contact pattern (colored block = stance)")
    ax.grid(True, axis="x", linestyle="--", alpha=0.7)
    return ax
