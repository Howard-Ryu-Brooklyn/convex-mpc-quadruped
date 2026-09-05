"""스윙 다리 발 궤적 생성.

상태를 없애는 리팩토링이 아니라 **주인을 정하는** 리팩토링이다.

스윙 궤적은 본질적으로 상태를 갖는다 — 발을 뗀 순간의 위치와 그 이후 흐른
시간을 기억해야 베지에 곡선을 그릴 수 있다. 문제는 그 상태가 없었다는 게
아니라, 메인 루프의 지역 변수 셋(swing_time_counter, swing_start_pos_wf,
current_s)으로 흩어져 누가 언제 갱신하는지 추적이 어려웠다는 것이다.

한 클래스가 소유하면 (a) 불변식을 한 곳에서 지킬 수 있고 (b) Plant 없이
단위 테스트가 가능해지며 (c) 이지/착지 전환 로직을 골든 테스트가 아니라
직접 검증할 수 있다.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from bezier import bezier


class SwingTrajectoryGenerator:
    """네 다리의 스윙 궤적을 생성하고 위상을 추적한다.

    Args:
        swing_duration_s: 한 번의 스윙에 걸리는 시간 [s]
        clearance_height_m: 발을 들어올리는 최대 높이 [m]
    """

    def __init__(self, swing_duration_s: float, clearance_height_m: float) -> None:
        self.swing_duration_s = swing_duration_s
        self.clearance_height_m = clearance_height_m

        # 소유하는 상태
        self._elapsed = np.zeros(4)          # 각 다리의 스윙 경과 시간 [s]
        self._liftoff_pos_W = np.zeros((3, 4))  # 발을 뗀 순간의 위치 (궤적 시작점)
        self._phase = np.zeros(4)            # 진행률 [0,1]. 로깅/디버깅용
        self._max_phase_seen = 0.0           # 클립 전 최댓값 — 위상 오버런 진단용

    # ── 읽기 전용 상태 ───────────────────────────────────────────────
    @property
    def phase(self) -> NDArray[np.float64]:
        """각 다리의 스윙 진행률 [0,1]. 지지기는 0."""
        return self._phase.copy()

    @property
    def max_phase_seen(self) -> float:
        """클립 전 위상의 최댓값.

        1.0 을 의미 있게 넘으면 스윙 시간(T_swing)과 위상 기반 접촉 구간이
        어긋났다는 뜻이다. 베지에가 외삽하게 되고, 그것이 결함 6 이었다.
        """
        return self._max_phase_seen

    # ── 갱신 ─────────────────────────────────────────────────────────
    def update(
        self,
        is_stance: NDArray[np.bool_],
        p_feet_W: NDArray[np.float64],
        p_feet_target_W: NDArray[np.float64],
        dt: float,
    ) -> NDArray[np.float64]:
        """한 스윙 틱을 진행하고 갱신된 발 위치를 돌려준다.

        Args:
            is_stance: (4,) True = 접지
            p_feet_W: (3,4) 현재 발 위치. 지지 다리는 그대로 유지된다
            p_feet_target_W: (3,4) 착지 목표점 (Raibert)
            dt: 이 틱이 실제로 흘린 시간 [s]

        Returns:
            (3,4) 갱신된 발 위치. 입력을 제자리 수정하지 않고 복사본을 낸다.

        지지 다리의 위치를 갱신하지 않는 것이 핵심이다 — 발이 땅에 박혀
        있으므로 이전 월드 좌표를 그대로 유지해야 한다.
        """
        p_out = p_feet_W.copy()

        for leg in range(4):
            if is_stance[leg]:
                # 착지(touchdown). 타이머를 되감고 위치는 건드리지 않는다.
                self._phase[leg] = 0.0
                if self._elapsed[leg] > 0:
                    self._elapsed[leg] = 0.0
                continue

            # 이지(liftoff) 순간: 궤적의 출발점을 고정한다
            if self._elapsed[leg] == 0.0:
                self._liftoff_pos_W[:, leg] = p_feet_W[:, leg].copy()

            self._elapsed[leg] += dt
            raw_phase = self._elapsed[leg] / self.swing_duration_s
            self._max_phase_seen = max(self._max_phase_seen, raw_phase)
            phase = np.clip(raw_phase, 0.0, 1.0)
            self._phase[leg] = phase

            # 3차 베지에: 이지점 -> 위로 -> 착지점 위 -> 착지점
            p0 = self._liftoff_pos_W[:, leg]
            p3 = p_feet_target_W[:, leg].copy()
            lift = np.array([0.0, 0.0, self.clearance_height_m])
            p_out[:, leg] = bezier(phase, [p0, p0 + lift, p3 + lift, p3])

        return p_out
