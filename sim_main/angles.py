"""각도 관측의 연속화(unwrap).

MuJoCo·IMU 처럼 자세를 쿼터니언이나 회전행렬로 들고 있는 소스에서 오일러각을
꺼내면 값이 항상 (-pi, pi] 로 접혀서 나온다. 반면 Convex MPC 는 yaw 를
'누적되는 연속량'으로 다룬다 — 참조 궤적을 current_yaw + omega_z * t 로 쌓고,
비용 함수가 (yaw - yaw_ref)^2 를 벌한다.

이 둘을 그대로 이으면 로봇이 반 바퀴를 도는 순간 측정 yaw 가 +pi 에서 -pi 로
점프하고, MPC 는 2pi (= 360도) 짜리 자세 오차를 본다. L_w_th 가 걸린 그 항은
즉시 최대 토크를 요구하고, 한 스텝 만에 발산한다.

SRBD 플랜트에는 이 문제가 없다. 오일러각을 직접 적분하므로 3.14 다음이 3.15 이지
-3.14 가 아니다. 다시 말해 이 결함은 **플랜트를 MuJoCo 로 바꾸는 순간에만 생기고,
기존 시나리오(S0/S1/S3)로는 절대 재현되지 않는다.** 그래서 여기서 미리 만들고
미리 테스트한다 — 발산을 본 뒤에 원인을 찾으면 훨씬 비싸다.

전제: 관측 간격 사이의 각도 변화가 pi 미만이어야 한다. 1 kHz 관측이면
pi rad/step = 3141 rad/s = 180000 deg/s 이므로 물리적으로 위반할 수 없다.
이 전제가 깨졌는지는 max_abs_step_rad 로 사후에 확인한다.
"""

from __future__ import annotations

import numpy as np

TWO_PI = 2.0 * np.pi


def wrap_to_pi(angle_rad):
    """각도를 [-pi, pi) 로 접는다. 스칼라와 배열 모두 받는다."""
    return (np.asarray(angle_rad) + np.pi) % TWO_PI - np.pi


class AngleUnwrapper:
    """접힌 각도열을 연속 각도열로 편다. 한 축(예: yaw)당 하나씩 쓴다.

    상태를 가지므로 '누가 이 상태의 주인인가'가 분명해야 한다.
    MuJoCoPlant 가 소유하고, observe() 안에서만 갱신한다.

    update() 는 같은 입력에 대해 멱등(idempotent)하다 — 같은 값을 두 번 넣으면
    delta 가 0 이므로 결과가 변하지 않는다. 따라서 한 스텝에 observe() 를
    두 번 불러도 안전하다. 반대로 **여러 스텝을 건너뛰고 부르면 안 된다**:
    그 사이 각도가 pi 이상 변했다면 복원할 방법이 원리적으로 없다.

    Examples:
        >>> u = AngleUnwrapper()
        >>> round(u.update(3.10), 3)
        3.1
        >>> round(u.update(-3.14), 3)   # +pi 를 넘어 접힌 관측
        3.143
    """

    def __init__(self, initial_angle_rad: float | None = None) -> None:
        """
        Args:
            initial_angle_rad: 연속 각도의 시작값. None 이면 첫 update() 의
                관측값을 그대로 시작값으로 삼는다. 로봇이 yaw=0 이 아닌
                자세에서 출발하는 경우에만 명시한다.
        """
        self._continuous: float | None = None
        self._prev_wrapped: float | None = None
        self._max_abs_step: float = 0.0

        if initial_angle_rad is not None:
            self._continuous = float(initial_angle_rad)
            self._prev_wrapped = float(wrap_to_pi(initial_angle_rad))

    @property
    def value(self) -> float:
        """현재 연속 각도 [rad]. update() 를 한 번도 안 불렀으면 ValueError."""
        if self._continuous is None:
            raise ValueError("update() 를 한 번도 호출하지 않았다.")
        return self._continuous

    @property
    def max_abs_step_rad(self) -> float:
        """관측된 한 스텝 최대 각도 변화량 [rad] (진단용).

        이 값이 pi 에 가까워지면 unwrap 전제가 위태롭다는 뜻이다.
        swing.max_phase_seen 과 같은 역할 — 조용히 넘어갈 수 있는 가정을
        숫자로 남겨 사후에 검사할 수 있게 한다.
        """
        return self._max_abs_step

    def update(self, wrapped_angle_rad: float) -> float:
        """접힌 관측값 하나를 받아 연속 각도를 갱신하고 반환한다."""
        wrapped = float(wrapped_angle_rad)

        if self._prev_wrapped is None:
            self._prev_wrapped = wrapped
            self._continuous = wrapped
            return self._continuous

        # 두 접힌 값의 차이를 다시 접으면 '최단 회전 경로'가 된다.
        # 이것이 unwrap 의 전부다: +pi -> -pi 점프(-2pi)는 +eps 로 읽힌다.
        delta = float(wrap_to_pi(wrapped - self._prev_wrapped))

        self._max_abs_step = max(self._max_abs_step, abs(delta))
        self._continuous += delta
        self._prev_wrapped = wrapped
        return self._continuous

    def reset(self, initial_angle_rad: float | None = None) -> None:
        """시나리오를 다시 돌릴 때 호출한다. 진단값도 함께 초기화된다."""
        self.__init__(initial_angle_rad)  # noqa: PLC2801
