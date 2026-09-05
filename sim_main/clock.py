"""다중 주기(multi-rate) 제어 타이밍.

이 시스템은 서로 다른 주기로 도는 네 개의 루프를 갖는다.

    blue  (MPC)          30 Hz   — 지면 반발력 최적화
    red   (스윙/추정)   1000 Hz   — 발 궤적 갱신
    green (저수준)      4500 Hz   — 관절 제어 (미구현)
    sim   (물리)        9000 Hz   — 동역학 적분

분리 전에는 이 주기들과 그로부터 나온 나눗셈(step_blue, step_red, step_vis,
sim_ts, mpc_dt, dt_red)이 메인 함수 앞부분에 흩어져 있었고, 어떤 것이 어떤
루프에 속하는지는 변수 이름의 색깔 주석에만 있었다.

여기 모아두면 (a) 주기 관계를 한눈에 볼 수 있고 (b) 단위 테스트가 가능해지며
(c) Step 7 의 실시간성 검증이 이 객체를 기준으로 예산을 계산할 수 있다.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MultiRateClock:
    """제어 주기와 스텝 판정.

    Attributes:
        sim_hz: 물리 적분 주기 [Hz]
        mpc_hz: MPC 연산 주기 [Hz]
        swing_hz: 스윙 궤적 갱신 주기 [Hz]
        log_hz: 로깅 주기 [Hz]
        duration_s: 전체 시뮬레이션 시간 [s]
    """

    sim_hz: int = 9000
    mpc_hz: int = 30
    swing_hz: int = 1000
    log_hz: int = 30
    duration_s: float = 3.0

    def __post_init__(self) -> None:
        """하위 주기가 물리 주기의 정수 약수인지 확인한다.

        정수배가 아니면 조용한 모델 오차가 생긴다. 예를 들어
        sim_hz=1000, mpc_hz=30 이면 mpc_every = round(33.33) = 33 이므로
        MPC 는 실제로 33ms 마다 호출되는데, 모델은 mpc_dt = 1/30 = 33.33ms
        로 이산화한다. 1% 의 시간 오차가 Ad = I + Ac*dt 에 그대로 들어가고,
        아무도 알아채지 못한다.

        현재 값(9000/30/1000/30)은 모두 나누어떨어지므로 동작 변경은 없다.
        """
        for name, hz in (("mpc_hz", self.mpc_hz),
                         ("swing_hz", self.swing_hz),
                         ("log_hz", self.log_hz)):
            if self.sim_hz % hz != 0:
                raise ValueError(
                    f"sim_hz({self.sim_hz}) 가 {name}({hz}) 의 정수배가 아니다. "
                    f"실제 호출 간격({round(self.sim_hz / hz) / self.sim_hz * 1e3:.3f}ms)과 "
                    f"모델이 가정하는 간격({1e3 / hz:.3f}ms)이 어긋난다."
                )

    # ── 적분 주기 ────────────────────────────────────────────────────
    @property
    def dt(self) -> float:
        """물리 적분 간격 [s]."""
        return 1 / self.sim_hz

    @property
    def n_steps(self) -> int:
        return round(self.duration_s / self.dt)

    # ── '몇 스텝마다' ────────────────────────────────────────────────
    @property
    def mpc_every(self) -> int:
        return round(self.sim_hz / self.mpc_hz)

    @property
    def swing_every(self) -> int:
        return round(self.sim_hz / self.swing_hz)

    @property
    def log_every(self) -> int:
        return round(self.sim_hz / self.log_hz)

    # ── 각 루프가 보는 dt ────────────────────────────────────────────
    @property
    def mpc_dt(self) -> float:
        """MPC 가 horizon 을 쪼개는 간격 [s]. 1/mpc_hz 다.

        주의: dt * mpc_every 와 미세하게 다를 수 있다 (round 때문에).
        MPC 모델의 이산화에는 이 값을 쓴다.
        """
        return 1 / self.mpc_hz

    @property
    def swing_dt(self) -> float:
        """스윙 루프가 실제로 겪는 간격 [s]. 스윙 위상 적분에 쓴다."""
        return self.dt * self.swing_every

    # ── 스텝 판정 ────────────────────────────────────────────────────
    def is_mpc_tick(self, step: int) -> bool:
        return step % self.mpc_every == 0

    def is_swing_tick(self, step: int) -> bool:
        return step % self.swing_every == 0

    def is_log_tick(self, step: int) -> bool:
        return step % self.log_every == 0

    def time_at(self, step: int) -> float:
        """스텝 인덱스 -> 시각 [s]."""
        return step * self.dt
