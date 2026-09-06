"""테스트 시나리오 정의. 여기가 유일한 출처(single source of truth)입니다."""
from __future__ import annotations

from typing import TypedDict


class ScenarioSpec(TypedDict, total=False):
    """run_scenario / run_scenario_mujoco 에 그대로 넘길 수 있는 인자 묶음.

    TypedDict 를 쓰는 이유
        평범한 dict 는 값 타입이 섞이면 dict[str, object] 로 추론되고, 그것을
        **kwargs 로 펼치면 타입 검사기가 모든 인자에 대해 불평한다. 그보다
        중요한 것은 **키 오타를 잡아 준다**는 점이다. duration_s 를 duration
        으로 잘못 쓰면 run_scenario 는 기본값 3.0 초로 조용히 돌아가고,
        기준선은 '내가 의도한 것과 다른 시나리오'로 캡처된다.

        total=False 인 이유: 시나리오마다 명시하는 인자가 다르고, 나머지는
        run_scenario 의 기본값을 쓰는 것이 의도이기 때문이다.
    """

    gait_name: str
    v_des_x: float
    v_des_y: float
    omega_z_deg_s: float
    duration_s: float
    horizon: int
    mpc_hz: int
    red_hz: int
    sim_hz: int
    log_hz: int
    clearance_height: float


SCENARIOS: dict[str, ScenarioSpec] = {
    "S0_standing":  ScenarioSpec(gait_name="standing", v_des_x=0.0, omega_z_deg_s=0.0,  duration_s=0.5),
    "S1_trot_fwd":  ScenarioSpec(gait_name="trotting", v_des_x=1.0, omega_z_deg_s=0.0,  duration_s=3.0),
    "S2_endurance": ScenarioSpec(gait_name="trotting", v_des_x=1.0, omega_z_deg_s=0.0,  duration_s=10.0),
    "S3_yaw":       ScenarioSpec(gait_name="trotting", v_des_x=0.0, omega_z_deg_s=20.0, duration_s=6.0),
}
