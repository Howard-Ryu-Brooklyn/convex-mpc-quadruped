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


# ═══════════════════════════════════════════════════════════════════════
# 두 개의 층. 섞으면 안 된다.
# ═══════════════════════════════════════════════════════════════════════
#
# REGRESSION  골든 기준선이 물려 있다. 여기 숫자를 바꾸면 test_golden 이
#             깨지고, `make accept` 로 기준선을 다시 떠야 한다. 즉 이 값을
#             바꾸는 것은 '실험'이 아니라 '기준의 변경'이다.
#
# GALLERY     아무 테스트도 물려 있지 않다. 마음껏 바꿔도 되고, 바꾸라고
#             있는 것이다. 속도·지속시간·클리어런스를 여기서 만진다.
#
# 층을 나누지 않으면 "그림 좀 보려고 duration 을 늘렸는데 테스트가 빨개졌다"
# 가 반복되고, 결국 사람이 빨간불을 무시하기 시작한다.

REGRESSION: dict[str, ScenarioSpec] = {
    "S0_standing":  ScenarioSpec(gait_name="standing", v_des_x=0.0, omega_z_deg_s=0.0,  duration_s=0.5),
    "S1_trot_fwd":  ScenarioSpec(gait_name="trotting", v_des_x=1.0, omega_z_deg_s=0.0,  duration_s=3.0),
    "S2_endurance": ScenarioSpec(gait_name="trotting", v_des_x=1.0, omega_z_deg_s=0.0,  duration_s=10.0),
    "S3_yaw":       ScenarioSpec(gait_name="trotting", v_des_x=0.0, omega_z_deg_s=20.0, duration_s=6.0),
}

#: 보행 모드별 관찰용. 속도는 각 게이트의 duty/주기에 맞춰 골랐다 —
#: galloping 을 0.5 m/s 로 돌리는 것은 물리적으로 말이 안 된다(체공 구간이
#: 있는데 전진하지 않으면 그냥 제자리 도약이다).
#: 공격적인 게이트는 **넘어질 것으로 예상한다.** 그것이 이 실험의 내용이다 —
#: 선형화된 MPC 가 어느 게이트까지 버티는가.
GALLERY: dict[str, ScenarioSpec] = {
    "G_stand":       ScenarioSpec(gait_name="standing",    v_des_x=0.0, omega_z_deg_s=0.0,  duration_s=2.0),
    "G_trot":        ScenarioSpec(gait_name="trotting",    v_des_x=1.0, omega_z_deg_s=0.0,  duration_s=4.0),
    "G_trot_slow":   ScenarioSpec(gait_name="trotting",    v_des_x=0.3, omega_z_deg_s=0.0,  duration_s=4.0),
    "G_trot_yaw":    ScenarioSpec(gait_name="trotting",    v_des_x=0.5, omega_z_deg_s=30.0, duration_s=4.0),
    "G_trot_side":   ScenarioSpec(gait_name="trotting",    v_des_x=0.0, v_des_y=0.4, omega_z_deg_s=0.0, duration_s=4.0),
    "G_flying_trot": ScenarioSpec(gait_name="flying_trot", v_des_x=1.5, omega_z_deg_s=0.0,  duration_s=3.0, clearance_height=0.07),
    "G_bound":       ScenarioSpec(gait_name="bounding",    v_des_x=1.5, omega_z_deg_s=0.0,  duration_s=3.0, clearance_height=0.08),
    "G_gallop":      ScenarioSpec(gait_name="galloping",   v_des_x=2.0, omega_z_deg_s=0.0,  duration_s=3.0, clearance_height=0.08),
}

#: 이름으로 찾을 때의 단일 창구. 두 층을 합치되 출처는 위에 남아 있다.
SCENARIOS: dict[str, ScenarioSpec] = {**REGRESSION, **GALLERY}
