"""보행 스케줄 — 어느 다리가 언제 땅에 있는가.

이 모듈은 **그림을 그리지 않는다.** 예전에는 여기 matplotlib 을 import 하는
시각화 함수와 __main__ 데모가 함께 있었다. 그러면 제어기를 import 하는 것만으로
플로팅 라이브러리가 딸려 오고, "control 은 viz 를 모른다"는 의존 방향이 깨진다.
그림은 quadruped_mpc.viz.gait_plot 으로 옮겼다.
"""
from __future__ import annotations

from typing import TypedDict

# --- 1. 상수 정의 ---
AIR = 1
GROUND = 0
FR, FL, RR, RL = 0, 1, 2, 3

class GaitSpec(TypedDict):
    """보행 모드 하나의 정의.

    평범한 dict 로 두면 값 타입이 섞여 dict[str, object] 로 추론되고,
    소비자 쪽에서 phase_offset 을 순회하는 순간 "object 는 iterable 이 아니다"
    로 막힌다. 실제로 scripts/run_sim.py 가 그렇게 막혔다.

    타입이 없는 경계는 그 경계를 쓰는 **모든 곳**에 비용을 전가한다.
    호출부마다 cast 를 붙이는 대신 정의 한 곳에 타입을 준다.
    """

    duty_cycle: float       #: 한 주기 중 발이 땅에 있는 비율 (1.0 = 항상 지지)
    phase_offset: list[float]   #: 다리별 위상 오프셋, [FR, FL, RR, RL]
    cycle_time: float       #: 보행 1 주기 [s]
    description: str


# --- 2. 보행 모드 파라미터 딕셔너리 ---
GAIT_PARAMS: dict[str, GaitSpec] = {
    "standing": {
        "duty_cycle": 1.0, # 보행 주기 내에 지면에 발이 닫는 비율
        "phase_offset": [0.0, 0.0, 0.0, 0.0], # 각 다리별 보행 사이클 오프셋
        "cycle_time": 0.5, # 보행 1 사이클 주기 "
        "description": "정지 상태 (모든 다리가 항상 지지)"
    },
    "trotting": {
        "duty_cycle": 0.5,
        "phase_offset": [0.0, 0.5, 0.5, 0.0],
        "cycle_time": 0.5,
        "description": "대각선 다리가 교차하는 2박자 보행"
    },
    "flying_trot": {
        "duty_cycle": 0.35,
        "phase_offset": [0.0, 0.5, 0.5, 0.0],
        "cycle_time": 0.4,
        "description": "체공 구간이 있는 빠른 트로팅"
    },
    "bounding": {
        "duty_cycle": 0.4,
        "phase_offset": [0.0, 0.0, 0.5, 0.5],
        "cycle_time": 0.33,
        "description": "앞다리 쌍과 뒷다리 쌍이 번갈아 도약"
    },
    "galloping": {
        "duty_cycle": 0.25,
        "phase_offset": [0.0, 0.1, 0.5, 0.6],
        "cycle_time": 0.3,
        "description": "가장 빠른 4박자 동적 보행 (Rotary gallop)"
    }
}

# --- 3. 제어 함수 ---
def get_gait_parameters(gait_name: str) -> tuple[float, float, list[float]]:
    """보행 모드 이름으로 (주기, duty, 위상 오프셋) 을 돌려준다.

    반환 순서가 정의 순서(duty, offset, cycle)와 다르다. 호출부가 전부
    `cycle_time, duty, offset = ...` 로 풀고 있어 지금 바꾸면 조용히
    뒤바뀔 수 있다. 세 값의 타입이 float, float, list 로 서로 달라
    검사기가 잡아주기는 하지만, 앞의 둘은 구별하지 못한다.

    Raises:
        ValueError: 등록되지 않은 보행 모드 이름.
    """
    if gait_name not in GAIT_PARAMS:
        raise ValueError(f"지원하지 않는 보행 모드입니다: {gait_name}. "
                         f"사용 가능한 모드: {list(GAIT_PARAMS)}")

    params = GAIT_PARAMS[gait_name]
    return params["cycle_time"], params["duty_cycle"], params["phase_offset"]


def get_contact_state(global_phase: float, duty_cycle: float,
                      phase_offsets: list[float]) -> list[int]:
    """현재 위상에서 네 다리의 접촉 상태 [FR, FL, RR, RL] 를 돌려준다.

    Args:
        global_phase: 보행 주기 내 진행률 0.0~1.0.
        duty_cycle: 지지 비율. 1.0 이면 항상 지지(standing).
        phase_offsets: 다리별 위상 오프셋.

    Returns:
        길이 4 의 리스트. 값은 GROUND(0) 또는 AIR(1).
        **이 규약은 옛것이다** — 0 이 접촉이라 `== 0` 으로 마스크를 만들어야
        한다. 러너는 `is_stance = np.asarray(sa) == 0` 으로 뒤집어 쓴다.
    """
    sa = [GROUND, GROUND, GROUND, GROUND]

    for leg in range(4):
        # 각 다리의 위상(offset 반영)을 0.0 ~ 1.0 으로 정규화
        leg_phase = (global_phase + phase_offsets[leg]) % 1.0
        # duty_cycle 이내면 지지(GROUND), 넘어가면 체공(AIR)
        sa[leg] = GROUND if leg_phase < duty_cycle else AIR

    return sa
