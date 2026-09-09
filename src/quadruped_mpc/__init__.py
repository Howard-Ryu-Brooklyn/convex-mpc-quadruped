"""Convex MPC 기반 사족보행 시뮬레이터.

하위 패키지의 경계는 '무엇이 바뀌면 여기가 바뀌는가'로 나눴다.

    core/         좌표·시간·기하 — 로봇이 바뀌어도 안 바뀐다
    control/      MPC·발판 계획·스윙 궤적 — 제어기가 바뀌면 바뀐다
    plants/       SRB(해석) / MuJoCo(엔진) — 물리 모델이 바뀌면 바뀐다
    experiments/  시나리오와 러너 — 실험 설계가 바뀌면 바뀐다
    viz/          그림 — 아무것도 여기에 의존하지 않는다

의존 방향은 위에서 아래로만 흐른다. core 는 control 을 모르고,
control 은 plants 를 모른다. 러너만이 셋을 조립한다.
"""
__all__ = ["paths"]
__version__ = "0.1.0"
