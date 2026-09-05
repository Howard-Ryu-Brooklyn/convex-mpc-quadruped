"""Plant 인터페이스 — 제어기가 보는 '세상'.

이 추상화가 필요한 이유
    현재 mujoco_main_ref.py 는 같은 제어 로직을 MuJoCo 에 꽂기 위해 전체를
    복사한 630 줄이다. 그리고 결함 1, 2, 8 이 그 파일에는 하나도 고쳐져
    있지 않다. 중복은 편의의 문제가 아니라 정확성의 문제다.
    같은 Controller 가 여러 Plant 에 꽂히면 그 복제본이 사라진다.

인터페이스를 어느 수준에 둘 것인가 — 이것이 설계의 핵심이다
    SRBD 는 발 위치를 운동학적으로 강제하는 시뮬레이터이고, MuJoCo 는
    관절 토크로 구동되는 물리 엔진이다. 추상화 수준이 다르다.

    해법은 인터페이스를 '제어기가 내는 것' 기준으로 정의하는 것이다.
    제어기가 내는 것은 언제나 (지지 다리 지면 반발력, 발 위치 목표) 이고,
    그것을 실제 구동으로 바꾸는 책임은 각 Plant 가 진다.

        SRBDPlant   : 발 위치를 그대로 강제하고 GRF 로 몸통을 적분
        MuJoCoPlant : GRF 를 자코비안 전치로 관절 토크로, 발 위치 목표를
                      스윙 토크로 변환한 뒤 엔진에 넣는다
                      (mujoco_main_ref.py 가 이미 이렇게 하고 있다)
        RealRobot   : 같은 변환을 실기 모터 드라이버로 보낸다

    이 경계가 유지되면 Step 10(하드웨어 이관)은 구현체 하나를 더하는 일이 된다.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from robot_types import ControlCommand, RobotState


class PlantBase(ABC):
    """시뮬레이터/실기의 공통 인터페이스."""

    @property
    @abstractmethod
    def dt(self) -> float:
        """물리 적분 주기 [s]."""

    @abstractmethod
    def step(self, command: ControlCommand) -> None:
        """한 물리 스텝을 진행한다.

        지령을 낱개 인자가 아니라 ControlCommand 하나로 받는 이유는
        robot_types.ControlCommand 의 독스트링에 있다. 요약하면, 다리의 두
        상태(스탠스/스윙)가 서로 다른 종류의 지령을 받고, 그 묶음이 실기에
        내려보내는 패킷과 같은 모양이기 때문이다.

        구현체는 자기가 쓰지 않는 필드를 무시해도 된다. 다만 **필요한데
        None 인 필드는 조용히 0 으로 대체하지 말고 거부해야 한다** —
        "안 줬다" 와 "0 을 지령했다" 는 다른 사실이다.

        Args:
            command: 이번 스텝의 지령. 스윙 다리 힘이 0 이라는 물리적 요건은
                ControlCommand 구성 시점에 이미 검사되어 있다.
        """

    @abstractmethod
    def observe(self) -> RobotState:
        """현재 상태의 스냅샷을 반환한다.

        반환값은 내부 배열의 참조가 아니라 복사본이어야 한다. 호출자가
        반환값을 통해 Plant 내부를 오염시킬 수 없어야 한다.
        """
