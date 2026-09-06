"""시스템 전역 데이터 구조.

파일명이 types.py 가 아닌 이유: sim_main 이 sys.path 에 직접 얹혀 있어
types.py 는 표준 라이브러리의 types 모듈을 가린다. dataclasses, enum 을
비롯한 다수의 표준/서드파티 모듈이 내부적으로 import types 를 하므로
무엇이 언제 깨질지 예측할 수 없다. 1-6 에서 정식 패키지
(quadruped_mpc.types)로 옮기면 네임스페이스가 생겨 안전해진다.

이 파일은 지금까지 코드가 얼버무리고 있던 규약들을 명시적으로 못 박는다.
Step 1-1 시점에서는 아무도 import하지 않는 순수 추가이며, 기존 함수와의
동등성은 tests/test_types.py 가 증명한다(characterization test).

프레임 접미사 규약 — 예외 없이 지킨다
    _W : world frame (관성계). 중력이 -z 방향.
    _B : body frame  (몸통 고정, 원점 = CoM)
    _H : hip-local   (다리별 고관절 원점. IK 입력)

배열 규약
    Vec3x4 의 열 인덱스는 항상 Leg enum 이다 (다리 4개).
    시간 축을 갖는 것은 이름에 _traj / _seq 를 붙여 구분한다.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Annotated

import numpy as np
from numpy.typing import NDArray

# ── 배열 별칭 ────────────────────────────────────────────────────────────
Vec3 = Annotated[NDArray[np.float64], "(3,)"]
Vec3x4 = Annotated[NDArray[np.float64], "(3, 4) — 열 인덱스 = Leg"]
Mat3 = Annotated[NDArray[np.float64], "(3, 3)"]
Bool4 = Annotated[NDArray[np.bool_], "(4,) — 인덱스 = Leg"]


# ── MPC 전용 별칭 ────────────────────────────────────────────────────────
# 13차원 상태의 레이아웃은 RobotState.to_mpc_vector() 가 정의한다.
# 그 규약이 지금까지 convex_mpc.py 쪽에는 이름으로 존재하지 않았고,
# 결함 1(Ac[0:3,6:9] 의 전치 누락)이 나온 자리가 정확히 거기다.
StateVec13 = Annotated[NDArray[np.float64], "(13,1) — [Θ(3) p(3) ω(3) v(3) g(1)]"]
StateTraj13k = Annotated[NDArray[np.float64], "(13k,1) — horizon 참조 궤적"]
ForceVec12 = Annotated[NDArray[np.float64], "(12,) — [fx fy fz] x 4, Leg 순서"]
ContactSeq4k = Annotated[NDArray[np.float64], "(4,k) — 구 규약: 0=GROUND, 1=AIR"]


class Leg(IntEnum):
    """다리 인덱스.

    지금까지 config.hip_location_bf 의 주석에만 존재하던 암묵 규약을 강제한다.
    IntEnum 이므로 배열 인덱스로 바로 쓸 수 있다: feet.pos_W[:, Leg.FR]

    MuJoCo 모델의 관절 순서가 이와 다를 수 있으므로, MuJoCoPlant 는 자신의
    __init__ 에서 매핑 테이블을 한 번만 만들어 이 순서로 정렬해 반환한다.
    """

    FR = 0  # front right
    FL = 1  # front left
    RR = 2  # rear right
    RL = 3  # rear left


class SolverStatus(IntEnum):
    """QP 솔버 결과. 실패의 종류를 구분해야 사후 분석이 가능하다."""

    SOLVED = 0
    MAX_ITER_REACHED = 1
    PRIMAL_INFEASIBLE = 2
    DUAL_INFEASIBLE = 3
    UNKNOWN_FAILURE = 4


@dataclass(frozen=True)
class RobotState:
    """단일 강체(SRB) 상태. 모든 계층이 공유하는 단일 진실.

    자세 표현 — ZYX 오일러각
        참고 논문(Di Carlo et al., Convex MPC)이 ZYX 오일러각을 전제로
        선형화를 유도했고, 현 SRBD 동역학과 MPC 의 Ac 행렬이 모두 그 유도를
        따른다. 쿼터니언으로 바꾸면 재작성이 되어 회귀 테스트의 의미가
        사라지므로 오일러를 정본으로 삼는다.

        한계: pitch = ±90° 에서 짐벌락. dynamics.py 가 ±89° 로 클립하고 있고,
        4족보행에서 그 클립이 발동하는 상황은 표현의 문제가 아니라 이미 전복이다.
        공중제비 등 pitch 가 ±90° 를 넘는 동작이 필요해지면 그때 쿼터니언으로
        옮기고, 그것은 별도 단계로 다룬다.

    yaw 는 unwrapped(연속)다
        현 SRBD 는 각속도를 순수 적분하므로 yaw 가 감기지 않는다(360°, 720° …).
        반면 쿼터니언에서 뽑은 오일러각은 (-pi, pi] 로 감긴다. 그대로 쓰면
        180° 를 넘는 순간 yaw 가 점프해 MPC 의 horizon 예측
        (current_yaw + omega_z * t_i)과 자세 오차항이 동시에 깨진다.
        따라서 MuJoCoPlant 는 이전 yaw 를 기억해 ±2pi 를 보정하는
        상태 있는(stateful) unwrap 을 수행할 책임을 진다.

    omega_W 는 world frame 각속도다
        MPC 의 Θ̇ = R_z(psi)^T · omega 유도와 dynamics 의 자이로항
        omega × (I_W omega) 가 모두 world 를 전제한다.
        MuJoCo 와 실기 IMU 는 통상 body frame 각속도를 주므로,
        Plant 경계에서 omega_B -> omega_W 변환이 필요하다.
    """

    p_com_W: Vec3  # CoM 위치 [m]
    v_com_W: Vec3  # CoM 선속도 [m/s]
    rpy_W: Vec3  # ZYX 오일러각 [rad] (roll, pitch, yaw), yaw 는 unwrapped
    omega_W: Vec3  # 각속도 [rad/s], world frame

    def __post_init__(self) -> None:
        for name in ("p_com_W", "v_com_W", "rpy_W", "omega_W"):
            value = getattr(self, name)
            if not isinstance(value, np.ndarray) or value.shape != (3,):
                raise ValueError(
                    f"{name} must be an ndarray of shape (3,), got "
                    f"{getattr(value, 'shape', type(value))}"
                )

    @property
    def roll(self) -> float:
        return float(self.rpy_W[0])

    @property
    def pitch(self) -> float:
        return float(self.rpy_W[1])

    @property
    def yaw(self) -> float:
        return float(self.rpy_W[2])

    def to_mpc_vector(self) -> NDArray[np.float64]:
        """MPC 용 13차원 상태 [Theta, p, omega, v, 1.0]^T.

        마지막 1.0 은 중력을 선형 시스템에 포함시키기 위한 상수항이다.
        배치는 구 config.get_13d_state (1-5b 에서 제거) 와 동일해야 하며,
        tests/test_types.py::test_to_mpc_vector_matches_legacy_layout 이
        그 수식을 인라인해 비트 단위로 검증한다.
        """
        return np.concatenate(
            [self.rpy_W, self.p_com_W, self.omega_W, self.v_com_W, [1.0]]
        )


@dataclass(frozen=True)
class FootState:
    """발 관련 상태의 단일 소유자.

    pos_W 와 rel_com_W 는 단위가 같지만 의미가 다르다. 별도 필드로 두어
    결함 5(상대 벡터를 절대 위치 자리에 대입)를 구조적으로 불가능하게 한다.

    is_stance 는 True 가 접지다. 기존 규약(AIR=1, GROUND=0)은 '접촉'이라는
    이름의 값에서 1 이 '접촉하지 않음'을 뜻해 이중 부정
    (stance_flags = 1.0 - contact_sequence)을 낳았다.
    """

    pos_W: Vec3x4  # 발끝 절대 위치 [m]. 지면 접촉 판정, 스윙 궤적 생성
    rel_com_W: Vec3x4  # CoM -> 발끝 벡터 [m]. MPC 토크암 r_i
    is_stance: Bool4  # True = 접지

    def __post_init__(self) -> None:
        for name in ("pos_W", "rel_com_W"):
            value = getattr(self, name)
            if not isinstance(value, np.ndarray) or value.shape != (3, 4):
                raise ValueError(
                    f"{name} must be an ndarray of shape (3, 4), got "
                    f"{getattr(value, 'shape', type(value))}"
                )
        if not isinstance(self.is_stance, np.ndarray) or self.is_stance.shape != (4,):
            raise ValueError(
                f"is_stance must be an ndarray of shape (4,), got "
                f"{getattr(self.is_stance, 'shape', type(self.is_stance))}"
            )


@dataclass(frozen=True)
class ControlOutput:
    """제어기 출력. 성공/실패와 진단 정보를 함께 나른다.

    실패를 np.zeros(12) 로 바꿔치기하면 상위 계층이 '성공한 0 힘'과 구분할 수
    없다. 실기에서 넘어진 원인을 로그에서 찾지 못하게 되는 가장 흔한 경로이며,
    0 힘은 네 다리가 동시에 힘을 놓는다는 뜻이라 그 자체로 위험하다.
    안전한 대체값(직전 유효 해, 중력 보상 균등 분배)의 선택은 호출자의 몫이고,
    제어기의 책임은 실패를 정확히 보고하는 데까지다.
    """

    forces_W: Vec3x4  # 지면 반발력 [N]
    status: SolverStatus
    solve_time_s: float  # Step 7 의 실시간성 검증이 여기서 시작된다
    iterations: int

    @property
    def is_valid(self) -> bool:
        return self.status is SolverStatus.SOLVED


@dataclass(frozen=True)
class HorizonPlan:
    """MPC 한 번의 풀이에 필요한 미래 정보 전부. 매 MPC 틱마다 새로 생성된다.

    ★ 결함 2 가 구조적으로 불가능해지는 지점.
      이전에는 r_feet_traj 가 루프 밖 리스트였고 append 만 되어, '누가 비우는가'에
      답할 사람이 없었다. 그 결과 solve() 가 읽는 앞의 k 개는 영원히 첫 호출의
      값이었다. 반환값에는 '이전 호출의 값이 남아 있다'가 존재할 수 없다.
      규율이 아니라 구조로 막는다.

    ★ 결함 8 이 제자리를 찾는 지점.
      r_feet_W 는 스텝별 리스트다. 현재는 전 구간에 같은 값을 넣는 근사지만
      (TODO Step 1-4b), 스텝 k 에서 접지할 다리의 위치를 예측해 넣을 자리가
      타입에 이미 마련되어 있다.
    """

    x_ref: NDArray[np.float64]      # (13*k, 1) 참조 상태 궤적
    yaw_ref: NDArray[np.float64]    # (k,)      각 스텝의 예측 yaw [rad]
    r_feet_W: list                  # k 개의 (3,4) — 스텝별 CoM 기준 발 위치(토크암)
    is_stance: NDArray[np.bool_]    # (4, k)    True = 접지

    def __post_init__(self) -> None:
        k = len(self.r_feet_W)
        if self.x_ref.shape != (13 * k, 1):
            raise ValueError(f"x_ref shape must be ({13 * k}, 1), got {self.x_ref.shape}")
        if self.yaw_ref.shape != (k,):
            raise ValueError(f"yaw_ref shape must be ({k},), got {self.yaw_ref.shape}")
        if self.is_stance.shape != (4, k):
            raise ValueError(f"is_stance shape must be (4, {k}), got {self.is_stance.shape}")
        for i, r in enumerate(self.r_feet_W):
            if r.shape != (3, 4):
                raise ValueError(f"r_feet_W[{i}] shape must be (3, 4), got {r.shape}")

    @property
    def horizon(self) -> int:
        return len(self.r_feet_W)

    @property
    def contact_legacy(self) -> NDArray[np.float64]:
        """구 규약 (AIR=1, GROUND=0) 으로 변환한 (4, k) 배열.

        ConvexMPC._build_friction_cone 이 아직 이 형태를 받고 내부에서
        stance_flags = 1.0 - contact_sequence 로 되돌린다. 이중 부정이다.
        TODO(Step 1-4b): MPC 가 is_stance 를 직접 받도록 바꾸고 이 프로퍼티를 제거한다.
        """
        return 1.0 - self.is_stance.astype(float)


@dataclass(frozen=True)
class ControlCommand:
    """제어기가 한 물리 스텝 동안 Plant 에 내리는 지령 전부.

    왜 인터페이스를 이렇게 넓히는가
    ─────────────────────────────
    1-3 에서 PlantBase.step(forces_W, r_feet_W) 로 시작했다. SRBD 에는
    충분했지만 MuJoCo 에는 부족하다. 이유는 다리의 두 상태가 **다른 종류의
    지령**을 받기 때문이다.

        스탠스 다리 : 힘을 받는다      -> tau = J^T f
        스윙 다리   : 궤적을 받는다     -> tau = J^T (Kp e + Kd e_dot) + tau_ff(a_des)

    SRBD 는 발 위치를 운동학적으로 강제하므로 스윙 지령이 위치 하나로 충분했다.
    실제 물리 엔진은 발을 순간이동시킬 수 없으니 속도·가속도까지 필요하다.

    선택지는 셋이었다.
      (A) Plant 가 스윙 궤적 생성기를 내부에 갖는다  -> 계층 위반. 궤적은 계획이지 물리가 아니다.
      (B) ForcePlant / TorquePlant 로 인터페이스를 쪼갠다 -> 제어기가 Plant 종류를
          알아야 하므로 추상화가 무너진다. Step 10(실기)에서 다시 쪼개야 한다.
      (C) 지령을 한 묶음으로 넓히고, 필요 없는 필드는 Plant 가 무시한다.

    (C) 를 택했다. SRBD 가 v/a 를 무시하는 것은 게으름이 아니라 물리적으로
    정직한 해석이다 — "이상적 플랜트라 저수준 추종이 완벽하다". 그리고 실기에
    내려보내는 패킷이 정확히 이 묶음이라, 경계가 현실과 같은 모양이 된다.

    위치를 r_feet_W(CoM 상대) 하나로만 들고 있는 이유
    ────────────────────────────────────────────
    절대 위치 p_feet_W 를 함께 넣으면 두 값이 갈라질 수 있다. 결함 5 가 정확히
    그 사고였다(절대 위치 자리에 상대 벡터를 넣어 목표점이 지면 34cm 아래를
    향했다). Plant 는 자기 CoM 을 알고 있으므로 p = r + p_com 으로 언제든
    복원한다. **한 사실은 한 곳에만 둔다.**

    Attributes:
        forces_W: (3,4) 각 발의 지면 반발력 [N], world frame.
            스윙 다리는 **정확히 0** 이어야 한다 (__post_init__ 에서 검사).
        r_feet_W: (3,4) CoM 기준 발 위치 [m], world 정렬.
            스탠스는 접지 위치, 스윙은 궤적의 현재 목표점.
        is_stance: (4,) True = 접지.
        v_feet_W: (3,4) 스윙 발 목표 속도 [m/s]. None 이면 미제공.
        a_feet_W: (3,4) 스윙 발 목표 가속도 [m/s^2]. None 이면 미제공.
            None 과 0 은 다르다 — 전자는 "안 줬다", 후자는 "0 을 지령했다".
            토크 구동 Plant 는 None 을 받으면 명시적으로 거부해야 한다.
    """

    forces_W: Vec3x4
    r_feet_W: Vec3x4
    is_stance: Bool4
    v_feet_W: Vec3x4 | None = None
    a_feet_W: Vec3x4 | None = None

    def __post_init__(self) -> None:
        for name in ("forces_W", "r_feet_W"):
            arr = getattr(self, name)
            if arr.shape != (3, 4):
                raise ValueError(f"{name} shape must be (3, 4), got {arr.shape}")
        if self.is_stance.shape != (4,):
            raise ValueError(f"is_stance shape must be (4,), got {self.is_stance.shape}")

        for name in ("v_feet_W", "a_feet_W"):
            arr = getattr(self, name)
            if arr is not None and arr.shape != (3, 4):
                raise ValueError(f"{name} shape must be (3, 4), got {arr.shape}")

        # 결함 7 을 규율이 아니라 타입으로 막는다.
        # OSQP 는 fz in [0,0] 제약을 허용오차 안에서만 만족시키므로 스윙 다리에
        # ~1e-4 N 의 잔류력이 남는다. 마스킹을 잊으면 공중에 뜬 다리에 토크가
        # 새고, 원인이 '솔버 수렴 오차'라 로그만 봐서는 절대 못 찾는다.
        # 물리적으로 반드시 성립해야 하는 조건은 구성 시점에 강제한다.
        swing_force = self.forces_W[:, ~self.is_stance]
        if swing_force.size and np.any(swing_force != 0.0):
            worst = np.abs(swing_force).max()
            raise ValueError(
                "스윙 다리에 0 이 아닌 힘이 실렸다 (최대 "
                f"{worst:.3e} N). 결함 7 — MPC 해에 접촉 마스크를 곱했는지 "
                "확인할 것: forces_W * is_stance"
            )

    @property
    def n_stance(self) -> int:
        return int(np.count_nonzero(self.is_stance))
