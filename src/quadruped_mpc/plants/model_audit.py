"""MuJoCo 모델과 config.py 의 대조 감사.

왜 이 모듈이 필요한가
─────────────────────
SRB 시뮬레이션과 MuJoCo 시뮬레이션은 **다른 질문에 답한다**.

    SRB     : MPC 알고리즘 자체가 옳은가?   → 모델 오차가 정의상 0 이어야 한다.
    MuJoCo  : 선형화 가정이 실제 조건에서 얼마나 강인한가?
              → 모델 오차가 **있어야 한다**. 그게 측정 대상이다.

그래서 두 모델의 상수를 억지로 맞추면 안 된다. 질량을 45.84 로 고치는 순간
"6.6% 질량 오차에서 이 MPC 가 걷는가" 라는, 답할 가치가 있는 질문이 사라진다.
마찰도 같다. 피라미드로 푼 해가 원뿔 제약에서 미끄러지는지가 곧 논문 근사의
대가를 재는 실험인데, 미리 mu/sqrt(2) 로 줄이면 재려던 것을 없애고 재게 된다.

다만 두 종류를 구분해야 한다.

    A. 모델 불확실성 = "값이 다르다"   → 제어기가 극복할 대상. 그대로 둔다.
    B. 좌표계/규약 불일치 = "의미가 다르다" → 제어기가 극복할 수 없다. 버그다.

B 의 예: p_feet_W 가 한쪽에서는 발 구(球)의 중심이고 다른 쪽에서는 접촉점이면,
"발끝을 지면에" 라는 명령이 "발 중심을 지면에(= 2.5cm 관통)" 가 된다. 이건
강인성 시험이 아니라 질문 자체가 성립하지 않는 상태다. B 를 먼저 없애야 A 의
결과를 믿을 수 있다 — 안 그러면 미끄러짐을 봤을 때 그게 마찰 근사 탓인지
관통 탓인지 구별할 방법이 없다.

이 모듈이 하는 일
─────────────────
1. 의도한 불일치(A)는 **크기를 계산해 보고**한다. 실험 결과를 해석하려면
   불확실성의 크기를 알아야 한다.
2. 의도하지 않은 불일치(B, 그리고 설정 표류)는 **즉시 예외**로 막는다.

의도를 주석이 아니라 실행 시 검증으로 남기는 것이 핵심이다. 나중에 XML 을
손대서 힙 위치가 바뀌었을 때 "이것도 의도한 불확실성인가?" 를 고민하지 않아도
된다 — 선언 목록에 없으면 터진다.

구조: 순수 로직(audit)과 엔진 경계(extract_model_facts)를 분리한다.
MuJoCo 없이도 이 모듈을 import 하고 테스트할 수 있어야, 실패했을 때 원인이
물리 엔진인지 판정 로직인지 헷갈리지 않는다. angles.py 와 같은 원칙이다.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from quadruped_mpc import config as cfg

LEG_NAMES = ("FR", "FL", "RR", "RL")

#: 힙 배치가 이보다 어긋나면 규약이 깨진 것으로 본다 [m].
HIP_POSITION_TOL_M = 1e-3


class UnexpectedModelMismatch(RuntimeError):
    """선언되지 않은 불일치. 실험을 시작하기 전에 반드시 해결해야 한다."""


@dataclass(frozen=True)
class ModelFacts:
    """MuJoCo 모델에서 읽어낸 사실들. 엔진 타입에 의존하지 않는 순수 값."""

    total_mass_kg: float
    trunk_mass_kg: float
    inertia_about_com_diag: tuple[float, float, float]  # 평행축 정리 추정치
    hip_positions_B: np.ndarray          # (3, 4) trunk 기준, FR FL RR RL 순
    foot_sphere_radius_m: float
    floor_friction_mu: float
    cone_type: str                       # "pyramidal" | "elliptic"
    torque_limit_nm: float
    joint_damping: float
    joint_frictionloss: float
    n_actuators: int
    foot_solimp_width_m: float = 0.0   # 접촉이 완전 강체가 되는 침투 깊이
    n_foot_collision_geoms: int = 1    # 발 하나당 지면에 닿는 충돌 geom 수


@dataclass(frozen=True)
class Mismatch:
    name: str
    expected: object
    actual: object
    rel_delta: float | None      # 상대 오차. 비수치 항목은 None
    intentional: bool
    note: str


def pyramid_effective_mu(mu: float) -> float:
    """사각(피라미드) 제약이 실제로 허용하는 최대 유효 마찰계수.

    MPC 제약은 |fx| <= mu*fz, |fy| <= mu*fz 다. 이는 반지름 mu*fz 인 원에
    **외접**하는 정사각형이므로, 대각선 방향으로는 ||f_xy|| 이 sqrt(2)*mu*fz
    까지 허용된다. MuJoCo 의 elliptic cone 은 그 원 자체다.

    따라서 대각선 방향 힘에서만 미끄러진다 — 축 방향으로는 절대 안 미끄러진다.
    "왜 특정 방향으로만 미끄러지는가" 의 답이 이 한 줄에 있다.
    """
    return float(np.sqrt(2.0) * mu)


def _rel(expected: float, actual: float) -> float:
    return float("inf") if expected == 0 else (actual - expected) / expected


def composite_inertia_diag(
    masses: Sequence[float] | NDArray[np.float64],
    positions: Sequence[Sequence[float]] | NDArray[np.float64],
    principal_inertias: Sequence[Sequence[float]] | NDArray[np.float64],
    orientations: Sequence[NDArray[np.float64]] | NDArray[np.float64],
    ref_point: Sequence[float] | NDArray[np.float64],
) -> NDArray[np.float64]:
    """여러 강체의 합성 관성 텐서 대각항을 ref_point 기준으로 계산한다.

    ★ 이 함수가 따로 있는 이유 (내 버그의 흔적)
        처음에는 이 계산을 extract_model_facts 안에 인라인으로 넣고, 각 바디의
        관성을 **주축 프레임 그대로** 더했다. 그 결과 mit_cheetah3 에서
        Ixx +481.8%, Izz -60.3% 라는 물리적으로 불가능한 값이 나왔다
        (합성 Izz 0.833 이 몸통 자체의 2.1 보다 작다 - 양수 항들의 합이
         한 항보다 작을 수는 없다).

        원인: MuJoCo 는 XML 의 fullinertia 를 대각화해 body_inertia(주 모멘트)와
        body_iquat(주축 방향)으로 나눠 저장한다. 이때 축 순서가 뒤바뀔 수 있다.
        주 모멘트만 더하면 축이 섞인 값을 더하는 셈이다.

        올바른 식: I_world = R diag(I_principal) R^T 로 먼저 돌린 뒤 더한다.

        교훈이 둘이다.
          1) 엔진이 주는 값의 '프레임'을 확인하지 않으면 조용히 틀린다.
             observe() 에서 free joint 의 qvel 규약을 외우지 않으려고
             mj_objectVelocity 를 쓴 것과 같은 종류의 함정이다.
          2) 감사 도구도 감사받아야 한다. 이 버그는 두 독립 계산(XML 손계산 vs
             MuJoCo 추출)이 어긋났기 때문에 잡혔다. 하나만 있었으면 481% 를
             '다리 관성 기여'로 납득했을 것이다.

    Args:
        masses: (n,) 질량 [kg]
        positions: (n,3) 각 바디 무게중심의 월드 좌표 [m]
        principal_inertias: (n,3) 각 바디의 주 모멘트 [kg m^2]
        orientations: (n,3,3) 각 바디 관성 주축 프레임의 월드 자세
        ref_point: (3,) 기준점 (보통 전체 CoM)

    Returns:
        (3,) 합성 관성 텐서의 대각항 [Ixx, Iyy, Izz]
    """
    masses = np.asarray(masses, dtype=float)
    positions = np.asarray(positions, dtype=float).reshape(-1, 3)
    principal_inertias = np.asarray(principal_inertias, dtype=float).reshape(-1, 3)
    orientations = np.asarray(orientations, dtype=float).reshape(-1, 3, 3)
    ref = np.asarray(ref_point, dtype=float).reshape(3)

    total = np.zeros((3, 3))
    for m_i, p_i, I_i, R_i in zip(masses, positions, principal_inertias, orientations):
        if m_i == 0.0:
            continue
        total += R_i @ np.diag(I_i) @ R_i.T             # 주축 -> 월드
        d = p_i - ref
        total += m_i * (np.dot(d, d) * np.eye(3) - np.outer(d, d))   # 평행축
    return np.diag(total).copy()


def audit(facts: ModelFacts, *, strict: bool = True) -> list[Mismatch]:
    """모델 사실과 config.py 를 대조한다.

    Args:
        facts: extract_model_facts() 가 만든 값, 또는 테스트용 수동 구성.
        strict: True 면 의도하지 않은 불일치에서 UnexpectedModelMismatch.

    Returns:
        모든 불일치. 의도한 것은 intentional=True 로 표시되어 함께 돌아온다
        (보고용). 일치하는 항목은 목록에 없다.
    """
    out: list[Mismatch] = []

    # ── A. 의도된 모델 불확실성 — 크기를 잰다 ──────────────────────────
    if not np.isclose(facts.total_mass_kg, cfg.m):
        out.append(Mismatch(
            "질량", cfg.m, facts.total_mass_kg,
            _rel(cfg.m, facts.total_mass_kg), True,
            "MPC 는 논문 값(몸통)만 안다. 다리 질량만큼 중력 보상이 모자란다.",
        ))

    for axis, expected, actual in zip(
        ("Ixx", "Iyy", "Izz"),
        (cfg.Ixx, cfg.Iyy, cfg.Izz),
        facts.inertia_about_com_diag,
    ):
        if not np.isclose(expected, actual, rtol=0.02):
            out.append(Mismatch(
                axis, expected, actual, _rel(expected, actual), True,
                "MPC 는 몸통 관성만 쓴다. 다리의 평행축 기여가 빠져 있다.",
            ))

    if facts.cone_type != "pyramidal":
        out.append(Mismatch(
            "마찰 원뿔 형상", "pyramidal (MPC 가정)", facts.cone_type, None, True,
            f"MPC 는 대각선 방향으로 유효 mu={pyramid_effective_mu(facts.floor_friction_mu):.3f} "
            f"까지 명령할 수 있으나 엔진은 {facts.floor_friction_mu:.3f} 에서 미끄러뜨린다.",
        ))

    if facts.foot_solimp_width_m > 1e-4:
        out.append(Mismatch(
            "접촉 강성", "강체 (SRBD)",
            f"soft, solimp width {facts.foot_solimp_width_m*1000:.0f} mm",
            None, True,
            "SRBD 는 발 위치를 운동학적으로 강제한다(무한 강성). MuJoCo 는 "
            "체중에서 발이 ~11mm 가라앉는다. 지면 높이 z=0 이라는 기하학적 "
            "가정과 실제 접촉면이 그만큼 어긋난다.",
        ))

    if facts.n_foot_collision_geoms > 1:
        out.append(Mismatch(
            "발 접촉 geom 수", 1, facts.n_foot_collision_geoms, None, True,
            "발 구(r=25mm)와 종아리 캡슐 아래 반구(r=15mm)가 같은 점을 중심으로 "
            "겹쳐 있다. 발이 10mm 넘게 잠기면 캡슐도 닿아 하중이 둘로 나뉜다. "
            "두 접촉점의 x,y 가 같으므로 토크암은 어긋나지 않지만, 유효 접촉 "
            "강성이 침투 깊이에 따라 계단식으로 커진다.",
        ))

    if facts.joint_damping > 0 or facts.joint_frictionloss > 0:
        out.append(Mismatch(
            "관절 마찰", "없음 (SRBD)",
            f"damping={facts.joint_damping}, frictionloss={facts.joint_frictionloss}",
            None, True, "SRBD 에는 존재하지 않는 소산 항.",
        ))

    # ── B. 규약 불일치 / 설정 표류 — 즉시 막는다 ────────────────────────
    if facts.n_actuators != 12:
        out.append(Mismatch(
            "액추에이터 수", 12, facts.n_actuators, None, False,
            "3 관절 x 4 다리 전제가 깨졌다.",
        ))

    hip_err = np.abs(facts.hip_positions_B - cfg.hip_location_bf).max()
    if hip_err > HIP_POSITION_TOL_M:
        out.append(Mismatch(
            "힙 배치", "cfg.hip_location_bf", f"최대 오차 {hip_err:.4f} m",
            None, False,
            "Raibert 발판 계획과 토크암이 전부 이 배치를 전제한다. "
            "다리 순서(FR FL RR RL)가 뒤바뀐 경우도 여기서 걸린다.",
        ))

    # mu 값 자체는 일치해야 한다. 의도한 차이는 '원뿔의 모양'이지 '마찰계수'가
    # 아니다. 여기가 흐려지면 미끄러짐의 원인을 근사 탓인지 설정 탓인지
    # 구별할 수 없게 된다.
    if not np.isclose(facts.floor_friction_mu, cfg.MU_FRICTION, rtol=1e-6):
        out.append(Mismatch(
            "마찰계수 mu", cfg.MU_FRICTION, facts.floor_friction_mu,
            _rel(cfg.MU_FRICTION, facts.floor_friction_mu), False,
            "원뿔 '모양'의 차이는 의도했지만 mu 값의 차이는 의도하지 않았다.",
        ))

    if not np.isclose(facts.torque_limit_nm, cfg.TAU_MAX, rtol=1e-6):
        out.append(Mismatch(
            "토크 한계", cfg.TAU_MAX, facts.torque_limit_nm,
            _rel(cfg.TAU_MAX, facts.torque_limit_nm), False,
            "엔진이 MPC 제약보다 관대하면 실기에서 불가능한 해로 걷게 된다.",
        ))

    if facts.foot_sphere_radius_m <= 0:
        out.append(Mismatch(
            "발 반지름", "> 0", facts.foot_sphere_radius_m, None, False,
            "접촉점 = site - [0,0,r] 규약을 세울 수 없다.",
        ))

    if strict:
        bad = [m for m in out if not m.intentional]
        if bad:
            raise UnexpectedModelMismatch(
                "선언되지 않은 모델 불일치:\n" + format_report(bad)
            )
    return out


def format_report(mismatches: list[Mismatch]) -> str:
    """사람이 읽는 보고서. 실행할 때마다 로그에 남긴다."""
    if not mismatches:
        return "[model_audit] 불일치 없음"

    def fmt(v: object) -> str:
        return f"{v:.4g}" if isinstance(v, (int, float)) and not isinstance(v, bool) else str(v)

    lines = ["[model_audit] 모델 대조"]
    for m in mismatches:
        tag = "의도됨" if m.intentional else "⚠️ 미선언"
        delta = "" if m.rel_delta is None else f"  (Δ {m.rel_delta:+.1%})"
        lines.append(f"  [{tag}] {m.name}: MPC {fmt(m.expected)} / 모델 {fmt(m.actual)}{delta}")
        lines.append(f"           {m.note}")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────
# 엔진 경계 — 이 아래에서만 mujoco 를 만진다
# ─────────────────────────────────────────────────────────────────────
def extract_model_facts(model: Any, data: Any) -> ModelFacts:
    """MuJoCo 모델/데이터에서 ModelFacts 를 읽는다.

    data 는 nominal 자세로 mj_forward 가 끝난 상태여야 한다. 관성 추정이
    현재 자세에 의존하기 때문이다 (다리를 접으면 관성이 달라진다).

    관성은 composite_inertia_diag 로 정확히 계산한다 (주축 회전 + 평행축).
    각 바디의 관성 주축 방향(data.ximat)을 반드시 반영해야 한다 - 무시하면
    MuJoCo 가 fullinertia 를 대각화하며 뒤바꾼 축 순서 때문에 조용히 틀린다.
    """
    import mujoco  # noqa: PLC0415  — 경계 안에서만 필요하다

    mujoco.mj_forward(model, data)

    com = data.subtree_com[0].copy()
    ids = [b for b in range(1, model.nbody) if model.body_mass[b] != 0.0]
    inertia = composite_inertia_diag(
        masses=[model.body_mass[b] for b in ids],
        positions=[data.xipos[b] for b in ids],
        principal_inertias=[model.body_inertia[b] for b in ids],
        # data.ximat = 관성 주축 프레임의 월드 자세. model.body_inertia 는
        # **그 프레임 기준** 주 모멘트이므로, 돌려서 더해야 한다.
        orientations=[np.asarray(data.ximat[b]).reshape(3, 3) for b in ids],
        ref_point=com,
    )

    trunk_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk")
    hips = np.zeros((3, 4))
    for i, leg in enumerate(LEG_NAMES):
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{leg}_hip")
        if bid < 0:
            raise UnexpectedModelMismatch(f"{leg}_hip body 를 찾을 수 없다.")
        hips[:, i] = model.body_pos[bid]

    floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    mu = float(model.geom_friction[floor_id][0]) if floor_id >= 0 else float("nan")

    # 발 구: calf 바디에 달린 sphere geom 중 가장 아래 있는 것
    radius = 0.0
    calf_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "FR_calf")
    for gid in range(model.ngeom):
        if model.geom_bodyid[gid] == calf_id and model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_SPHERE:
            radius = float(model.geom_size[gid][0])

    # 발 지점에 닿을 수 있는 충돌 geom 수와 접촉 소프트니스.
    # 둘 다 SRBD 의 '강체 + 점 접촉' 이상화와 다른 지점이다.
    solimp_width = 0.0
    n_foot_geoms = 0
    for gid in range(model.ngeom):
        if model.geom_bodyid[gid] != calf_id or model.geom_contype[gid] == 0:
            continue
        n_foot_geoms += 1
        if model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_SPHERE:
            solimp_width = float(model.geom_solimp[gid][2])

    cone = "elliptic" if model.opt.cone == mujoco.mjtCone.mjCONE_ELLIPTIC else "pyramidal"
    frange = np.abs(model.actuator_forcerange).max() if model.nu else 0.0

    return ModelFacts(
        total_mass_kg=float(model.body_mass.sum()),
        trunk_mass_kg=float(model.body_mass[trunk_id]),
        inertia_about_com_diag=(float(inertia[0]), float(inertia[1]), float(inertia[2])),
        hip_positions_B=hips,
        foot_sphere_radius_m=radius,
        floor_friction_mu=mu,
        cone_type=cone,
        torque_limit_nm=float(frange),
        joint_damping=float(np.max(model.dof_damping[6:])) if model.nv > 6 else 0.0,
        joint_frictionloss=float(np.max(model.dof_frictionloss[6:])) if model.nv > 6 else 0.0,
        n_actuators=int(model.nu),
        foot_solimp_width_m=solimp_width,
        n_foot_collision_geoms=n_foot_geoms,
    )
