"""MuJoCo 물리 엔진을 PlantBase 로 감싼다.

이 파일이 존재하는 이유
    mujoco_main_ref.py 는 제어 로직 전체를 복사한 630 줄이었고, 결함 1~11 이
    하나도 반영되어 있지 않았다. 같은 Controller 가 두 Plant 에 꽂히면 그
    복제본이 사라진다.

SRB 와 MuJoCo 는 다른 질문에 답한다 (model_audit.py 참조)
    SRB    : MPC 알고리즘 자체가 옳은가?  -> 모델 오차 0
    MuJoCo : 선형화 가정이 실제 조건에서 얼마나 강인한가? -> 모델 오차가 있어야 한다

    따라서 질량(43 vs 45.84), 관성, 마찰 원뿔 모양의 차이는 **의도적으로**
    남긴다. 반대로 좌표계와 규약의 차이는 남기면 안 된다. 값이 다른 것은
    제어기가 극복할 대상이지만 의미가 다른 것은 극복할 수 없기 때문이다.

이 파일에서 반드시 없애야 하는 규약 차이 세 가지
    1. CoM.        data.qpos[0:3] 은 trunk **원점**이지 CoM 이 아니다. 이 모델에서
                   둘은 (-7.7, 0, -12.2) mm 떨어져 있고, 그대로 쓰면 토크암 r 에
                   상수 오차가 실린다. subtree_com 을 쓴다.
    2. 접촉점.     발 site 는 발 구(球)의 **중심**이다. 접촉점은 그보다
                   foot_radius(25mm) 아래다. 이 차이를 무시하면 "발끝을 지면에"
                   라는 명령이 "발 중심을 지면에(= 25mm 관통)" 가 된다.
    3. yaw.        오일러각은 (-pi, pi] 로 접혀 나오는데 MPC 는 누적 연속량으로
                   쓴다. AngleUnwrapper 를 통과시킨다 (angles.py 참조).

    mujoco_main_ref.py 는 셋 다 그냥 두고 있었다.
"""

from __future__ import annotations

import numpy as np

from angles import AngleUnwrapper
from model_audit import audit, extract_model_facts, format_report
from plant_base import PlantBase
from robot_types import ControlCommand, RobotState
from rotations import matrix_to_rpy

LEG_NAMES = ("FR", "FL", "RR", "RL")
FOOT_SITE_NAMES = tuple(f"{n}_site" for n in LEG_NAMES)


class MuJoCoPlant(PlantBase):
    """MuJoCo 기반 Plant.

    Attributes 는 전부 private 이다. 바깥에서 보는 창은 observe() 하나이며,
    SRBDynamics 처럼 살아있는 참조를 노출하는 레거시 프로퍼티를 만들지 않는다
    (결함 2, 9 가 그 경로로 들어왔다).
    """

    def __init__(self, xml_path: str, dt: float, home_qpos=None,
                 audit_strict: bool = True, verbose: bool = True,
                 swing_omega_n: float = 240.0, swing_zeta: float = 1.0) -> None:
        import mujoco  # noqa: PLC0415 — 엔진 경계 안에서만 필요하다

        self._mj = mujoco
        self._model = mujoco.MjModel.from_xml_path(str(xml_path))
        self._data = mujoco.MjData(self._model)
        self._model.opt.timestep = dt
        self._dt = float(dt)

        if home_qpos is not None:
            self._data.qpos[:] = np.asarray(home_qpos, dtype=float)

        # ── 모델 대조를 '실행 시점'에 한다 ────────────────────────────
        # 의도한 불확실성은 크기와 함께 보고하고, 선언되지 않은 불일치
        # (다리 순서, 힙 배치, mu, 토크 한계)는 여기서 예외로 막는다.
        # 실험을 시작하기 전에 우리 쪽 결함을 없애 두어야, 나중에 나오는
        # 실패를 '강인성 한계'로 해석할 수 있다.
        self._facts = extract_model_facts(self._model, self._data)
        mismatches = audit(self._facts, strict=audit_strict)
        if verbose:
            print(format_report(mismatches))

        self._trunk_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, "trunk")
        self._site_ids = [
            mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_SITE, n)
            for n in FOOT_SITE_NAMES
        ]
        if min(self._site_ids) < 0:
            raise ValueError(f"발 site 를 찾을 수 없다: {FOOT_SITE_NAMES}")

        # ── dof / actuator 인덱스는 산술이 아니라 이름으로 만든다 ──────
        # "free joint 6개 뒤에 다리마다 3개" 라는 산술은 XML 이 조금만 바뀌어도
        # 조용히 틀린 관절에 토크를 넣는다. 그리고 그 증상은 '로봇이 이상하게
        # 걷는다' 뿐이라 로그로 잡을 수 없다. 이름으로 물어보면 틀리면 즉시
        # 터진다 - 결함 11 에서 다리 순서를 audit 으로 막은 것과 같은 이유다.
        self._dof_idx = np.zeros((4, 3), dtype=int)      # [leg][abad, hip, knee]
        self._act_idx = np.zeros((4, 3), dtype=int)
        for i, leg in enumerate(LEG_NAMES):
            for j, joint in enumerate(("hip", "thigh", "calf")):
                jid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT,
                                        f"{leg}_{joint}_joint")
                aid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_ACTUATOR,
                                        f"{leg}_{joint}")
                if jid < 0 or aid < 0:
                    raise ValueError(f"{leg}_{joint} 관절/액추에이터를 찾을 수 없다")
                self._dof_idx[i, j] = self._model.jnt_dofadr[jid]
                self._act_idx[i, j] = aid

        # 스윙 다리 작업공간 제어 게인.
        # Kp = omega_n^2 * Lambda, Kd = 2 zeta omega_n * Lambda 로 두면 닫힌
        # 루프가 관성과 무관하게 (omega_n, zeta) 인 2차계가 된다. 게인을 직접
        # 쓰지 않고 원하는 응답 특성으로 쓰는 것이 이 형태의 이점이다.
        #
        # 기본값 240 은 측정으로 정했다 (scripts/diag_mujoco_step.py 대역폭 스윕).
        # 150ms 스윙으로 발을 15cm 옮길 때 착지 오차:
        #     omega_n=  60 : 17.74 mm,  39.9 Nm   <- mujoco_main_ref.py 의 값
        #     omega_n= 120 :  4.97 mm,  68.8 Nm
        #     omega_n= 240 :  1.02 mm, 140.1 Nm   <- 채택
        #     omega_n= 480 :  0.22 mm, 285.4 Nm   <- 토크 한계 250 Nm 초과
        # 오차가 omega_n^-2 로 줄어든다. 이는 잔차가 '구조적으로 추종 불가능한
        # 항'이 아니라 '강성으로 눌리는 외란 힘'(주로 빠진 J_dot*q_dot)이라는
        # 뜻이다. 바닥이 없으므로 게인과 토크의 교환일 뿐이다.
        #
        # 60 을 쓰면 착지점이 18mm 어긋나고, 그 오차는 MPC 의 토크암 r 에
        # 그대로 실린다(다리 길이 ~300mm 대비 6%). 복사해 온 상수를 그대로
        # 쓰지 않고 재본 이유다.
        self._omega_n = np.full(3, swing_omega_n)
        self._zeta = np.full(3, swing_zeta)

        self._align_to_ground()

        # yaw 연속화는 Plant 가 소유한다. observe() 안에서만 갱신되며,
        # 같은 관측을 두 번 넣어도 값이 움직이지 않는다(멱등).
        mujoco.mj_forward(self._model, self._data)
        rpy0 = matrix_to_rpy(self._data.xmat[self._trunk_id])
        self._yaw = AngleUnwrapper(initial_angle_rad=float(rpy0[2]))
        self._max_abs_torque = 0.0

    # ── PlantBase 계약 ────────────────────────────────────────────────
    @property
    def dt(self) -> float:
        return self._dt

    def observe(self) -> RobotState:
        """현재 상태의 불변 스냅샷. 매 제어 스텝마다 호출되어야 한다.

        AngleUnwrapper 의 전제가 '관측 간격 사이 각도 변화 < pi' 이므로,
        여러 스텝을 건너뛰고 부르면 안 된다. 같은 스텝에 두 번 부르는 것은
        안전하다(멱등).
        """
        mj, m, d = self._mj, self._model, self._data
        mj.mj_forward(m, d)
        mj.mj_subtreeVel(m, d)          # subtree_linvel 을 채운다

        R_W_B = np.asarray(d.xmat[self._trunk_id]).reshape(3, 3)

        # 규약 차이 1 — CoM. qpos[0:3] 은 trunk 원점이다.
        p_com_W = np.asarray(d.subtree_com[self._trunk_id], dtype=float).copy()
        v_com_W = np.asarray(d.subtree_linvel[self._trunk_id], dtype=float).copy()

        # 각속도는 강체 위 어느 점에서나 같으므로 CoM 모호성이 없다.
        # 규약을 외우지 않기 위해 mj_objectVelocity(flg_local=0) 을 쓴다.
        # (free joint 의 qvel[3:6] 은 '바디 프레임' 각속도라는 규약을 기억해야
        #  하는데, 그런 종류의 지식은 코드가 아니라 사람 머리에 남는다.)
        vel6 = np.zeros(6)
        mj.mj_objectVelocity(m, d, mj.mjtObj.mjOBJ_BODY, self._trunk_id, vel6, 0)
        omega_W = vel6[0:3].copy()      # [rot(3), lin(3)] 순서

        # 규약 차이 3 — yaw 연속화.
        roll, pitch, yaw_wrapped = matrix_to_rpy(R_W_B)
        yaw = self._yaw.update(float(yaw_wrapped))

        return RobotState(
            p_com_W=p_com_W,
            v_com_W=v_com_W,
            rpy_W=np.array([roll, pitch, yaw]),
            omega_W=omega_W,
        )

    def step(self, command: ControlCommand) -> None:
        """지령을 관절 토크로 바꿔 엔진에 넣고 한 스텝 진행한다.

        SRBDynamics 가 발 위치를 운동학적으로 강제하는 자리에서, MuJoCoPlant 는
        실제 구동으로 바꾼다. 이 변환이 Plant 의 책임이라는 것이 PlantBase 를
        '제어기가 내는 것' 기준으로 정의한 이유다 (plant_base.py 참조).

            지지 다리: tau = J^T (-f_grf)
            스윙 다리: tau = J^T (Kp e + Kd e_dot) + J^T Lambda a_des + bias

        모든 계산을 월드 프레임에서 한다. mj_jacSite 와 mj_objectVelocity 가
        월드 프레임을 주므로 변환이 아예 필요 없다 - 변환이 없으면 변환을
        틀릴 수도 없다.
        """
        if command.v_feet_W is None or command.a_feet_W is None:
            raise ValueError(
                "토크 구동 Plant 는 스윙 발의 목표 속도/가속도가 필요하다. "
                "ControlCommand.v_feet_W / a_feet_W 가 None 이다. "
                "0 으로 대체하지 않는 이유는 '안 줬다'와 '0 을 지령했다'가 "
                "다른 사실이기 때문이다 (robot_types.ControlCommand 참조)."
            )

        mj, m, d = self._mj, self._model, self._data
        mj.mj_forward(m, d)

        p_feet_W = self.feet_positions_W()
        p_des_W = command.r_feet_W + self.observe().p_com_W.reshape(3, 1)

        M_full = np.zeros((m.nv, m.nv))
        mj.mj_fullM(m, d, M_full)
        bias = np.asarray(d.qfrc_bias, dtype=float)

        jacp = np.zeros((3, m.nv))
        jacr = np.zeros((3, m.nv))
        vel6 = np.zeros(6)
        tau_all = np.zeros(m.nu)

        for leg in range(4):
            mj.mj_jacSite(m, d, jacp, jacr, self._site_ids[leg])
            # 접촉점은 site 에서 월드 z 로 **상수**만큼 내린 점이다. 상수
            # 오프셋은 q 에 대한 미분에 영향이 없으므로 자코비안이 같다.
            J_full = jacp.copy()                       # (3, nv), 월드 프레임
            dofs = self._dof_idx[leg]
            J_leg = J_full[:, dofs]                    # (3,3) 이 다리 관절만

            if command.is_stance[leg]:
                # 부호: MPC 의 f 는 '지면이 발에 가하는 힘'이다. 따라서 발이
                # 지면에 가하는 힘은 -f 이고, 그것을 만드는 관절 토크가
                # tau = J^T (-f) 다. 이 부호는 논증이 아니라 측정으로 확인한다
                # (scripts/diag_mujoco_step.py 의 정지 지지 시험).
                tau = J_leg.T @ (-command.forces_W[:, leg])
            else:
                mj.mj_objectVelocity(m, d, mj.mjtObj.mjOBJ_SITE,
                                     self._site_ids[leg], vel6, 0)
                v_foot_W = vel6[3:6]                   # [rot(3), lin(3)] 순서

                # 작업공간 관성 Lambda = (J M^-1 J^T)^-1.
                # 정칙화 항은 다리가 쭉 펴져 자코비안이 특이해지는 자세를 위한
                # 것이다. 없으면 그 순간 토크가 발산한다.
                M_inv = np.linalg.inv(M_full + np.eye(m.nv) * 1e-6)
                Lambda = np.linalg.inv(J_full @ M_inv @ J_full.T + np.eye(3) * 1e-6)
                Lam_d = np.diag(Lambda)

                Kp = np.diag(self._omega_n ** 2 * Lam_d)
                Kd = np.diag(2.0 * self._zeta * self._omega_n * Lam_d)

                f_task = (Kp @ (p_des_W[:, leg] - p_feet_W[:, leg])
                          + Kd @ (command.v_feet_W[:, leg] - v_foot_W))
                # 피드포워드: 원하는 가속도를 내는 작업공간 힘.
                # TODO(결함 13 후보): J_dot * q_dot (원심/코리올리) 항이 빠져
                #   있다. 스윙이 빠를수록 커지므로 갤로핑에서 추종 오차로
                #   나타날 것이다. 수치 미분으로 넣으려면 상태(prev_J)가 필요해
                #   지므로 별도 스텝으로 다룬다.
                f_ff = Lambda @ command.a_feet_W[:, leg]
                tau = J_leg.T @ (f_task + f_ff) + bias[dofs]

            tau_all[self._act_idx[leg]] = tau

        # 토크 한계는 엔진의 forcerange 가 강제하지만, 우리가 얼마나 요구했는지는
        # 남겨 둔다. 실기 이관에서 '시뮬에서는 되던데'의 정체가 대개 이것이다.
        self._max_abs_torque = max(self._max_abs_torque,
                                   float(np.abs(tau_all).max()))
        d.ctrl[:] = tau_all
        mj.mj_step(m, d)

    # ── 관측 부가 정보 ────────────────────────────────────────────────
    def feet_positions_W(self) -> np.ndarray:
        """(3,4) 네 발의 **접촉점** 월드 좌표.

        규약 차이 2 — site 는 발 구의 중심이므로 반지름만큼 내린다.
        이 한 줄이 없으면 Raibert 의 p_feet_des[2] = 0 이 '발 중심을 지면에'
        가 되어 25mm 관통을 지령한다.
        """
        self._mj.mj_forward(self._model, self._data)
        p = np.stack([self._data.site_xpos[i] for i in self._site_ids], axis=1)
        p[2, :] -= self._facts.foot_sphere_radius_m
        return p.copy()

    def hip_positions_W(self) -> np.ndarray:
        """(3,4) 네 고관절의 월드 좌표. Raibert 발판 계획의 기준점."""
        mj, m, d = self._mj, self._model, self._data
        mj.mj_forward(m, d)
        ids = [mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, f"{n}_hip") for n in LEG_NAMES]
        return np.stack([d.xpos[i] for i in ids], axis=1).copy()

    @property
    def joint_angles(self) -> np.ndarray:
        """(3,4) 관절각 [q_abad, q_hip, q_knee] x 4. qpos[7:] 를 재배열한 것."""
        return np.asarray(self._data.qpos[7:], dtype=float).reshape(4, 3).T.copy()

    @property
    def facts(self):
        """감사에 쓰인 모델 사실들 (읽기 전용)."""
        return self._facts

    @property
    def max_yaw_step_rad(self) -> float:
        """unwrap 전제의 사후 감시값. pi 에 근접하면 관측 주기가 부족한 것이다."""
        return self._yaw.max_abs_step_rad

    @property
    def max_abs_torque_nm(self) -> float:
        """지금까지 지령한 관절 토크의 최댓값 [Nm] (진단용).

        cfg.TAU_MAX 에 근접하거나 넘으면 엔진이 잘라내고 있다는 뜻이고,
        그 경우 시뮬 결과는 실기에서 재현되지 않는다.
        """
        return self._max_abs_torque

    @property
    def time(self) -> float:
        return float(self._data.time)

    # ── 초기화 보조 ───────────────────────────────────────────────────
    def _align_to_ground(self) -> None:
        """현재 관절각 상태에서 가장 낮은 발의 접촉점이 z=0 이 되도록 띄운다.

        발 site 는 구의 중심이므로 반지름을 빼야 실제 접촉점이 된다 -
        feet_positions_W 와 같은 규약을 쓴다. 규약이 두 곳에 있으면 갈라진다.
        """
        self._mj.mj_forward(self._model, self._data)
        lowest = self.feet_positions_W()[2, :].min()
        self._data.qpos[2] -= lowest
        self._mj.mj_forward(self._model, self._data)
