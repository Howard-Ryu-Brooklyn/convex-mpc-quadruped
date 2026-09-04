"""회전 변환 검증.

이 파일의 존재 이유는 test_mpc_and_plant_attitude_kinematics_agree 하나다.
결함 1 을 찾는 데 나흘이 걸렸는데, 이 테스트가 있었다면 즉시 잡혔다.
"""
import numpy as np
import numpy.testing as npt
import pytest

import config as cfg
from convex_mpc import ConvexMPC
from rotations import Rx, Ry, Rz, omega_to_rpy_rate, rpy_to_matrix

ANGLES = [0.0, 0.2, np.pi / 6, np.pi / 4, np.pi / 3, np.pi / 2 - 0.05, 2.5, -1.1]


# ── 회전 행렬의 기본 성질 ────────────────────────────────────────────────
@pytest.mark.parametrize("fn", [Rx, Ry, Rz])
@pytest.mark.parametrize("angle", ANGLES)
def test_rotation_is_orthonormal(fn, angle):
    """회전 행렬은 직교이고 행렬식이 +1 이다 (반사가 아니라 회전)."""
    R = fn(angle)
    npt.assert_allclose(R @ R.T, np.eye(3), atol=1e-15)
    npt.assert_allclose(np.linalg.det(R), 1.0, atol=1e-15)


@pytest.mark.parametrize("angle", ANGLES)
def test_rpy_to_matrix_is_zyx_composition(angle):
    """rpy_to_matrix 가 R_z @ R_y @ R_x 순서인가 (ZYX 규약)."""
    r, p, y = 0.13, -0.27, angle
    npt.assert_allclose(rpy_to_matrix(r, p, y), Rz(y) @ Ry(p) @ Rx(r), atol=1e-15)


def test_rpy_to_matrix_identity_at_zero():
    npt.assert_allclose(rpy_to_matrix(0.0, 0.0, 0.0), np.eye(3), atol=1e-15)


# ── ★ 모델-플랜트 일치 (결함 1 의 회귀 감시) ─────────────────────────────
@pytest.mark.parametrize("yaw", ANGLES)
def test_omega_to_rpy_rate_is_Rz_transpose_at_zero_pitch(yaw):
    """pitch = 0 에서 T(0, psi) = R_z(psi)^T 인가.

    이것이 convex MPC 의 Ac[0:3, 6:9] 가 만족해야 하는 항등식이다.
    """
    npt.assert_allclose(omega_to_rpy_rate(0.0, yaw), Rz(yaw).T, atol=1e-15)


@pytest.mark.parametrize("yaw", ANGLES)
def test_mpc_and_plant_attitude_kinematics_agree(yaw):
    """MPC 모델의 자세 운동학이 플랜트와 같은 식을 쓰는가.

    결함 1: MPC 가 Ac[0:3,6:9] = R_z(psi) 로, 플랜트가 R_z(psi)^T 로
    구현되어 있었다. 두 행렬은 대각 성분이 같고 비대각의 부호만 다르므로
    psi=0 에서 완전히 일치한다. 그래서 직진 보행(S1)으로는 절대 잡히지
    않았고, 발견에 나흘이 걸렸다.

    이 테스트는 psi != 0 에서 둘을 직접 비교한다. 누가 .T 를 지우는 순간
    즉시 빨간불이 뜬다.
    """
    dt = 1.0 / 30.0
    mpc = ConvexMPC(
        m=cfg.m,
        I_body=[cfg.Ixx, cfg.Iyy, cfg.Izz],
        gz=cfg.gz,
        dt=dt,
        horizon=1,
        Lweights=np.zeros(13),
        Kweights=cfg.K_w_f,
        mu=cfg.MU_FRICTION,
        fmin=cfg.fmin,
        fmax=cfg.fmax,
    )
    Ad, _ = mpc.get_discrete_matrices(yaw, np.zeros((3, 4)))

    # Ad = I + Ac*dt 이고 단위행렬의 [0:3, 6:9] 블록은 0 이므로
    mpc_block = Ad[0:3, 6:9] / dt
    plant_block = omega_to_rpy_rate(pitch=0.0, yaw=yaw)

    npt.assert_allclose(
        mpc_block, plant_block, atol=1e-12,
        err_msg=(
            f"yaw={yaw:.3f} 에서 MPC 모델과 플랜트의 자세 운동학이 다르다. "
            "convex_mpc.get_discrete_matrices 의 Ac[0:3,6:9] 가 "
            "Rz.T 인지 확인할 것 (결함 1)."
        ),
    )


def test_why_straight_walking_could_not_catch_defect_1():
    """결함 1 이 왜 직진 보행으로 잡히지 않았는지를 코드로 남긴다.

    R_z 와 R_z^T 는 대각 성분(cos psi)이 같고 비대각(sin psi)의 부호만 다르다.
    따라서 오차의 크기는 |sin psi| 에 비례하고, psi=0 에서 정확히 0 이다.
    회귀 시나리오를 고를 때 'psi != 0 인 것이 반드시 있어야 한다'는 근거다.
    """
    npt.assert_allclose(Rz(0.0), Rz(0.0).T, atol=1e-15)          # psi=0: 구분 불가

    for yaw in [0.1, 0.5, np.pi / 4]:
        diff = np.max(np.abs(Rz(yaw) - Rz(yaw).T))
        npt.assert_allclose(diff, 2 * abs(np.sin(yaw)), atol=1e-12)

    # psi=90도에서 대각 성분이 사라진다 = 피드백 부호가 완전히 반전되는 지점
    npt.assert_allclose(np.diag(Rz(np.pi / 2))[:2], [0.0, 0.0], atol=1e-15)
