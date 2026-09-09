"""회전 변환 검증.

이 파일의 존재 이유는 test_mpc_and_plant_attitude_kinematics_agree 하나다.
결함 1 을 찾는 데 나흘이 걸렸는데, 이 테스트가 있었다면 즉시 잡혔다.
"""
import numpy as np
import numpy.testing as npt
import pytest

from quadruped_mpc import config as cfg
from quadruped_mpc.control.convex_mpc import ConvexMPC
from quadruped_mpc.core.rotations import Rx, Ry, Rz, matrix_to_rpy, omega_to_rpy_rate, rpy_to_matrix

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


# ── matrix_to_rpy: MuJoCo 자세 -> MPC 오일러각 경계 ─────────────────────
@pytest.mark.parametrize("seed", range(6))
def test_matrix_to_rpy_inverts_rpy_to_matrix(seed):
    """무작위 자세 500개에 대해 rpy -> R -> rpy 왕복.

    pitch 는 (-90, 90) 안에서만 뽑는다. 그 밖은 ZYX 표현 자체가 중복이라
    '원래 각도로 돌아오는가'를 물을 수 없다 - 아래 짐벌락 테스트가 다룬다.
    """
    rng = np.random.default_rng(seed)
    for _ in range(500):
        r = rng.uniform(-np.pi, np.pi)
        p = rng.uniform(-np.pi / 2 + 1e-3, np.pi / 2 - 1e-3)
        y = rng.uniform(-np.pi, np.pi)
        npt.assert_allclose(matrix_to_rpy(rpy_to_matrix(r, p, y)), [r, p, y], atol=1e-9)


@pytest.mark.parametrize("pitch", [np.pi / 2, -np.pi / 2])
@pytest.mark.parametrize("roll", [0.0, 0.7, -1.2])
@pytest.mark.parametrize("yaw", [0.0, 1.1, -2.0])
def test_gimbal_lock_preserves_the_rotation_even_when_angles_are_ambiguous(pitch, roll, yaw):
    """짐벌락에서는 각도가 유일하지 않다. 되돌린 '회전'이 같으면 충분하다.

    각도를 요구하면 실패하지만 회전을 요구하면 통과한다 - 무엇을 단언할지가
    곧 무엇을 보장하는지다. pitch 90도는 4족보행에서 이미 넘어진 상태라
    제어 목적으로는 무의미한 영역이지만, 조용히 NaN 을 내놓지 않는다는 것은
    보장되어야 한다.
    """
    R = rpy_to_matrix(roll, pitch, yaw)
    got = matrix_to_rpy(R)
    assert np.all(np.isfinite(got))
    npt.assert_allclose(rpy_to_matrix(*got), R, atol=1e-9)


def test_identity_matrix_maps_to_zero_angles():
    npt.assert_allclose(matrix_to_rpy(np.eye(3)), np.zeros(3), atol=1e-15)


@pytest.mark.parametrize("yaw", [0.5, -0.5, 3.0, -3.0])
def test_pure_yaw_is_read_back_as_pure_yaw(yaw):
    """MuJoCo 로 넘어가는 주된 경로 - 평지 보행은 대부분 이 상태다."""
    npt.assert_allclose(matrix_to_rpy(Rz(yaw)), [0.0, 0.0, yaw], atol=1e-12)


def test_matrix_to_rpy_returns_wrapped_yaw_so_unwrapping_is_still_required():
    """이 함수는 접힌 yaw 를 준다. 연속화 책임이 Plant 에 있음을 못박는다.

    여기서 알아서 펴주면 '누가 그 상태의 주인인가'가 흐려진다. 상태를 가진
    쪽(AngleUnwrapper)이 하나여야 멱등성과 초기값을 통제할 수 있다.
    """
    yaw = np.deg2rad(200.0)                      # +200도 = -160도로 접힌다
    got = matrix_to_rpy(Rz(yaw))[2]
    assert -np.pi <= got < np.pi
    npt.assert_allclose(got, np.deg2rad(-160.0), atol=1e-12)
