"""회전 표현과 좌표 변환.

ZYX 오일러각 규약: R_W_B = R_z(yaw) @ R_y(pitch) @ R_x(roll)

이 모듈이 따로 존재하는 이유
    결함 1(오일러각 변환 전치)은 MPC 의 자세 운동학과 플랜트의 자세 운동학이
    서로 다른 파일에 흩어져 있어서, 두 식이 전치 관계로 어긋나 있다는 것을
    아무도 볼 수 없었기 때문에 발생했다. 같은 물리를 두 곳에서 각자 구현하면
    반드시 갈라진다. 여기에 모아두고, tests/test_rotations.py 가 MPC 와
    플랜트가 같은 식을 쓰는지 직접 검증한다.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

# dynamics.py 의 원래 표현을 그대로 쓴다 (수치 재현성 유지)
_DEG2RAD = np.pi / 180
_PITCH_LIMIT = 89 * _DEG2RAD


def Rx(roll: float) -> NDArray[np.float64]:
    """x 축 회전 (roll)."""
    c, s = np.cos(roll), np.sin(roll)
    return np.array([[1, 0, 0],
                     [0, c, -s],
                     [0, s,  c]])


def Ry(pitch: float) -> NDArray[np.float64]:
    """y 축 회전 (pitch)."""
    c, s = np.cos(pitch), np.sin(pitch)
    return np.array([[ c, 0, s],
                     [ 0, 1, 0],
                     [-s, 0, c]])


def Rz(yaw: float) -> NDArray[np.float64]:
    """z 축 회전 (yaw)."""
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0],
                     [s,  c, 0],
                     [0,  0, 1]])


def rpy_to_matrix(roll: float, pitch: float, yaw: float) -> NDArray[np.float64]:
    """ZYX 오일러각 -> 회전 행렬 R_W_B (body -> world).

    dynamics.get_rotation_matrix 를 그대로 옮긴 것이며 수치적으로 동일하다.
    """
    return np.array([[np.cos(yaw), -np.sin(yaw), 0],
                     [np.sin(yaw),  np.cos(yaw), 0],
                     [0, 0, 1]]) @ \
           np.array([[np.cos(pitch), 0, np.sin(pitch)],
                     [0, 1, 0],
                     [-np.sin(pitch), 0, np.cos(pitch)]]) @ \
           np.array([[1, 0, 0],
                     [0, np.cos(roll), -np.sin(roll)],
                     [0, np.sin(roll), np.cos(roll)]])


def omega_to_rpy_rate(pitch: float, yaw: float) -> NDArray[np.float64]:
    """world frame 각속도 -> ZYX 오일러각 변화율 변환 행렬 T.

        Θ̇ = T(theta, psi) · omega_W

    ★ pitch = 0 일 때 T = R_z(psi)^T 이며, 이것이 convex MPC 의
      Ac[0:3, 6:9] 와 정확히 같아야 한다.

      결함 1 은 MPC 가 R_z(psi) 를 (전치 없이) 써서 이 둘이 전치 관계로
      어긋나 있던 것이다. psi=0 에서는 두 행렬이 동일하므로 직진 보행으로는
      절대 잡히지 않았고, psi=90° 에서 대각 성분이 0 이 되어 roll/pitch
      피드백의 부호가 전체 반전되며 지수 발산했다.

      tests/test_rotations.py::test_mpc_and_plant_attitude_kinematics_agree
      가 이 일치를 psi != 0 에서 검증한다.

    짐벌락 방어: pitch 를 ±89° 로 클립한다. 4족보행에서 이 클립이 발동하는
    상황은 표현의 문제가 아니라 이미 전복이다.
    """
    pitch_safe = np.clip(pitch, -_PITCH_LIMIT, _PITCH_LIMIT)
    return np.array([
        [np.cos(yaw) / np.cos(pitch_safe), np.sin(yaw) / np.cos(pitch_safe), 0],
        [-np.sin(yaw),                     np.cos(yaw),                      0],
        [np.cos(yaw) * np.tan(pitch_safe), np.sin(yaw) * np.tan(pitch_safe), 1],
    ])
