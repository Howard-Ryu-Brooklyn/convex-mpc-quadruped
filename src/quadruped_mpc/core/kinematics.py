"""다리 기구학 — 역기구학, 관절 제한, 자코비안, 좌표 변환.

config.py 는 로봇 상수만 담고, 계산은 전부 여기로 모은다.
(분리 전 config.py 는 상수 + 역기구학 + 회전행렬 + 베지에의 네 가지 책임을
 한 파일에 갖고 있었다.)

관절 규약
    q1 : ab/ad  (roll)  고관절 좌우
    q2 : hip    (pitch) 고관절 전후
    q3 : knee   (pitch) 무릎. 뒤로 꺾이므로 항상 음수

TODO(Step 3): compute_leg_ik 는 도달 불가능한 목표에 대해 NaN 배열을 반환하고,
    호출부가 이를 검사하지 않아 실패가 조용히 전파된다. 예외로 바꾼다.
TODO(Step 5): 순기구학(FK)이 없어 IK/FK 왕복 테스트를 쓸 수 없다.
    지금은 코사인 법칙 일관성으로 부분 검증만 한다.
"""
from __future__ import annotations

import numpy as np

from quadruped_mpc import config as cfg
from quadruped_mpc.core.robot_types import Mat3, Vec3, Vec3x4


#: 다리별 abad 링크가 뻗는 y 방향 부호. FR/RR 은 -y, FL/RL 은 +y.
#: 하드코딩하지 않고 힙 배치에서 파생한다 - 한 사실은 한 곳에만 둔다.
#: (레그 순서는 robot_types.Leg 및 cfg.hip_location_bf 와 같다: FR FL RR RL)
LEG_HIP_SIGN = np.sign(cfg.hip_location_bf[1, :])


def leg_link_positions(
    q: Vec3,
    leg: int,
    l_hip: float = cfg.link_hip,
    l_thigh: float = cfg.link_upper,
    l_calf: float = cfg.link_lower,
) -> Vec3x4:
    """관절각 -> 링크 연결점 4개 (3,4): [고관절, abad 끝, 무릎, 발끝].

    고관절 원점 기준, 바디 프레임. 시각화가 다리를 그릴 때 쓴다.
    leg_forward_kinematics 는 이 함수의 마지막 열이다 - FK 를 두 번 쓰지 않는다.
    """
    qa, qh, qk = float(q[0]), float(q[1]), float(q[2])
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(qa), -np.sin(qa)],
                   [0, np.sin(qa), np.cos(qa)]])
    Rh = np.array([[np.cos(qh), 0, np.sin(qh)],
                   [0, 1, 0],
                   [-np.sin(qh), 0, np.cos(qh)]])
    c = qh + qk
    Rk = np.array([[np.cos(c), 0, np.sin(c)],
                   [0, 1, 0],
                   [-np.sin(c), 0, np.cos(c)]])

    p_hip = np.zeros(3)
    p_abad = Rx @ np.array([0.0, LEG_HIP_SIGN[leg] * l_hip, 0.0])
    p_knee = p_abad + Rx @ Rh @ np.array([0.0, 0.0, -l_thigh])
    p_foot = p_knee + Rx @ Rk @ np.array([0.0, 0.0, -l_calf])
    return np.stack([p_hip, p_abad, p_knee, p_foot], axis=1)


def leg_forward_kinematics(
    q: Vec3,
    leg: int,
    l_hip: float = cfg.link_hip,
    l_thigh: float = cfg.link_upper,
    l_calf: float = cfg.link_lower,
) -> Vec3:
    """관절각 -> 고관절 원점 기준 발 위치 (바디 프레임). compute_leg_ik 의 역함수.

    FK 가 여기 하나만 있어야 하는 이유
        예전에는 FK 가 visualization.animate_quadruped 안에 손으로 다시 쓰여
        있었고, 그쪽 규약이 IK 와 어긋나 그려진 발이 실제 발과 최대 22cm
        떨어져 있었다. yaw 회전 시 '디딤발이 몸통과 함께 도는' 것처럼 보인
        원인이 이것이다 - 오차가 바디 고정이라 몸통과 같이 돌았다.

        파생값을 두 곳에서 각자 계산하면 규약이 어긋나는 순간 한쪽이 조용히
        거짓말을 한다. 결함 2(r_feet_traj), 결함 5(절대/상대 위치)와 같은 계열이다.

    Args:
        q: (3,) [q_abad, q_hip, q_knee] [rad]
        leg: 다리 인덱스 0..3 (FR FL RR RL). abad 링크 방향 부호를 결정한다.
        l_hip: abad 링크 길이 (크기). 부호는 leg 가 정한다.

    Returns:
        (3,) 고관절 원점 기준 발 위치 [m], 바디 프레임.
    """
    return leg_link_positions(q, leg, l_hip, l_thigh, l_calf)[:, 3]


def compute_leg_ik(
    p_foot: Vec3,
    leg: int,
    l_hip: float = cfg.link_hip,
    l_thigh: float = cfg.link_upper,
    l_calf: float = cfg.link_lower,
    clip: bool = True,
) -> Vec3:
        """
        해석적 역기구학(Analytical IK)을 통해 목표 발 위치에 대한 관절 각도를 계산합니다.
        
        Parameters:
            p_foot (np.array): 고관절 원점(Ab/Ad joint) 기준 목표 발 위치 [x, y, z]^T
                            단, 다리의 로컬 좌표계 기준입니다.
            leg (int): 다리 인덱스 0..3 (FR FL RR RL).
                            abad 링크가 뻗는 y 방향 부호를 결정한다.

                            ★ 결함 11 - 예전에는 이 자리가 부호 있는 l_hip 이었고
                              "왼쪽은 양수, 오른쪽은 음수" 라고 독스트링에만
                              적혀 있었다. 호출부(dynamics.step)는 네 다리 모두에
                              같은 값을 넘겨 오른쪽 두 다리를 왼쪽 다리로 계산했다.
                              독스트링은 규약을 강제하지 못한다. 시그니처가 해야 한다.
            l_hip (float): 엉덩이 링크 Y축 오프셋의 **크기**. 부호는 leg 가 정한다.
            l_thigh (float): 허벅지 링크 길이
            l_calf (float): 종아리 링크 길이
            
        Returns:
            np.array: 계산된 관절 각도 [q1, q2, q3] (단위: rad)
                    (계산 불가능한 위치인 경우 NaN 포함 배열 반환)
        """
        x, y, z = p_foot[0], p_foot[1], p_foot[2]
        l_hip = LEG_HIP_SIGN[leg] * abs(l_hip)
        
        # ----------------------------------------------------
        # 1. q1 (Ab/Ad Roll 각도) 계산
        # Y-Z 평면에서 발까지의 직선 거리를 빗변으로 하는 직각삼각형 활용
        # ----------------------------------------------------
        # y-z 평면 원점에서 발끝까지의 직선 거리 제곱
        L_yz_sq = y**2 + z**2 
        
        # 목표 위치가 l_hip 길이보다 안쪽에 있으면 기구학적으로 도달 불가능
        if L_yz_sq < l_hip**2:
            return np.array([np.nan, np.nan, np.nan])
        
        L_yz = np.sqrt(L_yz_sq)
        
        # 다리 평면(Pitch 평면)의 회전 각도 q1 도출.
        #
        # ★ 결함 11 - arctan2 항의 부호가 반대였다. 주석의 "부호가 반전될 수
        #   있습니다"가 실제로 반전되어 있었다는 뜻이다. 목표 y 가 +0.10 일 때
        #   FK 는 -0.10 을 돌려주었고(왕복 오차 0.2m), y=0 인 초기 자세에서는
        #   이 항이 0 이라 오차가 드러나지 않았다. 정지 상태로는 절대 못 잡는
        #   부호 오류이며, 결함 1(Rz vs Rz^T)이 psi=0 에서 숨어 있던 것과
        #   구조가 같다.
        #
        #   검증: leg_forward_kinematics 와의 왕복 오차가 400 개 무작위 목표에서
        #   1.6e-16 m (기존 부호로는 최대 0.24 m).
        alpha = np.arcsin(l_hip / L_yz)
        q1 = np.arctan2(y, -z) - alpha 

        # ----------------------------------------------------
        # 2. q3 (Knee Pitch 각도) 계산
        # X와 (새롭게 회전된 Z) 평면에서의 제2 코사인 법칙 활용
        # ----------------------------------------------------
        # Roll 회전(q1)을 풀었을 때, Pitch 평면상에서 발끝의 가상 Z 길이 (z_pitch)
        z_pitch = -np.sqrt(L_yz_sq - l_hip**2)
        
        # 고관절(Hip Pitch 모터)에서 발끝까지의 직선 거리 제곱
        L_xz_sq = x**2 + z_pitch**2
        
        # 제2 코사인 법칙: L^2 = l1^2 + l2^2 - 2*l1*l2*cos(pi - q3)
        # cos(q3) = (L^2 - l1^2 - l2^2) / (2*l1*l2)
        cos_q3 = (L_xz_sq - l_thigh**2 - l_calf**2) / (2 * l_thigh * l_calf)
        
        # 도달 불가능한 작업 공간(Workspace) 제한 처리
        cos_q3 = np.clip(cos_q3, -1.0, 1.0)
        
        # 무릎은 보통 뒤로 꺾이므로(역관절) 음수 값을 취함. 
        # 정관절을 원하면 양수로 변경. (치타3는 무릎이 뒤로 꺾이는 - 각도)
        q3 = -np.arccos(cos_q3) 

        # ----------------------------------------------------
        # 3. q2 (Hip Pitch 각도) 계산
        # X-Z_pitch 평면에서의 아크탄젠트와 무릎 꺾임에 의한 보상 각도 합산
        # ----------------------------------------------------
        # Hip에서 발끝을 바라보는 직선의 각도
        theta_1 = np.arctan2(-x, -z_pitch)
        
        # Hip에서 무릎이 차지하는 내부 각도
        theta_2 = np.arctan2(l_calf * np.sin(q3), l_thigh + l_calf * np.cos(q3))
        
        # 최종 Hip 각도
        q2 = theta_1 - theta_2

        # 관절 한계는 기구학이 아니라 포화(saturation)다. 둘을 섞으면
        # 'IK 가 틀렸는가'와 '한계에 걸렸는가'를 구별할 수 없어진다.
        # 런타임은 clip=True, 왕복 검증은 clip=False 로 본다.
        if not clip:
            return np.array([q1, q2, q3])
        return clip_q(q1, q2, q3)


def get_q(
    height: float,
    l_thigh: float = cfg.link_upper,
    l_calf: float = cfg.link_lower,
) -> Vec3x4:
    """발이 고관절 바로 아래(height 만큼)에 있는 자세의 관절각 (3,4).

    예전에는 q1 = 0 을 네 다리에 그대로 복사했다. 그런데 abad 링크가 y 로
    l_hip 만큼 뻗어 있으므로 q1 = 0 이면 발은 고관절 바로 아래가 아니라
    옆으로 8cm 밀린 곳에 있다. 즉 함수 이름과 실제 자세가 달랐다.

    지금은 compute_leg_ik 를 다리마다 불러 같은 기구학을 쓴다. 결과적으로
    q1 은 좌우가 부호 대칭이고 q2/q3 는 네 다리가 같다.
    """
    p_under_hip = np.array([0.0, 0.0, -height])
    return np.stack(
        [compute_leg_ik(p_under_hip, leg, l_thigh=l_thigh, l_calf=l_calf)
         for leg in range(4)],
        axis=1,
    )


def clip_q(q1: float, q2: float, q3: float) -> Vec3:
    clip_q1 = np.clip(q1, cfg.min_q1, cfg.max_q1)
    clip_q2 = np.clip(q2, cfg.min_q2, cfg.max_q2)
    clip_q3 = np.clip(q3, cfg.min_q3, cfg.max_q3)
    return np.array([clip_q1, clip_q2, clip_q3])


def get_interior_angle(a: float, b: float, c: float) -> float:
    # 1. 코사인 분수값 계산
    cos_theta = (a**2 + b**2 - c**2) / (2 * a * b)
    
    # 2. 수치적 예외 처리 (아크코사인 에러 방지용 안전장치)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    
    # 3. 역코사인으로 라디안 각도 계산
    return np.arccos(cos_theta)


def get_r_feet_bf(
    Pfoot_bf: Vec3x4,
    hip_location_bf: Vec3x4 | None = None,
) -> Vec3x4:
    """hip 기준 발 위치 -> CoM 기준 발 위치 (body frame).

    기존 시그니처는 body_length/body_width 를 받았으나 본문에서 쓰지 않고
    모듈 전역 hip_location_bf 를 참조했다. 그 암묵 의존을 인자로 드러낸다.
    (호출부는 모두 Pfoot_bf 하나만 넘기므로 동작은 동일하다.)

    Args:
        Pfoot_bf: (3,4) 고관절 원점 기준 발 위치. 열 = Leg
        hip_location_bf: (3,4) CoM 기준 고관절 위치. 기본값은 config 의 값

    Returns:
        (3,4) CoM 기준 발 위치
    """
    if hip_location_bf is None:
        hip_location_bf = cfg.hip_location_bf
    # 원래는 for i in range(4) 루프였다. 원소별 덧셈이라 결과가 비트 단위로 같다.
    return Pfoot_bf + hip_location_bf


def compute_leg_jacobian(
    q: Vec3,
    l_hip: float = cfg.link_hip,
    l_thigh: float = cfg.link_upper,
    l_calf: float = cfg.link_lower,
) -> Mat3:
    """
    관절 각도 q = [q1, q2, q3]를 받아 3x3 자코비안 행렬을 반환합니다.
    """
    q1, q2, q3 = q[0], q[1], q[2]
    
    # 삼각함수 연산
    s1, c1 = np.sin(q1), np.cos(q1)
    s2, c2 = np.sin(q2), np.cos(q2)
    s23, c23 = np.sin(q2 + q3), np.cos(q2 + q3)
    
    # 공통 항 
    term_c = l_thigh * c2 + l_calf * c23
    term_s = l_thigh * s2 + l_calf * s23
    
    # 3x3 자코비안 생성
    J = np.zeros((3, 3))
    
    # Row 0 (X)
    J[0, 0] = 0.0
    J[0, 1] = -term_c
    J[0, 2] = -l_calf * c23
    
    # Row 1 (Y)
    J[1, 0] = -l_hip * s1 - c1 * term_c
    J[1, 1] = s1 * term_s
    J[1, 2] = l_calf * s1 * s23
    
    # Row 2 (Z)
    J[2, 0] = -l_hip * c1 + s1 * term_c
    J[2, 1] = c1 * term_s
    J[2, 2] = l_calf * c1 * s23
    
    return J
