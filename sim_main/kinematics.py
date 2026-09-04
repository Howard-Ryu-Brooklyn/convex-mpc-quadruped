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

import config as cfg


def compute_leg_ik(p_foot, l_hip=cfg.link_hip, l_thigh=cfg.link_upper, l_calf=cfg.link_lower):
        """
        해석적 역기구학(Analytical IK)을 통해 목표 발 위치에 대한 관절 각도를 계산합니다.
        
        Parameters:
            p_foot (np.array): 고관절 원점(Ab/Ad joint) 기준 목표 발 위치 [x, y, z]^T
                            단, 다리의 로컬 좌표계 기준입니다.
            l_hip (float): 엉덩이 링크 Y축 오프셋 (왼쪽 다리: 양수, 오른쪽 다리: 음수)
            l_thigh (float): 허벅지 링크 길이
            l_calf (float): 종아리 링크 길이
            
        Returns:
            np.array: 계산된 관절 각도 [q1, q2, q3] (단위: rad)
                    (계산 불가능한 위치인 경우 NaN 포함 배열 반환)
        """
        x, y, z = p_foot[0], p_foot[1], p_foot[2]
        
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
        
        # 다리 평면(Pitch 평면)의 회전 각도 q1 도출
        # arctan2(z, y)로 전체 각도를 구하고, l_hip에 의한 오프셋 각도 보상
        # (주의: 로봇의 좌표계 방향에 따라 부호가 반전될 수 있습니다)
        alpha = np.arcsin(l_hip / L_yz)
        q1 = -np.arctan2(y, -z) - alpha 

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

        cliped_q = clip_q(q1,q2,q3)
        # print('cliped_q',cliped_q/DEG2RAD)
        return cliped_q


def get_q(height, l_thigh=cfg.link_upper, l_calf=cfg.link_lower):
    # 힙 바로 아래 발 위치가 있다고 가정
    q1 = 0
    q2 = get_interior_angle(l_thigh, height, l_calf)
    q3 = - np.pi + get_interior_angle(l_thigh, l_calf, height)
    cliped_q = clip_q(q1,q2,q3)
    
    q_matrix = np.tile(cliped_q.reshape(3,1), (1, 4))
    
    return q_matrix


def clip_q(q1,q2,q3):
    clip_q1 = np.clip(q1, cfg.min_q1, cfg.max_q1)
    clip_q2 = np.clip(q2, cfg.min_q2, cfg.max_q2)
    clip_q3 = np.clip(q3, cfg.min_q3, cfg.max_q3)
    return np.array([clip_q1, clip_q2, clip_q3])


def get_interior_angle(a, b, c):
    # 1. 코사인 분수값 계산
    cos_theta = (a**2 + b**2 - c**2) / (2 * a * b)
    
    # 2. 수치적 예외 처리 (아크코사인 에러 방지용 안전장치)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    
    # 3. 역코사인으로 라디안 각도 계산
    return np.arccos(cos_theta)


def get_r_feet_bf(Pfoot_bf, hip_location_bf=None):
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


def compute_leg_jacobian(q, l_hip=cfg.link_hip, l_thigh=cfg.link_upper, l_calf=cfg.link_lower):
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
