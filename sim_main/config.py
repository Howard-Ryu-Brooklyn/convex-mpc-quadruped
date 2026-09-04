# config.py
# 논문에 기제된 로봇 파라미터 및 상수
import numpy as np
import math

# Robot 
m = 43 # [kg]
Ixx = 0.41 # [kgm^2]
Iyy = 2.1 # [kgm^2]
Izz = 2.1 # [kgm^2]
body_length = 0.6 # [m]
body_width = 0.256 # [m]
body_height = 0.2 # [m]
leg_length_straight = 0.68 # [m]

# I = (1/12)mL^2
link_hip = 0.08 # [m]
link_upper = 0.34 # [m]
link_lower = 0.34 # [m]
Ilink_hip = 0.0005 # [kgm^2] 0.05kg, 0.08m
Ilink_upper = 0.00318 # [kgm^2] 0.33kg, 0.34m
Ilink_lower = 0.00318 # [kgm^2] 0.33kg, 0.34m

# Constant
MU_FRICTION = 0.6 #
gz = -9.8 # [m/s^2]

# Cost function matrix coefficient
L_w_th = 1 # weight of attitude (THETA) in L matrix
L_w_z = 50 # weight of height(z) in L matrix
L_w_yr = 1 # weight of yaw rate in L matrix
L_w_v = 1 # weight of velocity (Vxyz) in L matrix
K_w_f = 1e-6 # alpha

# Constraint parameters
TAU_MAX = 250 # [Nm]
fmin = 10 # [N]
fmax = 666 # [N]

DEG2RAD = np.pi/180

min_q1, max_q1 = -np.pi/4, np.pi/4
min_q2, max_q2 = -np.pi/2, np.pi/2
min_q3, max_q3 = -150*DEG2RAD, 0


hip_location_bf = np.array([
            [ body_length/2, -body_width/2, 0], # FR
            [ body_length/2,  body_width/2, 0], # FL
            [-body_length/2, -body_width/2, 0], # RR
            [-body_length/2,  body_width/2, 0]  # RL
        ]).T

def compute_leg_ik(p_foot, l_hip=link_hip, l_thigh=link_upper, l_calf=link_lower):
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


def get_q(height, l_thigh=link_upper, l_calf=link_lower):
    # 힙 바로 아래 발 위치가 있다고 가정
    q1 = 0
    q2 = get_interior_angle(l_thigh, height, l_calf)
    q3 = - np.pi + get_interior_angle(l_thigh, l_calf, height)
    cliped_q = clip_q(q1,q2,q3)
    
    q_matrix = np.tile(cliped_q.reshape(3,1), (1, 4))
    
    return q_matrix

def clip_q(q1,q2,q3):
    clip_q1 = np.clip(q1, min_q1, max_q1)
    clip_q2 = np.clip(q2, min_q2, max_q2)
    clip_q3 = np.clip(q3, min_q3, max_q3)
    return np.array([clip_q1, clip_q2, clip_q3])

def get_interior_angle(a, b, c):
    # 1. 코사인 분수값 계산
    cos_theta = (a**2 + b**2 - c**2) / (2 * a * b)
    
    # 2. 수치적 예외 처리 (아크코사인 에러 방지용 안전장치)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    
    # 3. 역코사인으로 라디안 각도 계산 후 도(Degree) 단위 변환
    radian = np.arccos(cos_theta)
    degree = np.degrees(radian)
    
    return radian


# 13차원 상태 벡터 구성 헬퍼 함수 (중력 항 1.0 추가)
def get_13d_state(ang, p, angvel, v):
    # x = [Roll, Pitch, Yaw, X, Y, Z, Wx, Wy, Wz, Vx, Vy, Vz, gravity=1.0]^T
    return np.vstack((ang, p, angvel, v, np.array([[1.0]])))

def get_r_feet_bf(Pfoot_bf, body_length=body_length, body_width=body_width):
    L = body_length/2
    W = body_width/2
    # hip_location_bf = np.array([
    #     [ L, -W, 0], # FR
    #     [ L,  W, 0], # FL
    #     [-L, -W, 0], # RR
    #     [-L,  W, 0]  # RL
    # ]).T # vector from hip to foot in body frame

    r_feet_bf = np.zeros((3,4))
    for i in range(4):
        r_feet_bf[:,i] = Pfoot_bf[:,i] + hip_location_bf[:,i]
    return r_feet_bf


def compute_bezier(s, control_points):
    """
    N차 베지에 곡선상의 한 점을 계산하는 함수
    
    :param s: 0.0 ~ 1.0 사이의 궤적 진행률 (상태 위상, Phase)
    :param control_points: 제어점 리스트, shape=(N+1, 3) (3차원 좌표)
    :return: 진행률 s에서의 발의 3차원 위치 (x, y, z)
    """
    # s가 [0,1] 밖이면 번스타인 다항식이 외삽한다. 3차 베지에는
    # s>1에서 p3 + 3(s-1)(p3-p2) 로 발산하며, 스윙 궤적에서는
    # p3-p2 = -clearance*ẑ 이므로 발이 지면 아래로 파고든다.
    # swing_trajectory_generator.compute_bezier_with_kinematics와 동일하게
    # 함수 내부에서 방어한다.
    s = np.clip(s, 0.0, 1.0)

    # 제어점 배열을 numpy 배열로 변환
    pts = np.array(control_points)
    n = len(pts) - 1 # 베지에 곡선의 차수
    
    p_s = np.zeros(3) # 반환할 (x, y, z) 좌표 초기화
    
    for i in range(n + 1):
        # 번스타인 다항식 (Bernstein Polynomial) 계산: (n C i) * (1-s)^(n-i) * s^i
        coeff = math.comb(n, i) * ((1 - s) ** (n - i)) * (s ** i)
        
        # 제어점에 가중치를 곱하여 누적
        p_s += coeff * pts[i]
        
    return p_s

# get_Rx / get_Ry / get_Rz 는 rotations.py 로 옮겼다.
# (이 파일 안에서만 정의되고 어디서도 호출되지 않던 죽은 코드였으며,
#  dynamics 가 같은 행렬을 인라인으로 다시 만들고 있었다.)
