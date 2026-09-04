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


# 기구학 계산은 kinematics.py 로, 회전 변환은 rotations.py 로 옮겼다.
# 이 파일은 로봇 상수만 담는다.
# TODO(Step 1-2c): compute_bezier 는 궤적 생성이므로 bezier.py 로 옮긴다.
# TODO(Step 1-5): get_13d_state 는 RobotState.to_mpc_vector 로 대체된다.


def get_13d_state(ang, p, angvel, v):
    # x = [Roll, Pitch, Yaw, X, Y, Z, Wx, Wy, Wz, Vx, Vy, Vz, gravity=1.0]^T
    return np.vstack((ang, p, angvel, v, np.array([[1.0]])))


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
