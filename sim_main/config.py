# config.py
# 논문에 기제된 로봇 파라미터 및 상수
from enum import IntEnum

# Robot 
m = 43 # [kg]
Ixx = 0.41 # [kgm^2]
Iyy = 2.1 # [kgm^2]
Izz = 2.1 # [kgm^2]
body_length = 0.6 # [m]
body_width = 0.256 # [m]
body_height = 0.2 # [m]
leg_length = 0.34 # [m]

# Constant
μ = 0.6 #
gz = -9.8 # [m/s^2]

# Cost function matrix coefficient
L_w_th = 1 # weight of attitude (THETA) in L matrix
L_w_z = 50 # weight of height(z) in L matrix
L_w_yr = 1 # weight of yaw rate in L matrix
L_w_v = 1 # weight of velocity (Vxyz) in L matrix
K_w_f = 1e-6 # alpha

# Constraint parameters
τmax = 250 # [Nm]
fmin = 10 # [N]
fmax = 666 # [N]

# ==========================================
# 4. 차원 및 인덱스 정의 (Dimensions & Enumerations)
# ==========================================
# 차원 상수
N_STATE = 13            # 13차원 상태 벡터
N_LEG = 4               # 4족 보행
N_U_LEG = 3             # 다리당 3개의 힘 (Fx, Fy, Fz)
N_U = N_LEG * N_U_LEG   # 총 제어 입력 12차원

N_XYZ = 3

# 공간 좌표 인덱스 (XYZ)
class XYZ(IntEnum):
    X = 0
    Y = 1
    Z = 2

# 다리 인덱스 (Leg ID)
class LegID(IntEnum):
    FR = 0  # Front Right (우측 앞다리)
    FL = 1  # Front Left  (좌측 앞다리)
    RR = 2  # Rear Right  (우측 뒷다리)
    RL = 3  # Rear Left   (좌측 뒷다리)

# 상태 벡터 인덱스 (State ID)
class StateID(IntEnum):
    ROLL = 0
    PITCH = 1
    YAW = 2
    X = 3
    Y = 4
    Z = 5
    WX = 6
    WY = 7
    WZ = 8
    VX = 9
    VY = 10
    VZ = 11
    GRAV = 12

# ==========================================
# 5. 행렬 슬라이싱 편의 도구 (Slicing Objects)
# ==========================================
IDX_ANG = slice(0, 3)     # 각도 [Roll, Pitch, Yaw]
IDX_POS = slice(3, 6)     # 위치 [X, Y, Z]
IDX_ANGVEL = slice(6, 9)  # 각속도 [Wx, Wy, Wz]
IDX_VEL = slice(9, 12)    # 선속도 [Vx, Vy, Vz]