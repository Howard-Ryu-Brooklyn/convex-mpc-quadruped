# config.py
# 논문에 기제된 로봇 파라미터 및 상수
import numpy as np

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


# ─────────────────────────────────────────────────────────────────────
# 이 파일은 이제 로봇 상수만 담는다.
#   기구학  -> kinematics.py
#   회전    -> rotations.py
#   궤적    -> bezier.py
#   상태    -> robot_types.RobotState (get_13d_state 를 대체)
#
# TODO(Step 6): 이 상수들을 dataclass + YAML 로 옮긴다. 지금은 모듈 전역이라
#   설정을 바꿔가며 스윕을 돌릴 수 없고, 두 설정을 한 프로세스에서 비교할 수도
#   없다. MPC 가중치(L_w_*) 튜닝이 그 첫 사용처가 된다.
# ─────────────────────────────────────────────────────────────────────
