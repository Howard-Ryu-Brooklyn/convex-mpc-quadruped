# 프로젝트 이력 (초기 기록)


> 원본: `simulation.ipynb` — 저장소 정리 과정에서 노트북을 버리고 내용만 옮겼다.


# 📌 Convex MPC 기반 4족 보행 로봇 최적 제어 구현 및 이력 관리

본 주피터 노트북은 **MIT Cheetah 3의 Convex MPC(볼록 모델 예측 제어)** 논문을 단계적으로 분석하고 파이썬으로 구현하며 시뮬레이션을 검증하기 위해 작성되었습니다.

---

## 📌 프로젝트 개요

- **목적**: Convex MPC 논문 구조를 깊이 있게 이해하고, 단계별 구현을 통해 4족 보행 로봇의 동역학 제어 기반을 마련합니다.
- **개발 환경**: Jupyter Notebook (Python 3) & VS Code (Git/GitHub 연동)
- **현재 구현 단계**:
  - [x] **Step 1**: 오픈소스 라이브러리 기반 MPC 설계 및 기본 시각화(Visualization) 구현
  - [x] **Step 2**: 선형 이산 시간 모델 유도 및 QP(이차계획법) 솔버 행렬 직접 설계/구현 (Unconstrained 및 마찰 원뿔 제약조건 기초)
  - [ ] **Step 3 (우선순위 제외)**: Gait Scheduler(보행 계획) 및 스윙 다리 토크 제어 (추후 구현 예정)

---


## 🛠️ 개발 로드맵 및 단계별 구현 세부 사항
![제어기 블록도](plot/diagram.png)

### 1. 상태 및 입력 벡터 정의
- **상태 벡터 ($\mathbf{x} \in \mathbb{R}^{12}$):** $$\mathbf{x} = [\boldsymbol{\Theta}^\top, \mathbf{p}^\top, \boldsymbol{\omega}^\top, \mathbf{v}^\top 1]^\top$$
  - $\boldsymbol{\Theta}$: 자세(Roll, Pitch, Yaw)
  - $\mathbf{p}$: 무게중심(CoM) 위치
  - $\boldsymbol{\omega}$: 몸체 각속도
  - $\mathbf{v}$: 몸체 선속도 (행렬 계산 단순화를 위한 상태변수 편입)
  - 1: 중력 가속도 처리용 상수 상태 변수
- **입력 벡터 ($\mathbf{u} \in \mathbb{R}^{12}$):** 4개 다리의 지면 반발력(GRF)
  $$\mathbf{u} = [\mathbf{f}_1^\top, \mathbf{f}_2^\top, \mathbf{f}_3^\top, \mathbf{f}_4^\top]^\top$$

---

### 2. 단일 강체 동역학 (Simplified Dynamics)
- **뉴턴-오일러 방정식:**
  $$\ddot{\mathbf{p}} = \sum \mathbf{f}_i / m - \mathbf{g}$$
  $$\frac{d}{dt}(\mathbf{I}\boldsymbol{\omega}) = \sum (\mathbf{r}_i \times \mathbf{f}_i)$$
  $$ \mathbf{\dot{R}}= [\omega]_\times \mathbf{R} $$

- **선형화된 시스템 ($\dot{\mathbf{x}}(t) = \mathbf{A}_c(\psi) \mathbf{x}(t) + \mathbf{B}_c(r_1,\dots,r4,\psi) \mathbf{u}(t)$):**
  $$\begin{bmatrix} \dot{\boldsymbol{\Theta}} \\ \dot{\mathbf{p}} \\ \dot{\boldsymbol{\omega}} \\ \dot{\mathbf{v}} \\ \dot{g} \end{bmatrix} = 
  
  \begin{bmatrix} \mathbf{0_3} & \mathbf{0_3} & \mathbf{R}(\psi)^\top & \mathbf{0_3} & \mathbf{0}_{3X1}\\ \mathbf{0_3} & \mathbf{0_3} & \mathbf{0_3} & \mathbf{\hat{I}_3} & \mathbf{0}_{3X1} \\ \mathbf{0_3} & \mathbf{0_3} & \mathbf{0_3} & \mathbf{0_3} & \mathbf{0}_{3X1} \\ \mathbf{0_3} & \mathbf{0_3} & \mathbf{0_3} & \mathbf{0_3} & \mathbf{g}_{world} \\ \mathbf{0}_{1X3} & \mathbf{0}_{1X3} & \mathbf{0}_{1X3} & \mathbf{0}_{1X3} & 0 \end{bmatrix} \mathbf{x} + 
  
  \begin{bmatrix} \mathbf{0_3} & \dots & \mathbf{0_3} \\ \mathbf{0_3} & \dots & \mathbf{0_3} \\ \mathbf{\hat{I}}^{-1}(\mathbf{r}_1 \times) & \cdots & \mathbf{\hat{I}}^{-1} (\mathbf{r}_4 \times) \\ \frac{1}{m}\mathbf{1}_3 & \dots & \frac{1}{m}\mathbf{1}_{3} \\
  \mathbf{0}_{1X3} & \dots & \mathbf{0}_{1X3} \end{bmatrix} \mathbf{u} 
  $$
$\mathbf{g}_{world} = [0, 0, -9.81]^\top $

$\mathbf{\hat{I}} = \mathbf{R}_{\psi} \mathbf{I}_{B} \mathbf{R}^\top _{\psi} $

$ \mathbf{A}_c \text{와} \mathbf{B}_c \text{가 yaw 각도와 footstep location}(r_i) \text{에 의한 함수이고 이를 미리 계산할 수 있다면 LTV 시스템이 되어} $
$ \text{convex MPC를 사용하기에 적절해진다. reference trajectory와 footstep controller로 부터 받은} $
$ \psi\text{와} r_i \text{ 값을 받아 } \mathbf{\hat{B}}_c\text{를 계산할 수 있다.} \mathbf{\hat{A}}_c \text{또한 reference trajectory의 } \psi \text{값을 받아 계산 가능하다.}$


---
$ \mathbf{A}_c \text{와} \mathbf{B}_c를 ZOH 이산화 하면 \mathbf{A}, \mathbf{B}$
$$ \frac{d}{dt} 
\begin{bmatrix} \mathbf{x} \\ \mathbf{u} \end{bmatrix} = 
\begin{bmatrix} \mathbf{A} & \mathbf{B} \\ 0 & 0 \end{bmatrix} 
\begin{bmatrix} \mathbf{x} \\ \mathbf{u} \end{bmatrix} $$

$\mathbf{A}와 \mathbf{B}를 reference 값을 받아 계산하면 \mathbf{\hat{A}}, \mathbf{\hat{B}} $
$$\mathbf{x}[k+1] = \mathbf{\hat{A}} \mathbf{x}[k] + \mathbf{\hat{B}}[n]\mathbf{u}[k]$$
---

---
### IV. 모델 예측 제어 (Model Predictive Control) - 식 (18)~(24)
#### 1. MPC 정식화 ####
$$\min_{\mathbf{x}, \mathbf{u}} \sum_{i=0}^{k-1} \Vert{}\mathbf{x}_{i+1} - \mathbf{x}_{i+1,\text{ref}}\Vert{}_{\mathbf{Q}_i}^2 + \Vert{}\mathbf{u}_i\Vert{}_{\mathbf{R}_i}^2 \tag{18}$$
$$\text{subject to } \mathbf{x}_{i+1} = \mathbf{A}_i\mathbf{x}_i + \mathbf{B}_i\mathbf{u}_i, \quad i = 0, \dots, k-1 \tag{19}$$
$$\mathbf{c}_i \le \mathbf{C}_i\mathbf{u}_i \le \bar{\mathbf{c}}_i, \quad i = 0, \dots, k-1 \tag{20}$$
$$\mathbf{D}_i\mathbf{u}_i = \mathbf{0}, \quad i = 0, \dots, k-1 \tag{21}$$

$$f_{\text{min}} \le f_z \le f_{\text{max}} \tag{22}$$
$$-\mu f_z \le f_x \le \mu f_z \tag{23}$$
$$-\mu f_z \le f_y \le \mu f_z \tag{24}$$


#### 2. 예측 모델 행렬 구성 (Prediction Matrix) - 식 (19) ~ (22)
미래 $K$ 스텝 동안의 상태를 초기 상태 $\mathbf{x}_0$와 입력 시퀀스 $\mathbf{U}$의 함수로 나타내기 위해 식을 전개합니다.

- **상태 전이식:**
  $$\mathbf{x}_1 = \mathbf{A}_0 \mathbf{x}_0 + \mathbf{B}_0 \mathbf{u}_0$$
  $$\mathbf{x}_2 = \mathbf{A}_1 \mathbf{x}_1 + \mathbf{B}_1 \mathbf{u}_1 = \mathbf{A}_1 \mathbf{A}_0 \mathbf{x}_0 + \mathbf{A}_1 \mathbf{B}_0 \mathbf{u}_0 + \mathbf{B}_1 \mathbf{u}_1$$

- **전체 예측식 (Condensed Formulation: 상태변수 $\mathbf{X}$를 초기상태값 $\mathbf{x}_0$로 대체):**
  $$\mathbf{X} = \mathbf{A}_{qp} \mathbf{x}_0 + \mathbf{B}_{qp} \mathbf{U}$$

  - **$\mathbf{X}$**: $[ \mathbf{x}_1^\top, \mathbf{x}_2^\top, \dots, \mathbf{x}_K^\top ]^\top$ (미래 상태 전체)
  - **$\mathbf{U}$**: $[ \mathbf{u}_0^\top, \mathbf{u}_1^\top, \dots, \mathbf{u}_{K-1}^\top ]^\top$ (미래 입력 전체)
  - **$\mathbf{A}_{qp}$**: 초기 상태가 미래에 미치는 영향을 담은 행렬 (13 X K)
  - **$\mathbf{B}_{qp}$**: 입력 시퀀스가 미래 상태에 미치는 하삼각 구조(Lower triangular)의 행렬 (13 X 3nK)
  - 상태변수 x를 입력변수 u로 완전히 대체하여 최적화 문제 크기를 줄일 수 있음, 하지만 큰 K에 대해 행렬이 밀집되어 계산량이 급증하고 A의 거듭제곱 계산으로 수치적 불안정성 발생 가능
  - 제약조건을 줄일 수 있음, 1. 동역학이 제약조건에서 제거됨(식 19제거) 목적함수에 그대로 녹아있음. 2. 공중에 뜬 다리를 선택적으로 제거하는 행렬 S 를 곱해 제약조건 D을 없앨 수 있음(식21제거)

#### 3. 비용 함수 및 최적화 - 식 (23) ~ (24)
목표 궤적(Reference) 추종을 위한 비용 함수를 정의합니다.

- **비용 함수:**
  $$J = \sum_{k=1}^{N} \|\mathbf{x}_k - \mathbf{x}_{ref,k}\|^2_{\mathbf{Q}} + \sum_{k=0}^{N-1} \|\mathbf{u}_k\|^2_{\mathbf{R}}$$
  (여기서 $\mathbf{Q}, \mathbf{R}$은 가중 행렬)

- **QP 정식화 (표준 이차 형식):**
$$\mathbf{X} = \mathbf{A}_{\text{qp}}\mathbf{x}_0 + \mathbf{B}_{\text{qp}}\mathbf{U} \tag{27}$$
$$J(\mathbf{U}) = \Vert{}\mathbf{A}_{\text{qp}}\mathbf{x}_0 + \mathbf{B}_{\text{qp}}\mathbf{U} - \mathbf{x}_{\text{ref}}\Vert{}_{\mathbf{L}}^2 + \Vert{}\mathbf{U}\Vert{}_{\mathbf{K}}^2 \tag{28}$$
$$\min_{\mathbf{U}} \frac{1}{2} \mathbf{U}^\top \mathbf{H} \mathbf{U} + \mathbf{U}^\top \mathbf{g} \tag{29}$$
$$\text{s. t. } \mathbf{c} \le \mathbf{C}\mathbf{U} \le \bar{\mathbf{c}} \tag{30}$$
$$\text{where } \mathbf{H} = 2(\mathbf{B}_{\text{qp}}^\top \mathbf{L} \mathbf{B}_{\text{qp}} + \mathbf{K}) \tag{31}$$
$$\mathbf{g} = 2\mathbf{B}_{\text{qp}}^\top \mathbf{L}(\mathbf{A}_{\text{qp}}\mathbf{x}_0 - \mathbf{x}_{ref}) \tag{32}$$

```python
import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from IPython.display import HTML # 주피터 환경에서 애니메이션 띄우기 위함
from mpl_toolkits.mplot3d.art3d import Poly3DCollection # ✅ 시각화를 위해 반드시 추가!



# ==========================================
# 1. 파라미터 설정
# ==========================================

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

loop_blue_hz = 30 #[Hz]
loop_red_hz = 1000 #[Hz]
loop_green_hz = 4500 #[Hz]
loop_sim_hz = 9000 #[Hz]

RAD2DEG = 180/np.pi
DEG2RAD = np.pi/180

# Robot Init State
P0 = np.array([[0.0], [0.0], [leg_length]]) # [m m m]^T
V0 = np.array([[0.0], [0.0], [0.0]]) # [m/s m/s m/s]^T
ANG0 = np.array([[0.00*DEG2RAD], [0.01*DEG2RAD], [90.0*DEG2RAD]]) # [roll, pitch, yaw]^T 
ANGVEL0 = np.array([[0.0], [0.0], [0.0]]) # pitch roll yaw [rad, rad ,rad]^T

X0 = np.vstack((P0, V0, ANG0, ANGVEL0))

# Input Variable
F_G = np.zeros((3,4)) # [f1(FR), f2(FL), f3(RR), f4(RL)], fi= [x,y,z]^T
T_motor = np.zeros((12,1)) # [t1(FR), t2(FL), t3(RR), t4(RL)] t1 in R^3 (ad/ab, hip, knee)

T_com = np.zeros((3,1))
g_vec = np.array([[0], [0], [gz]])

# Robot Sim State Vector (SRB)
ROBOT_Pos = P0.copy()
ROBOT_Vel = V0.copy()
ROBOT_Ang = ANG0.copy()
ROBOT_AngVel = ANGVEL0.copy()

ROBOT_Acc = np.zeros((3,1))
ROBOT_AngAcc = np.zeros((3,1))

RW_B = np.zeros((3,3)) # Rotation Matrix Body to World R=RzRyRx

IB = np.diag([Ixx, Iyy, Izz])
IW = np.zeros((3,3))
IW_INV = np.zeros((3,3))
rw_feet = np.zeros((3,4)) # 3(xyz) x 4(feet) FR FL RR RL

# Vector from COM to each feet in world frame
rb_feet = np.array([[body_length/2, body_length/2, -body_length/2, -body_length/2], 
                   [-body_width/2, body_width/2, -body_width/2, body_width/2],
                   [-body_height, -body_height, -body_height, -body_height]]) 
# stand at origin with yaw 0 degree (aligned with x-axis)

# rw_feet0 = RW_B*rb_feet
# r_feeti
# swing leg -> 토크 계산시 무시 
# ground leg -> 토크 계산

# Func: Cal Rotation Matrix
def get_rotation_matrix(roll, pitch, yaw):
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(roll), -np.sin(roll)], # 수정: roll 적용
                   [0, np.sin(roll), np.cos(roll)]])
    Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)], # 수정: pitch 적용
                   [0, 1, 0],
                   [-np.sin(pitch), 0, np.cos(pitch)]])
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                   [np.sin(yaw), np.cos(yaw), 0],
                   [0, 0, 1]])
    return Rz @ Ry @ Rx # Z-Y-X 순서

rw_feet = RW_B @ rb_feet

# simulation parameters
sim_freq = loop_sim_hz
sim_ts = 1/sim_freq # 10khz, 100us
sim_t0 = 0 # [sec]
sim_tf = 5 # [sec]
sim_t0_cnt = round(sim_t0/sim_ts)
sim_tf_cnt = round(sim_tf/sim_ts)

step_blue = round(sim_freq / loop_blue_hz)   # 9000 / 30 = 300
step_red = round(sim_freq / loop_red_hz)     # 9000 / 1000 = 9
step_green = round(sim_freq / loop_green_hz) # 9000 / 4500 = 2

# ==========================================
# 2. 실시간 시각화 창 초기 설정
# ==========================================
body_dims = [body_length, body_width, body_height] # ✅ 시각화 함수에 넘겨줄 리스트

def update_simulation_plot(ax, p, R, r_feet_body, body_dims):
    ax.cla() # 이전 프레임 지우기
    
    l, w, h = body_dims[0]/2, body_dims[1]/2, body_dims[2]/2

    # 1. 반투명 땅 그리기
    ground_range = 1.5
    xx, yy = np.meshgrid(
        np.linspace(p[0,0] - ground_range, p[0,0] + ground_range, 10),
        np.linspace(p[1,0] - ground_range, p[1,0] + ground_range, 10)
    )
    zz = np.zeros_like(xx)
    ax.plot_surface(xx, yy, zz, color='lightgray', alpha=0.25, edgecolor='none')

    # 2. 반투명 로봇 몸체 그리기
    corners_body = np.array([
        [ l,  w,  h], [ l, -w,  h], [-l, -w,  h], [-l,  w,  h], 
        [ l,  w, -h], [ l, -w, -h], [-l, -w, -h], [-l,  w, -h]  
    ]).T
    corners_world = p + R @ corners_body
    
    faces = [
        [corners_world[:, 0], corners_world[:, 1], corners_world[:, 2], corners_world[:, 3]],
        [corners_world[:, 4], corners_world[:, 5], corners_world[:, 6], corners_world[:, 7]],
        [corners_world[:, 0], corners_world[:, 1], corners_world[:, 5], corners_world[:, 4]],
        [corners_world[:, 2], corners_world[:, 3], corners_world[:, 7], corners_world[:, 6]],
        [corners_world[:, 1], corners_world[:, 2], corners_world[:, 6], corners_world[:, 5]],
        [corners_world[:, 0], corners_world[:, 3], corners_world[:, 7], corners_world[:, 4]]
    ]
    
    body_collection = Poly3DCollection(faces, facecolors='cyan', linewidths=1.2, edgecolors='blue', alpha=0.35)
    ax.add_collection3d(body_collection)

    # 3. rw_feet 지면 프로젝션 및 다리 선 긋기
    rw_feet = np.zeros((3, 4))
    for i in range(4):
        foot_world_raw = p + R @ r_feet_body[:, [i]]
        
        # XY는 그대로, Z는 0으로 프로젝션
        rw_feet[0, i] = foot_world_raw[0, 0]
        rw_feet[1, i] = foot_world_raw[1, 0]
        rw_feet[2, i] = 0.0  
        
        hip_world = p + R @ r_feet_body[:, [i]]
        
        # 힙에서 지면 발끝까지 다리 선 긋기
        ax.plot([hip_world[0,0], rw_feet[0,i]], 
                [hip_world[1,0], rw_feet[1,i]], 
                [hip_world[2,0], rw_feet[2,i]], color='darkorange', linewidth=3)
        ax.scatter(rw_feet[0, i], rw_feet[1, i], rw_feet[2, i], color='red', s=60)

    # 4. 플롯 시각 설정
    ax.set_xlim([p[0,0]-1.0, p[0,0]+1.0])
    ax.set_ylim([p[1,0]-1.0, p[1,0]+1.0])
    ax.set_zlim([-0.2, 1.0])

# ==========================================
# 2. 시각화 데이터 저장용 리스트 (History)
# ==========================================
history_p = []
history_R = []
history_r_feet = []

# 시각화 업데이트 주기 설정 (초당 30프레임)
vis_fps = 30
step_vis = round(sim_freq / vis_fps) # 9000 / 30 = 300스텝마다 1번씩 그림


for sim_cnt in range(sim_t0_cnt, sim_tf_cnt):

    # 행렬 업데이트
    roll, pitch, yaw = ROBOT_Ang[0,0], ROBOT_Ang[1,0], ROBOT_Ang[2,0]
    RW_B = get_rotation_matrix(roll, pitch, yaw)
    IW = RW_B @ IB @ RW_B.T
    IW_INV = np.linalg.inv(IW)
    rw_feet = RW_B @ rb_feet
    rw_feet[2, :] = -ROBOT_Pos[2, 0]

    if (sim_cnt % step_blue == 0):
        # Operator Input
        # Reference Trajectory
        # MPC (Rigid Body Model)

        # Test Ground Force for just standing intital height
        f_z_per_leg = (m * 9.8) / 4.0  # 약 105.35 N
        
        # 모든 다리(열)의 Z축(인덱스 2)에 105.35 N 인가
        F_G[2, :] = f_z_per_leg 
        # if sim_cnt*sim_ts >= 0.9:
        #     F_G[2, :] = 0


    if (sim_cnt % step_red == 0):
        # State Estimator
        # Swing Trajectory (multi-body model)

        # feet position update (just stand)
        rb_feet = np.array([[body_length/2, body_length/2, -body_length/2, -body_length/2], 
                   [-body_width/2, body_width/2, -body_width/2, body_width/2],
                   [-body_height, -body_height, -body_height, -body_height]]) 
        rw_feet = RW_B @ rb_feet
        rw_feet[2, :] = -ROBOT_Pos[2, 0]


    if (sim_cnt % step_green == 0):
        # Leg Position,Torque,Force Control
        motor_torque = np.zeros((4,3))

    
    # Robot Simulation (Single Rigid Body Model)
    # a = sum(F)/m
    # tau = sum(rxF) = d/dt(Iw) = I_dot w + I w_dot = I w_dot + [w]x(Iw)
    # w_dot = I^-1 (tau - [w]x(Iw))
    # Rdot = [w]XR

    # input = Fvec from MPC
    TotalForce = np.sum(F_G, axis=1, keepdims=True)
    ROBOT_Acc = TotalForce/m + g_vec

    T_com = np.zeros((3,1))
    for i in range(4):
        # 1D array cross product 후 3x1 로 reshape
        torque_i = np.cross(rw_feet[:, i], F_G[:, i])
        T_com += torque_i.reshape(3, 1)
    gyro_term = np.cross(ROBOT_AngVel.flatten(), (IW @ ROBOT_AngVel).flatten()).reshape(3,1)
    ROBOT_AngAcc = IW_INV @ (T_com - gyro_term)

    # update states
    ROBOT_Vel += ROBOT_Acc * sim_ts 
    ROBOT_Pos += ROBOT_Vel * sim_ts

    ROBOT_AngVel += ROBOT_AngAcc * sim_ts

    # 분모가 0이 되는 짐벌락 방지 (Pitch가 +-90도가 되는 것을 막음)
    pitch_safe = np.clip(pitch, -89 * DEG2RAD, 89 * DEG2RAD)

    # 논문 식 (10) 완벽 구현
    R_rate2euler = np.array([
        [np.cos(yaw) / np.cos(pitch_safe),    np.sin(yaw) / np.cos(pitch_safe),    0],
        [-np.sin(yaw),           np.cos(yaw),            0],
        [np.cos(yaw) * np.tan(pitch_safe),    np.sin(yaw) * np.tan(pitch_safe),    1]
    ])

    # 상태 업데이트
    ROBOT_Ang += R_rate2euler @ ROBOT_AngVel * sim_ts

    # ----------------------------------------------------
    # 데이터 로깅 (그리지 않고 리스트에 데이터만 저장)
    # ----------------------------------------------------
    if (sim_cnt % step_vis == 0):
        history_p.append(ROBOT_Pos.copy())
        history_R.append(RW_B.copy())
        history_r_feet.append(rb_feet.copy()) # 애니메이션은 바디 기준 발 위치를 원함

print("연산 완료! 애니메이션 렌더링 시작...")


# ==========================================
# 4. 저장된 데이터로 애니메이션 재생 (FuncAnimation)
# ==========================================
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

def update_anim(frame_idx):
    # 저장해둔 히스토리에서 현재 프레임의 데이터를 꺼냄
    p = history_p[frame_idx]
    R = history_R[frame_idx]
    r_feet_body = history_r_feet[frame_idx]
    
    # ✅ 여기서 아까 정의한 시각화 함수를 호출!
    update_simulation_plot(ax, p, R, r_feet_body, body_dims)

# 애니메이션 객체 생성
ani = animation.FuncAnimation(fig, update_anim, frames=len(history_p), interval=33, blit=False)

# 주피터 노트북에 HTML 동영상 플레이어로 출력
plt.close(fig) # 불필요한 빈 피규어 출력 방지
HTML(ani.to_jshtml())
```
