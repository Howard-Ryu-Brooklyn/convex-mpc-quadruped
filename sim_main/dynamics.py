import numpy as np

from kinematics import clip_q, compute_leg_ik, get_q
from plant_base import PlantBase
from robot_types import RobotState
from rotations import omega_to_rpy_rate, rpy_to_matrix

class SRBDynamics(PlantBase):
    """단일 강체(SRB) 시뮬레이터.

    발 위치는 운동학적으로 강제되고(다리에 질량이 없다), 몸통만 GRF 로 적분된다.

    TODO(Step 1-6): 정식 패키지로 옮기며 파일을 plant/srbd.py 로,
        클래스를 SRBDPlant 로 이름을 맞춘다.
    """

    def __init__(self, dt, mass, inertia_diag, gz, init_pos, init_vel, init_ang, init_angvel, link_info, init_leg_ang, init_leg_angvel, hip_offsets_body, r_feet_wf, Ilink):
        """
        단일 강체 동역학(SRBD) 시뮬레이터 초기화
        """
        self._dt = dt
        self.m = mass
        self.IB = np.diag(inertia_diag)
        self.g_vec = np.array([[0], [0], [gz]])
        
        # 로봇 상태 벡터 (State Variables)
        self.pos = init_pos.copy()
        self.vel = init_vel.copy()
        self.ang = init_ang.copy()
        self.ang_vel = init_angvel.copy()
        
        self.acc = np.zeros((3, 1))
        self.ang_acc = np.zeros((3, 1))

        self.l1 = link_info[0]
        self.l2 = link_info[1]
        self.lhip = link_info[2]
        self.q = init_leg_ang.copy()
        # print(self.q)
        self.qdot = init_leg_angvel.copy()
        self.qddot = np.zeros([3, 4])

        self.hip_offsets_body = hip_offsets_body.copy() # (3,4)
        self.r_feet_wf = r_feet_wf.copy() # (3,4)
        self.Ilink = np.diag(Ilink) # (1,3)
        
        self.Rw_b = rpy_to_matrix(init_ang.flatten()[0], init_ang.flatten()[1], init_ang.flatten()[2])
        # [[Ihip-roll],[Iupper_pitch],[Ilower_pitch]].T

    # ==========================================
    # PlantBase 인터페이스
    # ==========================================
    @property
    def dt(self) -> float:
        """물리 적분 주기 [s]."""
        return self._dt

    def observe(self) -> RobotState:
        """현재 상태의 스냅샷.

        아래의 P / V / ANG / ANGVEL 프로퍼티와 달리 내부 배열의 참조가 아니라
        복사본을 담은 불변 객체를 돌려준다. 기존 프로퍼티는 robot.P += x 같은
        코드로 시뮬레이터 내부를 오염시킬 수 있었다.

        ang_vel 을 omega_W 로 담는 근거: 자이로항 omega × (I_W omega) 와
        MPC 의 Θ̇ = R_z(psi)^T · omega 가 모두 world frame 을 전제한다.
        """
        return RobotState(
            p_com_W=self.pos.flatten(),      # flatten() 은 항상 복사본을 만든다
            v_com_W=self.vel.flatten(),
            rpy_W=self.ang.flatten(),
            omega_W=self.ang_vel.flatten(),
        )

    # ==========================================
    # 레거시 프로퍼티 (내부 배열의 살아있는 참조를 반환한다)
    # TODO(Step 1-5): 호출부를 observe() 로 옮기고 제거한다.
    # ==========================================
    @property
    def P(self):
        """현재 CoM 위치 벡터 [x, y, z]^T (3x1)"""
        return self.pos

    @property
    def V(self):
        """현재 CoM 선속도 벡터 [vx, vy, vz]^T (3x1)"""
        return self.vel

    @property
    def ANG(self):
        """현재 오일러 각 [roll, pitch, yaw]^T (3x1)"""
        return self.ang

    @property
    def ANGVEL(self):
        """현재 바디/기준 각속도 [wx, wy, wz]^T (3x1)"""
        return self.ang_vel
    
    @property
    def Q(self):
        """현재 모터 각도 [q1, q2, a3]^T (3x4)"""
        return self.q
    
    @property
    def RW_B(self):
        """rotation matrix from body to world (3x3)"""
        return self.Rw_b
    

    
    
    def step(self, F_G, r_feet_wf):
        """한 물리 스텝 진행 (PlantBase.step 구현).

        Args:
            F_G: (3,4) 지면 반발력 [N], world frame. 스윙 다리는 0 이어야 한다.
            r_feet_wf: (3,4) CoM 기준 발 위치 [m], world frame.

        Returns:
            (3,4) 고관절 기준 발 위치. 시각화·로깅용이며 PlantBase 계약에는
            없다. TODO(Step 1-5): 로깅 경로를 정리하며 반환값을 없앤다.
        """
        # 1. 행렬 업데이트 (회전 변환 및 관성 텐서)
        roll, pitch, yaw = self.ang[0, 0], self.ang[1, 0], self.ang[2, 0]
        self.Rw_b = rpy_to_matrix(roll, pitch, yaw)
        IW = self.Rw_b @ self.IB @ self.Rw_b.T
        IW_INV = np.linalg.inv(IW)
        
        self.r_feet_wf = r_feet_wf.copy()
        
        # 월드 프레임 발 위치 역계산 하여 바디 좌표계 발 위치 업데이트 
        
        r_feet_bf = self.Rw_b.T @ self.r_feet_wf
    

        # 3. 병진 운동 동역학 (Translation Dynamics)
        TotalForce = np.sum(F_G, axis=1, keepdims=True)
        self.acc = TotalForce / self.m + self.g_vec

        # 4. 회전 운동 동역학 (Rotational Dynamics)
        T_com = np.zeros((3, 1))
        for i in range(4):
            torque_i = np.cross(self.r_feet_wf[:, i], F_G[:, i])
            T_com += torque_i.reshape(3, 1)
            
        gyro_term = np.cross(self.ang_vel.flatten(), (IW @ self.ang_vel).flatten()).reshape(3, 1)
        self.ang_acc = IW_INV @ (T_com - gyro_term)

        # 5. 수치 적분 (Euler Integration)
        self.vel += self.acc * self.dt
        self.pos += self.vel * self.dt
        self.ang_vel += self.ang_acc * self.dt

        # 6. 각속도 -> 오일러 각 변화율 변환 (짐벌락 방지)
        #    Θ̇ = T(theta, psi) · omega_W. 이 식이 MPC 의 Ac[0:3,6:9] 와
        #    일치해야 하며, 어긋나 있던 것이 결함 1 이었다.
        self.ang += omega_to_rpy_rate(pitch, yaw) @ self.ang_vel * self.dt

        # 원점: hip (ab/ad motor)
        # 입력: f -> torque -> 출력: q1q2q3
        # 다리 기구학 계산
        # q1 q2 q3 = ad/ab(roll, -Z~link_upper, -90~90) hip(pitch, link_upper~link_lower, -150~0) knee(pitch, Y-link_hip, -45~45)
        # link_upper link_lower
        # pb_foot [x,y,z]

        # =======================================================
        # 6. 다리 역기구학 (IK) 및 발 위치 유지
        # =======================================================
        p_foot_local = np.zeros((3,4))
        for i in range(4):
            # 땅에 닿아있는 다리는 이전 루프의 글로벌 발 위치(rw_feet)를 그대로 유지함.
            # 스윙 다리라면 MPC/발 궤적 생성기가 만든 새로운 rw_feet_target을 받음.
            
            p_foot_local[:,i] = r_feet_bf[:,i] - self.hip_offsets_body[:,i] # hip->foot
            # print('p_foot_local', p_foot_local)
            q_ik = compute_leg_ik(p_foot_local[:,i], self.lhip, self.l1, self.l2) #get_q(-p_foot_local[2]) 
            
            # [Step C] 상태 업데이트 (시각화를 위해)
            # 이제 qddot, qdot을 적분하지 않고 그냥 q를 덮어씌웁니다.

            # self.q[:, i] = q_ik[:,i]
            self.q[:, i] = q_ik

        # print(self.q)
        # 로깅 및 시각화용 데이터 반환
        return p_foot_local.copy()
    



        # 실제로는 지면 반발력 -> 토크 -> 각도 변환 순서지만
        # 지면 반발력에 대한 접촉 반발력을 구하기 번거롭기 때문에 
        # 역기구학으로 대체 
        
            # Torque_motor = np.zeros((3, 4))
            # pb_foot = np.zeros((3, 4))
            # pb_knee = np.zeros((3, 4))
            # F_reaction= np.where(F_G != 0, 45*9.81/2, F_G)

            # for i in range(4):
            #     J = self.compute_leg_jacobian(self.q[:, i], self.lhip, self.l1, self.l2)
            #     F_B = self.R.T @ F_G[:, i] - F_reaction[:,i]
            #     Torque_motor[:, i] = J.T @ F_B
            #     # tau max 제한

            #     # 관절 가속도, 속도, 위치 업데이트 (Euler Integration)
            #     self.qddot[:, i] = np.linalg.inv(self.Ilink) @ Torque_motor[:, i]
            #     self.qdot[:, i] = self.qdot[:, i] + self.qddot[:, i] * self.dt
            #     self.q[:, i] = self.q[:, i] + self.qdot[:, i] * self.dt

            #     # =======================================================
            #     # 2. 순기구학(FK) 발바닥 위치 업데이트
            #     # =======================================================
            #     q1, q2, q3 = self.q[0, i], self.q[1, i], self.q[2, i]
                
            #     # [수정 4] 무릎 위치 (l1 사용, Z축은 cos)
            #     pb_knee[0, i] = -self.l1 * np.sin(q2)
            #     pb_knee[2, i] = -self.l1 * np.cos(q2)
                
            #     # 2D Pitch 평면에서의 발끝 위치 계산 (Z축은 cos)
            #     x_pitch = pb_knee[0, i] - self.l2 * np.sin(q2 + q3)
            #     z_pitch = pb_knee[2, i] - self.l2 * np.cos(q2 + q3) 
            #     y_pitch = self.lhip
                
            #     # [수정 5] Roll(X축) 회전 행렬 부호 수정
            #     Rroll = np.array([
            #         [1, 0, 0],
            #         [0, np.cos(q1), np.sin(q1)],
            #         [0, -np.sin(q1), np.cos(q1)]
            #     ])
                
            #     # Roll 회전 적용하여 최종 로컬 바디 기준 좌표 획득
            #     local_foot = Rroll @ np.array([x_pitch, y_pitch, z_pitch])
            #     pb_foot[:, i] = local_foot
            
            #     # [수정 5] 차원 덮어쓰기 방지 및 rw_feet 할당 차원 맞춤
            #     # 글로벌 발 위치 = 글로벌 CoM 위치 + 글로벌 회전 @ (힙 오프셋 + 로컬 발 위치)
            #     # self.pos는 (3,1), hip_offsets_body는 (4,3) 배열이라 가정하고 연산
            #     rw_feet[:, i] = self.pos.flatten() + self.R @ (self.hip_offsets_body[:, i] + local_foot)
