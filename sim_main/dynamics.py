import numpy as np

# utils
DEG2RAD = np.pi / 180

class SRBDynamics:
    def __init__(self, dt, mass, inertia_diag, gz, init_pos, init_vel, init_ang, init_angvel):
        """
        단일 강체 동역학(SRBD) 시뮬레이터 초기화
        """
        self.dt = dt
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
        self.R = np.eye(3)

    # ==========================================
    # ✅ 추가: 외부(main.py / MPC) 피드백용 속성(Property)
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

    @staticmethod
    def get_rotation_matrix(roll, pitch, yaw):
        """오일러 각(Z-Y-X) 기반 회전 행렬 계산"""
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(roll), -np.sin(roll)],
                       [0, np.sin(roll), np.cos(roll)]])
        Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                       [0, 1, 0],
                       [-np.sin(pitch), 0, np.cos(pitch)]])
        Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                       [np.sin(yaw), np.cos(yaw), 0],
                       [0, 0, 1]])
        return Rz @ Ry @ Rx

    def step(self, F_G, rb_feet):
        """
        [입력] F_G: 4족 지면 반발력 (3x4)
        [입력] rb_feet: 바디 기준 발 로컬 좌표 (3x4)
        [출력] 업데이트된 상태 변수들
        """
        # 1. 행렬 업데이트 (회전 변환 및 관성 텐서)
        roll, pitch, yaw = self.ang[0, 0], self.ang[1, 0], self.ang[2, 0]
        self.R = self.get_rotation_matrix(roll, pitch, yaw)
        IW = self.R @ self.IB @ self.R.T
        IW_INV = np.linalg.inv(IW)
        
        # 2. 발 위치 월드 좌표 변환
        rw_feet = self.R @ rb_feet
        # (Z축은 바닥 평면상에 있다고 가정하는 로직)
        rw_feet[2, :] = -self.pos[2, 0] 

        # 3. 병진 운동 동역학 (Translation Dynamics)
        TotalForce = np.sum(F_G, axis=1, keepdims=True)
        self.acc = TotalForce / self.m + self.g_vec

        # 4. 회전 운동 동역학 (Rotational Dynamics)
        T_com = np.zeros((3, 1))
        for i in range(4):
            torque_i = np.cross(rw_feet[:, i], F_G[:, i])
            T_com += torque_i.reshape(3, 1)
            
        gyro_term = np.cross(self.ang_vel.flatten(), (IW @ self.ang_vel).flatten()).reshape(3, 1)
        self.ang_acc = IW_INV @ (T_com - gyro_term)

        # 5. 수치 적분 (Euler Integration)
        self.vel += self.acc * self.dt
        self.pos += self.vel * self.dt
        self.ang_vel += self.ang_acc * self.dt

        # 6. 각속도 -> 오일러 각 변화율 변환 (짐벌락 방지)
        pitch_safe = np.clip(pitch, -89 * DEG2RAD, 89 * DEG2RAD)
        R_rate2euler = np.array([
            [np.cos(yaw) / np.cos(pitch_safe), np.sin(yaw) / np.cos(pitch_safe), 0],
            [-np.sin(yaw),                     np.cos(yaw),                      0],
            [np.cos(yaw) * np.tan(pitch_safe), np.sin(yaw) * np.tan(pitch_safe), 1]
        ])
        
        self.ang += R_rate2euler @ self.ang_vel * self.dt

        # 로깅 및 시각화용 데이터 반환
        return self.pos.copy(), self.R.copy(), rw_feet.copy()