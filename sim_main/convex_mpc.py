import numpy as np
import scipy.sparse as sparse
import osqp

class ConvexMPC:
    def __init__(self, m, I_body, gz, dt, horizon, Lweights, Kweights, mu, fmin, fmax):
        self.m = m
        self.I_body = np.diag(I_body)
        self.gz = gz
        self.dt = dt
        self.k = horizon
        self.mu = mu
        self.fmin = fmin
        self.fmax = fmax
        
        self.L = np.diag(Lweights) 
        self.K = np.eye(12) * Kweights 
        
        # 전체 QP 가중치 행렬 미리 구성 (크기 불변)
        self.Lqp = np.kron(np.eye(self.k), self.L)
        self.Kqp = np.kron(np.eye(self.k), self.K)
        
        # OSQP 솔버 인스턴스 생성
        self.prob = osqp.OSQP()
        self.solver_initialized = False

        # --- 마찰 원뿔(Friction Cone) 제약조건 행렬 미리 구성 ---
        

    @staticmethod
    def skew_symmetric(v):
        """ 3D 벡터를 반대칭 행렬(Skew-symmetric matrix)로 변환 """
        return np.array([
            [ 0,    -v[2],  v[1]],
            [ v[2],  0,    -v[0]],
            [-v[1],  v[0],  0   ]
        ])
    
    @staticmethod
    def build_state_weight(L_w_th, L_w_z, L_w_yr, L_w_v):
        """ 사용자 편의를 위한 13차원 가중치 벡터 생성 헬퍼 함수 """
        weights = np.zeros(13)
        weights[0:3] = L_w_th  
        weights[3:5] = 0.0  
        weights[5]   = L_w_z 
        weights[6:8] = 0.0  
        weights[8]   = L_w_yr  
        weights[9:12] = L_w_v 
        weights[12] = 0.0
        return weights
    
    def _build_friction_cone(self, contact_sequence):
        """ 완벽한 피라미드 마찰 원뿔 제약조건 생성 """
        # fmin <= fz <= fmax
        # -µfz <= fx <= µfz
        # -µfz <= fy <= µfz
        # C [fx fy fz]T
        # fx - µfz <= 0
        # -fx - µfz <= 0
        # fy - µfz <= 0
        # -fy - µfz <= 0
        
        C_leg = np.array([
            [ 1,  0, -self.mu], 
            [-1,  0, -self.mu], 
            [ 0,  1, -self.mu], 
            [ 0, -1, -self.mu], 
            [ 0,  0,  1      ]  
        ]) # 5X3 
        # 4=n -> leg dimension
        C_step = np.kron(np.eye(4), C_leg) # 5n X 3n
        C_total = np.kron(np.eye(self.k), C_step) # 5nk X 3nk
        # U \in R^(3nk x 1)
        
        # u_max_leg = np.array([0, 0, 0, 0, self.fmax]) # 5 X 1
        # u_min_leg = np.array([-np.inf, -np.inf, -np.inf, -np.inf, self.fmin])
        
        # u_max = np.tile(u_max_leg, 4 * self.k) # 5nk X 1
        # u_min = np.tile(u_min_leg, 4 * self.k)

        # 2. 수학적 스위치 생성 (AIR: 1 -> Stance: 0 / GROUND: 0 -> Stance: 1)
        stance_flags = 1.0 - contact_sequence  # shape: (4, k)
        
        # 3. 빈 Bound 배열 생성 (k스텝, 4다리, 5개 제약조건)
        # 나중에 flatten()으로 1차원 벡터로 쫙 펼칠 예정입니다.
        u_max_3d = np.zeros((self.k, 4, 5))
        u_min_3d = np.full((self.k, 4, 5), -np.inf) # 기본값을 -inf로 채움
        
        # 4. Vectorized Bounds 할당 (for문 없이 한 번에 계산)
        # 4-1. 마찰 원뿔 조건 (0~3번째 행): 항상 상한은 0, 하한은 -inf
        u_max_3d[:, :, 0:4] = 0.0
        
        # 4-2. Z축 수직 항력 조건 (4번째 행): Stance Flag를 곱해서 공중이면 0으로 강제
        # stance_flags.T 는 shape이 (k, 4)가 됩니다.
        u_max_3d[:, :, 4] = self.fmax * stance_flags.T
        u_min_3d[:, :, 4] = self.fmin * stance_flags.T
        
        # 5. QP 솔버에 넣기 위해 1차원 벡터로 변환
        u_max = u_max_3d.flatten()
        u_min = u_min_3d.flatten()
        
        # Sparse한 행렬의 0이 아닌 부분만 메모리에 저장
        return sparse.csc_matrix(C_total), u_min, u_max

    def get_discrete_matrices(self, yaw, r_feet):
        """ Ac, Bc 생성 및 이산화 (ZOH 1차 근사) """
        Ac = np.zeros((13, 13))
        Bc = np.zeros((13, 12))
        
        cos_y, sin_y = np.cos(yaw), np.sin(yaw)
        Rz = np.array([
            [cos_y, -sin_y, 0],
            [sin_y,  cos_y, 0],
            [0,      0,     1]
        ])
        
        I_world_est = Rz @ self.I_body @ Rz.T
        I_world_est_inv = np.linalg.inv(I_world_est)

        Ac[0:3, 6:9] = Rz
        Ac[3:6, 9:12] = np.eye(3)
        Ac[9:12, 12] = np.array([0, 0, self.gz]) 

        inv_mass_eye = np.eye(3) / self.m
        
        for i in range(4):
            r_i = r_feet[:, i] 
            r_skew = self.skew_symmetric(r_i)
            
            Bc[6:9, i*3:(i+1)*3] = I_world_est_inv @ r_skew
            Bc[9:12, i*3:(i+1)*3] = inv_mass_eye

        Ad_hat = np.eye(13) + Ac * self.dt
        Bd_hat = Bc * self.dt
        
        return Ad_hat, Bd_hat

    def build_qp_matrices(self, Ad_hat, Bd_hat_list):
        """ Aqp, Bqp 행렬 조립 (DP 적용 완료) """
        k = self.k
        Aqp = np.zeros((13 * k, 13))
        Bqp = np.zeros((13 * k, 12 * k))

        Aqp[0:13, :] = Ad_hat
        Bqp[0:13, 0:12] = Bd_hat_list[0]

        for i in range(1, k):
            r_start, r_end   = 13 * i, 13 * (i + 1)
            r_prev_start, r_prev_end = 13 * (i - 1), 13 * i

            Aqp[r_start:r_end, :] = Ad_hat @ Aqp[r_prev_start:r_prev_end, :]
            
            c_prev_end = 12 * i
            Bqp[r_start:r_end, 0:c_prev_end] = Ad_hat @ Bqp[r_prev_start:r_prev_end, 0:c_prev_end]
            
            c_start, c_end = 12 * i, 12 * (i + 1)
            Bqp[r_start:r_end, c_start:c_end] = Bd_hat_list[i]

        return Aqp, Bqp

    def solve(self, x0, x_ref_traj, yaw_traj, r_feet_traj, contact_sequence):
        """ 메인 제어 함수 """
        C_mat, u_min, u_max = self._build_friction_cone(contact_sequence)
        Ad_hat, _ = self.get_discrete_matrices(np.mean(yaw_traj), r_feet_traj[0])
        Bd_hat_list = [self.get_discrete_matrices(yaw_traj[i], r_feet_traj[i])[1] for i in range(self.k)]
        
        Aqp, Bqp = self.build_qp_matrices(Ad_hat, Bd_hat_list)
        
        # 3. 비용 함수 H, g 구성
        H = 2.0 * (Bqp.T @ self.Lqp @ Bqp + self.Kqp)
        
        # OSQP를 위해 절반(상삼각)만 떼어내고 구조 고정하기
        H_upper = np.triu(H) 
        
        # 0이 발생해 메모리 칸 개수(6652개)가 바뀌는 것을 방지하는 트릭
        H_upper[np.triu_indices_from(H_upper)] += 1e-9
        
        # 이제 완벽하게 고정된 상삼각 행렬을 Sparse 형태로 변환
        H_sparse = sparse.csc_matrix(H_upper)

        # g 벡터 구성 (차원 꼬임 방지를 위해 flatten 유지)
        X_ref = np.concatenate(x_ref_traj).flatten()
        x0_flat = x0.flatten()
        g = (2.0 * Bqp.T @ self.Lqp @ (Aqp @ x0_flat - X_ref)).flatten()

        # 5. OSQP 풀이
        if not self.solver_initialized:
            self.prob.setup(P=H_sparse, q=g, A=C_mat, l=u_min, u=u_max, 
                            warm_start=True, verbose=False)
            self.solver_initialized = True
        else:
            # 6652개의 사이즈가 완벽히 일치하므로 초고속으로 업데이트 됨
            self.prob.update(q=g, Px=H_sparse.data, l=u_min, u=u_max)

        # 최적화 풀이
        res = self.prob.solve()
        
        if res.info.status != 'solved':
            # 실패 시 안전을 위해 지면 반발력 0 (또는 이전 값) 반환
            print(f"QP Solver Failed! Status: {res.info.status}")
            return np.zeros(12)

        return res.x[:12], u_max