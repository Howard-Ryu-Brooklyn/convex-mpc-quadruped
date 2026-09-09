"""Convex MPC (Di Carlo et al. 스타일 SRBD + QP).

프레임 규약
    이 모듈이 다루는 벡터는 전부 **월드 정렬**이다. r_feet_W 는 CoM 을 원점으로
    한 월드 정렬 벡터이지 바디 프레임 좌표가 아니다. 이름의 _W 접미사가 그것을
    말한다 - 결함 1(Ac[0:3,6:9] 의 전치 누락)과 결함 5(절대/상대 위치 혼동)가
    모두 '이름에 프레임이 없어서' 눈으로 확인할 수 없었던 사고다.

상태 벡터 레이아웃 (13차원, RobotState.to_mpc_vector 가 정의)
    [0:3]   Θ = (roll, pitch, yaw)  ZYX 오일러각, **연속량**
    [3:6]   p_com_W
    [6:9]   omega_W
    [9:12]  v_com_W
    [12]    1.0  (중력을 선형 시스템에 넣기 위한 상수항)

    ★ yaw 가 연속량이라는 것이 중요하다. 참조 궤적을 current_yaw + omega*t 로
      쌓기 때문에, 접힌 yaw((-pi,pi])가 들어오면 반 바퀴에서 2pi 오차를 본다.
      MuJoCo 같은 소스는 angles.AngleUnwrapper 를 통과시켜야 한다.
"""
from __future__ import annotations

import numpy as np
import osqp
import scipy.sparse as sparse
from numpy.typing import NDArray

from quadruped_mpc.core.robot_types import ContactSeq4k, ForceVec12, StateTraj13k, StateVec13, Vec3, Vec3x4


class ConvexMPC:
    """SRBD 선형화 + QP 로 지면 반발력을 구한다.

    Attributes:
        k: horizon 길이 (스텝 수).
        mu: 마찰계수. 제약은 **사각뿔**이라 대각선 방향으로는 실효 sqrt(2)*mu
            까지 허용한다 (model_audit.pyramid_effective_mu 참조).
    """

    def __init__(
        self,
        m: float,
        I_body: list[float] | NDArray[np.float64],
        gz: float,
        dt: float,
        horizon: int,
        Lweights: NDArray[np.float64],
        Kweights: float,
        mu: float,
        fmin: float,
        fmax: float,
    ) -> None:
        """
        Args:
            m: 몸통 질량 [kg]. **실제 로봇 전체 질량이 아니다** - SRBD 모델이
                믿는 값이며, MuJoCo 와의 차이는 의도된 불확실성이다.
            I_body: 바디 프레임 관성 주 모멘트 [Ixx, Iyy, Izz] [kg m^2].
            gz: 중력 가속도 z 성분 [m/s^2]. 음수.
            dt: MPC 이산화 주기 [s]. 물리 스텝이 아니라 MPC 주기다.
            horizon: 예측 스텝 수 k.
            Lweights: (13,) 상태 오차 가중치. build_state_weight 로 만든다.
            Kweights: 제어 입력(힘) 정규화 가중치 스칼라.
            mu: 마찰계수.
            fmin: 접지 다리의 수직력 하한 [N]. 발이 떠버리는 해를 막는다.
            fmax: 수직력 상한 [N].
        """
        self.m = m
        self.I_body = np.diag(I_body)
        self.gz = gz
        self.dt = dt
        self.k = horizon
        self.mu = mu        # 마찰 계수 μ
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
    def skew_symmetric(v: Vec3) -> NDArray[np.float64]:
        """3D 벡터를 반대칭 행렬로 변환한다. [v]x @ w == np.cross(v, w).

        Args:
            v: (3,) 벡터.

        Returns:
            (3,3) 반대칭 행렬.
        """
        return np.array([
            [ 0,    -v[2],  v[1]],
            [ v[2],  0,    -v[0]],
            [-v[1],  v[0],  0   ]
        ])
    
    @staticmethod
    def build_state_weight(
        L_w_th: float, L_w_z: float, L_w_yr: float, L_w_v: float
    ) -> NDArray[np.float64]:
        """13차원 상태 오차 가중치 벡터를 만든다.

        0 으로 두는 항이 곧 **설계 선택**이다. x/y 위치[3:5]와 roll/pitch
        각속도[6:8]에 0 을 주는 것은 '절대 위치와 자세를 추종하지 않고 속도만
        추종한다'는 뜻이다. 그래서 S1 에서 3초에 yaw 가 1.14도 드리프트한다 -
        버그가 아니라 이 가중치의 귀결이다.

        Args:
            L_w_th: 자세(roll, pitch, yaw) 가중치.
            L_w_z: 높이 가중치.
            L_w_yr: yaw 각속도 가중치.
            L_w_v: 선속도 가중치. 정상상태 속도 오차를 이 값과 Kweights 의
                비가 결정한다 (S1 실측 -6.7%).

        Returns:
            (13,) 가중치 벡터.
        """
        weights = np.zeros(13)
        weights[0:3] = L_w_th  
        weights[3:5] = 0.0  
        weights[5]   = L_w_z 
        weights[6:8] = 0.0  
        weights[8]   = L_w_yr  
        weights[9:12] = L_w_v 
        weights[12] = 0.0
        return weights
    
    def _build_friction_cone(
        self, contact_sequence: ContactSeq4k
    ) -> tuple[sparse.csc_matrix, NDArray[np.float64], NDArray[np.float64]]:
        """마찰 사각뿔 제약 (C, u_min, u_max) 를 만든다.

        '원뿔'이라 부르지만 실제로는 원뿔에 **외접**하는 사각뿔이다:
        |fx| <= mu*fz, |fy| <= mu*fz 이므로 대각선 방향으로는 ||f_xy|| 가
        sqrt(2)*mu*fz 까지 허용된다. MuJoCo 는 진짜 원뿔(elliptic)이라
        그 차이가 대각선 방향 미끄러짐으로 나타난다 - 이것은 고치지 않고
        측정하는 대상이다 (model_audit.py 의 의도된 불확실성).

        Args:
            contact_sequence: (4,k) **구 규약** — 0=GROUND(접지), 1=AIR.
                내부에서 stance_flags = 1 - contact_sequence 로 되돌린다.
                TODO(Step 3): is_stance(bool) 를 직접 받아 이중 부정을 없앤다.
                HorizonPlan.contact_legacy 도 함께 제거된다.

        Returns:
            (C, u_min, u_max) — C 는 (5*4*k, 3*4*k) sparse, 경계는 (5*4*k,).
        """
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

    def get_discrete_matrices(
        self, yaw_W: float, r_feet_W: Vec3x4
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """선형화된 연속 시스템을 만들고 1차(오일러) 이산화한다.

        Args:
            yaw_W: 월드 yaw [rad]. **연속량이어야 한다** (모듈 독스트링 참조).
                Ad 는 이 값에만, Bd 는 이 값과 r_feet_W 둘 다에 의존한다.
            r_feet_W: (3,4) CoM 원점 기준 **월드 정렬** 발 위치. 바디 프레임이
                아니다. 이 벡터가 토크암 tau = r x f 를 만든다.

        Returns:
            (Ad, Bd) — (13,13) 과 (13,12).

        TODO(Step 7): 호출부는 Ad 와 Bd 를 각각 다른 인자로 필요로 하는데
            (Ad 는 평균 yaw 로 한 번, Bd 는 스텝마다) 이 함수는 매번 둘 다
            만든다. horizon 10 이면 11 번 호출하며 절반을 버린다.
            get_Ad(yaw) / get_Bd(yaw, r_feet_W) 로 나누면 연산이 절반이 된다.
            실시간성 측정 후에 다룬다.
        """
        Ac = np.zeros((13, 13))
        Bc = np.zeros((13, 12))
        
        cos_y, sin_y = np.cos(yaw_W), np.sin(yaw_W)
        Rz = np.array([
            [cos_y, -sin_y, 0],
            [sin_y,  cos_y, 0],
            [0,      0,     1]
        ])
        
        I_world_est = Rz @ self.I_body @ Rz.T
        I_world_est_inv = np.linalg.inv(I_world_est)

        # Θ̇ = R_z(ψ)ᵀ · ω   (roll/pitch 소각 근사)
        #
        #   유도: ω = ψ̇·ẑ + θ̇·(R_z ŷ) + φ̇·(R_z R_y x̂)
        #         θ≈φ≈0  ⇒  ω = R_z(ψ)·Θ̇  ⇒  Θ̇ = R_z(ψ)ᵀ·ω
        #
        # R_z와 R_zᵀ는 대각 성분(cos ψ)이 같고 비대각 성분(sin ψ)의 부호만
        # 다르다. 전치를 빠뜨리면 ψ=0에서만 우연히 맞고, ψ가 커질수록
        # roll/pitch 피드백에 부호가 반대인 교차항이 섞인다.
        # ψ=90°에서는 대각 성분이 0이 되어 피드백 전체의 부호가 반전되고,
        # 음의 되먹임이 양의 되먹임으로 바뀌어 지수 발산한다.
        #
        # 실측: ω_z=20°/s → ψ≈80°(t=4.0s)에서 개시, ψ≈90°(t=4.6s)부터
        #       roll이 0.2초마다 6배씩 증가. 10°/s에서는 같은 ψ에 도달하는
        #       t=8s로 밀림 → 발산은 각속도가 아니라 누적 yaw 각도에 의존.
        # 플랜트(dynamics.py의 R_rate2euler)는 pitch=0에서 R_z(ψ)ᵀ로
        # 올바르게 구현되어 있다. 틀린 쪽은 MPC 모델이었다.
        Ac[0:3, 6:9] = Rz.T
        Ac[3:6, 9:12] = np.eye(3)
        Ac[9:12, 12] = np.array([0, 0, self.gz]) 

        inv_mass_eye = np.eye(3) / self.m
        
        for i in range(4):
            r_i = r_feet_W[:, i]
            r_skew = self.skew_symmetric(r_i)
            
            Bc[6:9, i*3:(i+1)*3] = I_world_est_inv @ r_skew
            Bc[9:12, i*3:(i+1)*3] = inv_mass_eye

        Ad_hat = np.eye(13) + Ac * self.dt
        Bd_hat = Bc * self.dt
        
        return Ad_hat, Bd_hat

    def build_qp_matrices(
        self, Ad_hat: NDArray[np.float64], Bd_hat_list: list[NDArray[np.float64]]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """horizon 전체의 예측 행렬 (Aqp, Bqp) 를 조립한다.

            X = Aqp @ x0 + Bqp @ U

        점화식으로 쌓으므로 k 에 대해 O(k^2) 블록이지만 재계산은 없다.

        Args:
            Ad_hat: (13,13) 이산 상태 행렬. horizon 전체에 같은 것을 쓴다.
            Bd_hat_list: 스텝별 (13,12) 입력 행렬 k 개.

        Returns:
            (Aqp, Bqp) — (13k,13) 과 (13k,12k).
        """
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

    def solve(
        self,
        x0: StateVec13,
        x_ref_traj: StateTraj13k,
        yaw_traj: NDArray[np.float64],
        r_feet_traj_W: list[Vec3x4],
        contact_sequence: ContactSeq4k,
    ) -> tuple[ForceVec12, NDArray[np.float64] | float]:
        """QP 를 한 번 풀어 현재 스텝의 지면 반발력을 돌려준다.

        Args:
            x0: (13,1) 현재 상태. RobotState.to_mpc_vector() 레이아웃.
            x_ref_traj: (13k,1) 참조 궤적.
            yaw_traj: (k,) 스텝별 예측 yaw [rad]. 연속량.
            r_feet_traj_W: 스텝별 (3,4) 토크암 k 개. HorizonPlan 이 만든다 -
                스탠스는 실제 접지 위치, 스윙은 착지 목표점 (결함 8).
            contact_sequence: (4,k) 구 규약 (0=GROUND, 1=AIR).

        Returns:
            (forces, u_max) — forces 는 (12,) [fx fy fz] x 4.

            ★ 두 번째 값은 float 가 아니라 (5*4*k,) 배열이다 (k=10 이면 200개).
              이름이 u_max 라 '최대 힘'으로 읽히지만 실제로는 _build_friction_cone
              이 만든 **QP 제약 경계 벡터 전체**다. 이름과 실체가 다르고,
              호출부는 아무도 쓰지 않는다 (`fmpc, _ = mpc.solve(...)`).

              mypy 가 이것을 잡았다. 나는 이름을 보고 타입을 float 로 썼고
              검사기가 실체를 보고 반박했다. '쓰이지 않아서 아무도 몰랐다'는
              결함 11(self.q), 결함 12(클립 구간 속도)와 같은 형태다.

              ★★ 더 나쁜 것: 두 반환 경로의 타입이 다르다.
                 성공: (ndarray(12), ndarray(5*4*k))
                 실패: (ndarray(12), 0.0)          <- float
                 실패 경로에서만 조용히 스칼라가 된다. 이 값을 쓰는 코드가
                 있었다면 정상 동작 중에는 배열을, 실패 순간에만 스칼라를
                 받았을 것이고, 그 차이는 '가장 확인하기 어려운 순간'에
                 나타난다. 지금은 아무도 안 써서 드러나지 않았을 뿐이다.

                 이것도 mypy 가 잡았다 - 그것도 **첫 번째 거짓말을 고친 뒤에**
                 드러났다. 타입을 사실에 맞추자 두 경로의 불일치가 남았고,
                 검사기는 그것을 다시 지적했다. 거짓말 하나가 다른 거짓말을
                 가리고 있었던 셈이다.

              TODO(Step 3): 반환에서 제거한다. MPCSolution 을 도입하면서
              필요한 진단값(솔버 상태, 반복 횟수, 풀이 시간)으로 대체하면
              성공/실패가 같은 타입의 값이 된다.
              지금 지우면 시그니처 변경이라 Step 2(동작 무변경) 범위 밖이다.

        ★ 실패를 반환값으로 구별할 수 없다.
          실패 시 np.zeros(12) 를 돌려주는데, 이것은 '네 다리가 동시에 힘을
          놓는다'는 뜻이라 실기에서 가장 위험한 동작이다. 호출부는 성공/실패를
          구별할 방법이 없다.
          TODO(Step 3): SolverStatus 를 담은 MPCSolution 을 반환한다.
          robot_types.SolverStatus 는 이미 정의되어 있으나 쓰이지 않는다.

        구현 메모 — Ad 에 평균 yaw 를 쓴다
            Ad_hat 은 np.mean(yaw_traj) 하나로 만들어 horizon 전체에 재사용한다.
            엄밀히는 스텝마다 달라야 하지만, Aqp 를 점화식으로 쌓으려면 Ad 가
            상수여야 O(k) 로 조립된다. yaw 변화가 horizon 내에서 작다는 가정이며
            (20 deg/s x 0.33s = 6.6도), 큰 회전에서는 근사가 나빠진다.
        """
        C_mat, u_min, u_max = self._build_friction_cone(contact_sequence)
        Ad_hat, _ = self.get_discrete_matrices(float(np.mean(yaw_traj)), r_feet_traj_W[0])
        Bd_hat_list = [
            self.get_discrete_matrices(yaw_traj[i], r_feet_traj_W[i])[1]
            for i in range(self.k)
        ]
        
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

        if res.info.status_val != osqp.constant('OSQP_SOLVED'):
            print(f"⚠️ QP Solver Failed! Status: {res.info.status}")
            # TODO(Step 3): 실패를 MPCSolution 값으로 표현할 것.
            #   np.zeros(12)는 네 다리가 동시에 힘을 놓는다는 뜻이고 실기에서
            #   가장 위험한 동작이다. 안전한 fallback은 직전 유효 해 또는
            #   중력 보상 균등 분배이며, 그 판단은 호출자의 몫이다.
            #   동작을 바꾸는 수정이므로 Step 0 범위 밖에 둔다.
            return np.zeros(12), 0.0

        return res.x[:12], u_max