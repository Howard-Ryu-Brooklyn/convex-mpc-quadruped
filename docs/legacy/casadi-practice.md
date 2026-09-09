# CasADi 연습 (초기 학습)


> 원본: `mpc_casadi_practice.ipynb` — 저장소 정리 과정에서 노트북을 버리고 내용만 옮겼다.


```python
import casadi as ca
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. 시뮬레이션 및 시스템 파라미터 설정
# ==========================================
N = 30           # 예측 호라이즌 (Prediction Horizon)
dt = 0.1         # 샘플링 시간 (초)
m = 1.0          # 수레의 질량 (kg)

u_max = 10.0     # 최대 제어 입력 (힘 제한)
v_max = 2.0      # 최대 속도 (상태 제한)

# ==========================================
# 2. CasADi Opti 스택 초기화 (최적화 문제 생성기)
# ==========================================
opti = ca.Opti()

# ==========================================
# 3. 최적화 변수 선언
# ==========================================
# X: 상태 변수 행렬 (2 x (N+1)) -> [위치 p; 속도 v]
X = opti.variable(2, N + 1)
p = X[0, :]
v = X[1, :]

# U: 제어 입력 변수 행렬 (1 x N) -> [힘 F]
U = opti.variable(1, N)

# ==========================================
# 4. 초기 상태 파라미터화 (실시간 제어 루프를 위해)
# ==========================================
# 초기 조건은 고정된 값이 아니라 매 제어 주기마다 업데이트되어야 하므로 parameter로 설정
x0 = opti.parameter(2, 1)
opti.subject_to(X[:, 0] == x0) # 첫 번째 상태는 현재 상태와 같아야 함

# ==========================================
# 5. 제약조건 (Constraints) 설정
# ==========================================
# 5-1. 동역학 제약조건 (Multiple Shooting 방식)
for k in range(N):
    # 다음 상태 예측 (오일러 적분: x_{k+1} = x_k + x_dot * dt)
    # 위치 변화: p_{k+1} = p_k + v_k * dt
    # 속도 변화: v_{k+1} = v_k + (F_k / m) * dt
    x_next = X[:, k] + dt * ca.vertcat(X[1, k], U[:, k] / m)
    opti.subject_to(X[:, k+1] == x_next)

# 5-2. 물리적 한계 제약조건 (Inequality Constraints)
opti.subject_to(opti.bounded(-u_max, U, u_max)) # 제어 입력(힘) 제한
opti.subject_to(opti.bounded(-v_max, v, v_max)) # 최고/최저 속도 제한

# ==========================================
# 6. 비용 함수 (Cost Function) 설정
# ==========================================
Q = ca.diag([10.0, 1.0]) # 상태 오차 가중치 (위치 오차를 더 중요하게 취급)
R = 5                  # 제어 입력 가중치 (에너지 소모 최소화)
x_ref = ca.vertcat(10.0, 0.0) # 목표 상태: 위치 10m, 속도 0m/s (정지)

cost = 0
for k in range(N):
    err = X[:, k] - x_ref
    # J = x^T Q x + u^T R u
    cost += ca.mtimes([err.T, Q, err]) + R * U[:, k]**2

# 터미널 비용 (마지막 상태에 대한 비용 추가)
err_N = X[:, N] - x_ref
cost += ca.mtimes([err_N.T, Q, err_N])

opti.minimize(cost) # 비용 함수 최소화 선언

# ==========================================
# 7. 솔버(Solver) 설정
# ==========================================
# 비선형 최적화 솔버인 IPOPT 사용 (CasADi 기본)
# print_level 등을 0으로 설정하여 콘솔 출력 최소화
p_opts = {"expand": True}
s_opts = {"max_iter": 100, "print_level": 0, "sb": "yes", "print_timing_statistics": "no"}
opti.solver('ipopt', p_opts, s_opts)

# ==========================================
# 8. 실시간 MPC 제어 루프 (Receding Horizon)
# ==========================================
current_x = np.array([0.0, 0.0]) # 실제 로봇(수레)의 현재 상태
t_sim = 0.0
t_end = 8.0                      # 총 8초 시뮬레이션 진행

# 결과 저장을 위한 리스트
history_p = [current_x[0]]
history_v = [current_x[1]]
history_u = []
t_steps = [0.0]

while t_sim < t_end:
    # 1. 현재 로봇 상태를 솔버의 초기 조건 파라미터로 세팅
    opti.set_value(x0, current_x)
    
    try:
        sol = opti.solve()
        
        # 2. 최적화된 입력 시퀀스 중 "첫 번째(u_0)" 제어 입력만 추출! (MPC의 핵심)
        u_opt_seq = sol.value(U)
        u_apply = u_opt_seq[0] if isinstance(u_opt_seq, np.ndarray) else u_opt_seq
        
    except Exception as e:
        print(f"솔버 에러 (t={t_sim:.2f}):", e)
        u_apply = 0.0
        
    # 3. 추출한 입력을 실제 시스템에 적용하여 다음 상태 계산 (물리 엔진 역할)
    # x_{k+1} = x_k + x_dot * dt
    current_x[0] += current_x[1] * dt
    current_x[1] += (u_apply / m) * dt
    
    # 4. 데이터 저장 및 시간 업데이트
    history_p.append(current_x[0])
    history_v.append(current_x[1])
    history_u.append(u_apply)
    t_sim += dt
    t_steps.append(t_sim)

# ==========================================
# 9. 결과 시각화
# ==========================================
print(f"최종 도달 위치: {history_p[-1]:.2f}m")

plt.figure(figsize=(10, 8))

plt.subplot(3, 1, 1)
plt.plot(t_steps, history_p, 'b-', label='Actual Position (p)')
plt.axhline(10.0, color='r', linestyle='--', label='Reference (10m)')
plt.ylabel('Position [m]')
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(t_steps, history_v, 'g-', label='Actual Velocity (v)')
plt.axhline(v_max, color='r', linestyle='--', label='Speed Limit')
plt.axhline(-v_max, color='r', linestyle='--')
plt.ylabel('Velocity [m/s]')
plt.legend()

plt.subplot(3, 1, 3)
plt.step(t_steps[:-1], history_u, 'k-', where='post', label='Applied Force (F)')
plt.axhline(u_max, color='r', linestyle='--', label='Force Limit')
plt.axhline(-u_max, color='r', linestyle='--')
plt.ylabel('Force [N]')
plt.xlabel('Time [s]')
plt.legend()

plt.tight_layout()
plt.show()
```
