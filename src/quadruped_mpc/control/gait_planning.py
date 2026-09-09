import matplotlib.pyplot as plt
import numpy as np

# --- 1. 상수 정의 ---
AIR = 1
GROUND = 0
FR, FL, RR, RL = 0, 1, 2, 3

# --- 2. 보행 모드 파라미터 딕셔너리 ---
GAIT_PARAMS = {
    "standing": {
        "duty_cycle": 1.0, # 보행 주기 내에 지면에 발이 닫는 비율
        "phase_offset": [0.0, 0.0, 0.0, 0.0], # 각 다리별 보행 사이클 오프셋
        "cycle_time": 0.5, # 보행 1 사이클 주기 "
        "description": "정지 상태 (모든 다리가 항상 지지)"
    },
    "trotting": {
        "duty_cycle": 0.5,
        "phase_offset": [0.0, 0.5, 0.5, 0.0],
        "cycle_time": 0.5,
        "description": "대각선 다리가 교차하는 2박자 보행"
    },
    "flying_trot": {
        "duty_cycle": 0.35,
        "phase_offset": [0.0, 0.5, 0.5, 0.0],
        "cycle_time": 0.4,
        "description": "체공 구간이 있는 빠른 트로팅"
    },
    "bounding": {
        "duty_cycle": 0.4,
        "phase_offset": [0.0, 0.0, 0.5, 0.5],
        "cycle_time": 0.33,
        "description": "앞다리 쌍과 뒷다리 쌍이 번갈아 도약"
    },
    "galloping": {
        "duty_cycle": 0.25,
        "phase_offset": [0.0, 0.1, 0.5, 0.6],
        "cycle_time": 0.3,
        "description": "가장 빠른 4박자 동적 보행 (Rotary gallop)"
    }
}

# --- 3. 제어 함수 ---
def get_gait_parameters(gait_name):
    """
    보행 모드 이름을 입력받아 duty_cycle과 phase_offset 배열을 반환합니다.
    """
    if gait_name not in GAIT_PARAMS:
        raise ValueError(f"지원하지 않는 보행 모드입니다: {gait_name}. 사용 가능한 모드: {list(GAIT_PARAMS.keys())}")
    
    params = GAIT_PARAMS[gait_name]
    return params["cycle_time"], params["duty_cycle"], params["phase_offset"]

def get_contact_state(global_phase, duty_cycle, phase_offsets):
    """
    현재 전체 위상(global_phase, 0.0~1.0)과 보행 파라미터를 받아
    네 다리의 접촉 상태 [FR, FL, RR, RL] 를 반환합니다.
    """
    sa = [GROUND, GROUND, GROUND, GROUND]
    
    for leg in range(4):
        # 각 다리의 위상(offset 반영)을 0.0 ~ 1.0 사이로 정규화하여 계산
        leg_phase = (global_phase + phase_offsets[leg]) % 1.0
        
        # 계산된 다리 위상이 duty_cycle 이내면 지지(GROUND), 넘어가면 체공(AIR)
        if leg_phase < duty_cycle:
            sa[leg] = GROUND
        else:
            sa[leg] = AIR
            
    return sa


# --- 2. 시각화 함수 ---
def plot_gait_pattern(history_Sa, dt=1.0):
    """
    시간에 따른 다리 접촉 상태(history_Sa)를 입력받아 Gait Pattern을 그립니다.
    - history_Sa: [[FR, FL, RR, RL], [FR, FL, RR, RL], ...] 형태의 리스트
    - dt: 한 스텝당 시간 (기본값 1.0을 쓰면 x축이 스텝 인덱스로 표시됨)
    """
    sa_array = np.array(history_Sa)
    time = np.arange(len(sa_array)) * dt
    
    leg_names = ['FR', 'FL', 'RR', 'RL']
    
    fig, ax = plt.subplots(figsize=(10, 4))
    
    # 각 다리(0~3)에 대해 가시화
    for leg_idx in range(4):
        # 상태가 GROUND(0)일 때 True가 되도록 마스킹 (지면에 닿아있을 때만 색칠하기 위함)
        is_stance = (sa_array[:, leg_idx] == GROUND)
        
        # 막대 그래프처럼 보이도록 y축 중심(leg_idx) 위아래로 0.3만큼 색을 채움
        ax.fill_between(time, 
                        y1 = leg_idx - 0.3, 
                        y2 = leg_idx + 0.3, 
                        where = is_stance, 
                        color = f'C{leg_idx}', 
                        alpha = 0.8,
                        label = leg_names[leg_idx] if np.any(is_stance) else "")
        
    ax.set_yticks(range(4))
    ax.set_yticklabels(leg_names)
    ax.invert_yaxis()  # 직관적으로 FR이 맨 위에 오도록 y축 반전
    
    ax.set_xlabel('Time (s)' if dt != 1.0 else 'Simulation Step')
    ax.set_ylabel('Leg')
    ax.set_title('Quadruped Gait Contact Pattern (Colored Block = Stance Phase)')
    ax.grid(True, axis='x', linestyle='--', alpha=0.7)
    
    # 레이아웃 정돈 및 출력
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    gait_mode = ["standing", "trotting", "flying_trot", "bounding","galloping"]

    cycle_time, duty, phase_offset = get_gait_parameters(gait_mode[0])

    sim_freq = 100
    sim_dt = 1/sim_freq
    sim_tf = 1
    sim_cnt = sim_freq * sim_tf

    gait_freq = 1/cycle_time
    gait_dt = cycle_time
    
    history_Sa = []

    for i in range(sim_cnt):
    # 3. 현재 Phase에서의 접촉 상태 계산
        sim_time = i*sim_dt
        current_phase = (sim_time % cycle_time) / cycle_time
        Sa_current = get_contact_state(current_phase, duty, phase_offset)


        history_Sa.append(Sa_current)

    plot_gait_pattern(history_Sa, dt=sim_dt)