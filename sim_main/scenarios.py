"""테스트 시나리오 정의. 여기가 유일한 출처(single source of truth)입니다."""

SCENARIOS = {
    "S0_standing":  dict(gait_name="standing", v_des_x=0.0, omega_z_deg_s=0.0,  duration_s=0.5),
    "S1_trot_fwd":  dict(gait_name="trotting", v_des_x=1.0, omega_z_deg_s=0.0,  duration_s=3.0),
    "S2_endurance": dict(gait_name="trotting", v_des_x=1.0, omega_z_deg_s=0.0,  duration_s=10.0),
    "S3_yaw":       dict(gait_name="trotting", v_des_x=0.0, omega_z_deg_s=20.0, duration_s=6.0),
}