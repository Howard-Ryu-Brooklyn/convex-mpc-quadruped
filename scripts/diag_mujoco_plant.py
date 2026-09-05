"""MuJoCoPlant 관측 경계 진단 (1-7d-1).

step() 을 쓰기 전에 **관측이 옳은지** 먼저 확인한다. 관측이 틀린 채로 제어를
붙이면, 나중에 나오는 실패가 제어 탓인지 관측 탓인지 구별할 수 없다.

각 항목은 '통과/실패'가 아니라 **숫자**를 낸다. 물리량의 크기를 보고 사람이
판단해야 하는 것들이기 때문이다.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "sim_main"))

import config as cfg                      # noqa: E402
from mujoco_plant import MuJoCoPlant       # noqa: E402
from rotations import rpy_to_matrix        # noqa: E402

XML = ROOT / "mujoco_test" / "mit_cheetah3" / "scene.xml"
HOME_QPOS = np.array([
    0.0, 0.0, 0.65,
    1.0, 0.0, 0.0, 0.0,
    0.0, 0.9, -1.8,
    0.0, 0.9, -1.8,
    0.0, 0.9, -1.8,
    0.0, 0.9, -1.8,
])


def main() -> None:
    plant = MuJoCoPlant(XML, dt=1.0 / 9000, home_qpos=HOME_QPOS)
    mj, m, d = plant._mj, plant._model, plant._data

    print("\n" + "=" * 68)
    print("1. 규약 차이 — 없앴는지 확인")
    print("=" * 68)
    trunk = plant._trunk_id
    origin = np.asarray(d.qpos[0:3])
    com = np.asarray(d.subtree_com[trunk])
    print(f"  trunk 원점  {np.round(origin, 4)}")
    print(f"  CoM        {np.round(com, 4)}")
    print(f"  차이       {np.round((com - origin) * 1000, 2)} mm"
          f"   <- qpos[0:3] 을 CoM 으로 쓰면 이만큼 토크암이 틀린다")

    site_z = np.array([d.site_xpos[i][2] for i in plant._site_ids])
    contact_z = plant.feet_positions_W()[2, :]
    print(f"\n  발 site z    {np.round(site_z, 5)}   (구의 중심)")
    print(f"  접촉점 z     {np.round(contact_z, 5)}   (site - 반지름)")
    print(f"  발 반지름    {plant.facts.foot_sphere_radius_m} m")
    print(f"  지면 정렬 후 최저 접촉점 = {contact_z.min():+.6f} m  <- 0 이어야 한다")

    print("\n" + "=" * 68)
    print("2. 관측 일관성 — observe() 가 스스로와 맞는가")
    print("=" * 68)
    s = plant.observe()
    R_mj = np.asarray(d.xmat[trunk]).reshape(3, 3)
    R_rt = rpy_to_matrix(*s.rpy_W)
    print(f"  rpy -> R 재구성 오차            {np.abs(R_rt - R_mj).max():.2e}")
    s2 = plant.observe()
    print(f"  observe() 멱등성 (2회 호출 차)  "
          f"{np.abs(s2.to_mpc_vector() - s.to_mpc_vector()).max():.2e}")
    print(f"  상태: rpy {np.round(np.rad2deg(s.rpy_W), 3)} deg,  z {s.p_com_W[2]:.4f} m")
    print(f"        v {np.round(s.v_com_W, 5)},  omega {np.round(s.omega_W, 5)}")

    print("\n" + "=" * 68)
    print("3. 자유낙하 — 관측이 물리와 맞는가 (제어 없이)")
    print("=" * 68)
    # 중력만 작용시키면 CoM 은 정확히 -g 로 가속해야 한다.
    mj.mj_resetData(m, d)
    d.qpos[:] = HOME_QPOS
    d.qpos[2] += 0.5                       # 공중으로 띄운다
    mj.mj_forward(m, d)
    plant._yaw.reset()
    z0 = float(plant.observe().p_com_W[2])
    n = 900                                 # 0.1 s
    for _ in range(n):
        d.ctrl[:] = 0.0
        mj.mj_step(m, d)
    s_end = plant.observe()
    t = n * plant.dt
    z_expect = z0 - 0.5 * abs(cfg.gz) * t ** 2
    print(f"  {t*1000:.0f} ms 자유낙하:  측정 dz {s_end.p_com_W[2]-z0:+.6f} m"
          f"   이론 {z_expect-z0:+.6f} m   오차 {abs(s_end.p_com_W[2]-z_expect)*1000:.3f} mm")
    print(f"  낙하 속도 vz {s_end.v_com_W[2]:+.4f} m/s   이론 {-abs(cfg.gz)*t:+.4f} m/s")
    print("  (다리가 관성으로 흔들리므로 CoM 은 완전히 강체가 아니다. mm 수준이면 정상)")

    print("\n" + "=" * 68)
    print("4. yaw unwrap — 몸통을 강제로 3바퀴 돌린다")
    print("=" * 68)
    mj.mj_resetData(m, d)
    d.qpos[:] = HOME_QPOS
    plant._yaw.reset()
    truth, got = [], []
    for k in range(1081):                   # 0 ~ 1080 deg
        ang = np.deg2rad(k)
        q = np.zeros(4)
        mj.mju_axisAngle2Quat(q, np.array([0.0, 0.0, 1.0]), ang)
        d.qpos[3:7] = q
        mj.mj_forward(m, d)
        truth.append(ang)
        got.append(plant.observe().rpy_W[2])
    truth, got = np.array(truth), np.array(got)
    print(f"  0 -> 1080 deg 복원 오차 최대 {np.abs(got-truth).max():.2e} rad")
    print(f"  관측된 yaw 최종값 {np.rad2deg(got[-1]):.1f} deg  <- 1080 이어야 한다")
    print(f"  max_yaw_step {np.rad2deg(plant.max_yaw_step_rad):.2f} deg "
          f"(pi=180 에 근접하면 관측 주기 부족)")
    print("\n  unwrap 이 없었다면: "
          f"{np.rad2deg(((truth+np.pi)%(2*np.pi))-np.pi)[-1]:.1f} deg 로 접혀 "
          "MPC 가 2pi 오차를 본다")

    print("\n" + "=" * 68)
    print("5. 기구학 일치 — 결함 11 이 MuJoCo 에서도 맞는가")
    print("=" * 68)
    from kinematics import leg_forward_kinematics   # noqa: PLC0415
    mj.mj_resetData(m, d)
    d.qpos[:] = HOME_QPOS
    mj.mj_forward(m, d)
    q = plant.joint_angles
    hips_W = plant.hip_positions_W()
    feet_W = plant.feet_positions_W()
    R = np.asarray(d.xmat[trunk]).reshape(3, 3)
    for i, name in enumerate(("FR", "FL", "RR", "RL")):
        fk_W = hips_W[:, i] + R @ leg_forward_kinematics(q[:, i], i)
        # FK 는 발 구의 중심을 준다. 접촉점과 비교하려면 반지름을 뺀다.
        fk_W[2] -= plant.facts.foot_sphere_radius_m
        print(f"  {name}: FK(q) vs MuJoCo 발  오차 {np.linalg.norm(fk_W-feet_W[:,i])*1000:8.3f} mm"
              f"   q={np.round(np.rad2deg(q[:,i]),2)} deg")
    print("  (0 에 가까워야 우리 기구학과 MuJoCo 모델이 같은 로봇을 말하는 것이다)")


if __name__ == "__main__":
    main()
