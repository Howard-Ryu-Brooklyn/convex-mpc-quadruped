"""MuJoCoPlant.step() 진단 (1-7d-2b) — 부호와 추종을 '측정'으로 확인한다.

tau = J^T (-f) 의 부호는 논증으로 정하지 않는다. 논증은 틀려도 그럴듯하고,
측정은 틀리면 숫자가 다르다.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "sim_main"))

import config as cfg                       # noqa: E402
from mujoco_plant import MuJoCoPlant        # noqa: E402
from robot_types import ControlCommand      # noqa: E402
from swing import SwingTrajectoryGenerator  # noqa: E402

XML = ROOT / "mujoco_test" / "mit_cheetah3" / "scene.xml"
HOME_QPOS = np.array([0.0, 0.0, 0.65, 1.0, 0.0, 0.0, 0.0,
                      0.0, 0.9, -1.8, 0.0, 0.9, -1.8,
                      0.0, 0.9, -1.8, 0.0, 0.9, -1.8])
DT = 1.0 / 9000


def make_plant():
    return MuJoCoPlant(XML, dt=DT, home_qpos=HOME_QPOS, verbose=False)


def stance_command(plant, fz_each):
    s = plant.observe()
    r = plant.feet_positions_W() - s.p_com_W.reshape(3, 1)
    f = np.zeros((3, 4))
    f[2, :] = fz_each
    return ControlCommand(forces_W=f, r_feet_W=r,
                          is_stance=np.ones(4, dtype=bool),
                          v_feet_W=np.zeros((3, 4)), a_feet_W=np.zeros((3, 4)))


def main() -> None:
    print("=" * 70)
    print("1. 부호 확인 — 지지력을 mg/4 씩 주면 버티는가")
    print("=" * 70)
    print("   MPC 는 m=43kg 만 알지만 실제는 45.84kg 이다(의도된 불확실성).")
    print("   그래서 mg/4 만 주면 6.6% 부족해 천천히 가라앉는 것이 정답이다.")
    for label, mass in (("MPC 가 믿는 43kg", 43.0), ("실제 45.84kg", 45.84)):
        plant = make_plant()
        z0 = plant.observe().p_com_W[2]
        fz = mass * abs(cfg.gz) / 4
        for _ in range(int(0.5 / DT)):
            plant.step(stance_command(plant, fz))
        s = plant.observe()
        print(f"   {label:<18} fz={fz:6.2f} N/leg → 0.5s 후 dz {(s.p_com_W[2]-z0)*1000:+8.3f} mm"
              f"   vz {s.v_com_W[2]:+.4f} m/s   최대토크 {plant.max_abs_torque_nm:6.1f} Nm")
    print("   ⚠️ 부호가 반대면 즉시 주저앉는다 (dz 가 -100mm 수준)")

    print("\n" + "=" * 70)
    print("2. 접촉력 확인 — 지령한 힘이 실제로 지면에 실리는가")
    print("=" * 70)
    plant = make_plant()
    fz = 45.84 * abs(cfg.gz) / 4
    for _ in range(int(0.3 / DT)):
        plant.step(stance_command(plant, fz))
    mj, m, d = plant._mj, plant._model, plant._data
    total = np.zeros(6)
    for c in range(d.ncon):
        f6 = np.zeros(6)
        mj.mj_contactForce(m, d, c, f6)
        total[:3] += f6[:3]
    print(f"   지령 총 수직력 {4*fz:7.2f} N")
    print(f"   실측 접촉력 합 {np.linalg.norm(total[:3]):7.2f} N  (접촉 {d.ncon}개)")
    print(f"   실제 무게      {45.84*abs(cfg.gz):7.2f} N")
    # 설명되지 않는 숫자를 남기지 않는다: 접촉이 왜 8개인가?
    print("   접촉 내역:")
    for c in range(d.ncon):
        con = d.contact[c]
        g1 = mj.mj_id2name(m, mj.mjtObj.mjOBJ_GEOM, con.geom1) or f"geom{con.geom1}"
        g2 = mj.mj_id2name(m, mj.mjtObj.mjOBJ_GEOM, con.geom2) or f"geom{con.geom2}"
        f6 = np.zeros(6); mj.mj_contactForce(m, d, c, f6)
        print(f"     {g1:>12} ↔ {g2:<12} dist {con.dist:+.5f} m  법선력 {f6[0]:8.3f} N")

    print("\n" + "=" * 70)
    print("3. 스윙 추종 — 한 다리를 들어 베지에를 따라가게 한다")
    print("=" * 70)
    plant = make_plant()
    T_swing = 0.15
    swing = SwingTrajectoryGenerator(swing_duration_s=T_swing, clearance_height_m=0.05)
    is_stance = np.array([False, True, True, True])
    p_feet = plant.feet_positions_W()
    target = p_feet.copy()
    target[0, 0] += 0.15                      # FR 을 앞으로 15cm
    fz3 = 45.84 * abs(cfg.gz) / 3             # 나머지 세 다리가 체중을 든다

    err = []
    n = int(T_swing / DT)
    for k in range(n):
        p_feet = swing.update(is_stance=is_stance, p_feet_W=p_feet,
                              p_feet_target_W=target, dt=DT)
        s = plant.observe()
        f = np.zeros((3, 4))
        f[2, 1:] = fz3
        cmd = ControlCommand(
            forces_W=f, r_feet_W=p_feet - s.p_com_W.reshape(3, 1),
            is_stance=is_stance,
            v_feet_W=swing.velocity_W, a_feet_W=swing.acceleration_W)
        plant.step(cmd)
        err.append(np.linalg.norm(plant.feet_positions_W()[:, 0] - p_feet[:, 0]))

    err_arr = np.array(err)
    print(f"   omega_n=60 : 평균 오차 {err_arr.mean()*1000:6.2f} mm, 최대 {err_arr.max()*1000:6.2f} mm,"
          f" 최대토크 {plant.max_abs_torque_nm:5.1f} Nm")

    print("\n   대역폭 스윕 — 오차가 게인에 반비례하면 '지연'이고, 안 변하면 '구조'다")
    for wn in (60.0, 120.0, 240.0, 480.0):
        pl = MuJoCoPlant(XML, dt=DT, home_qpos=HOME_QPOS, verbose=False,
                         swing_omega_n=wn)
        sw = SwingTrajectoryGenerator(swing_duration_s=T_swing, clearance_height_m=0.05)
        pf = pl.feet_positions_W()
        tg = pf.copy(); tg[0, 0] += 0.15
        e = []
        for _ in range(n):
            pf = sw.update(is_stance=is_stance, p_feet_W=pf, p_feet_target_W=tg, dt=DT)
            st = pl.observe()
            ff = np.zeros((3, 4)); ff[2, 1:] = fz3
            pl.step(ControlCommand(forces_W=ff,
                                   r_feet_W=pf - st.p_com_W.reshape(3, 1),
                                   is_stance=is_stance,
                                   v_feet_W=sw.velocity_W, a_feet_W=sw.acceleration_W))
            e.append(np.linalg.norm(pl.feet_positions_W()[:, 0] - pf[:, 0]))
        e_arr = np.array(e)
        final = np.linalg.norm(pl.feet_positions_W()[:, 0] - tg[:, 0])
        print(f"     omega_n={wn:5.0f} rad/s : 평균 {e_arr.mean()*1000:6.2f} mm  최대 {e_arr.max()*1000:6.2f} mm"
              f"  착지 오차 {final*1000:6.2f} mm  최대토크 {pl.max_abs_torque_nm:6.1f} Nm")
    print(f"   (토크 한계 {cfg.TAU_MAX} Nm)")

    print("\n" + "=" * 70)
    print("4. 지령 검증 — v/a 없이 부르면 거부하는가")
    print("=" * 70)
    plant = make_plant()
    s = plant.observe()
    bad = ControlCommand(forces_W=np.zeros((3, 4)),
                         r_feet_W=plant.feet_positions_W() - s.p_com_W.reshape(3, 1),
                         is_stance=np.ones(4, dtype=bool))
    try:
        plant.step(bad)
        print("   ❌ 통과했다 - None 을 조용히 0 으로 대체하고 있다")
    except ValueError as e:
        print(f"   ✅ 거부: {str(e)[:60]}...")


if __name__ == "__main__":
    main()
