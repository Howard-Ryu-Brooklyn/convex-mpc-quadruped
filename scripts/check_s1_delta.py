# check_s1_delta.py

import numpy as np
from quadruped_mpc.experiments.scenarios import SCENARIOS
from quadruped_mpc.paths import BASELINE_DIR
from quadruped_mpc.experiments.scenario_runner import run_scenario

ref = np.load(BASELINE_DIR / "S1_trot_fwd.npz")
got = run_scenario(**SCENARIOS["S1_trot_fwd"])
for k in ("com_pos", "com_vel", "rpy", "grf"):
    d = np.max(np.abs(got[k] - ref[k]))
    print(f"{k:10s} max|Δ| = {d:.3e}")
print(f"S1의 |yaw| 최댓값 = {np.abs(got['rpy'][:,2]).max():.3e} rad")