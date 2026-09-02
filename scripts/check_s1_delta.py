# check_s1_delta.py
import sys
from pathlib import Path

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "sim_main"))
from scenarios import SCENARIOS
from scenario_runner import run_scenario

ref = np.load(str(Path(__file__).resolve().parent.parent / "baselines") + "/S1_trot_fwd.npz")
got = run_scenario(**SCENARIOS["S1_trot_fwd"])
for k in ("com_pos", "com_vel", "rpy", "grf"):
    d = np.max(np.abs(got[k] - ref[k]))
    print(f"{k:10s} max|Δ| = {d:.3e}")
print(f"S1의 |yaw| 최댓값 = {np.abs(got['rpy'][:,2]).max():.3e} rad")