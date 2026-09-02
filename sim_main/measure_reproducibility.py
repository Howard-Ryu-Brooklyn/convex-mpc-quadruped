# scripts/measure_reproducibility.py
"""허용오차는 감으로 정하지 않는다. 측정해서 정한다."""
import numpy as np
from scenario_runner import run_scenario

S1 = dict(gait_name="trotting", v_des_x=1.0, omega_z_deg_s=0.0, duration_s=3.0)

runs = [run_scenario(**S1) for _ in range(5)]
ref = runs[0]

print(f"{'signal':10s} {'max|Δ|':>12s}  {'max|Δ|/scale':>14s}")
for key in ("com_pos", "com_vel", "rpy", "omega", "grf"):
    scale = max(np.max(np.abs(ref[key])), 1e-12)
    dev = max(np.max(np.abs(r[key] - ref[key])) for r in runs[1:])
    print(f"{key:10s} {dev:12.3e}  {dev / scale:14.3e}")
    max_s = ref["max_s"]
    print(f"max_s: {max_s}")

# ── 음성 대조군: 이 도구가 '차이'를 감지할 수 있는가? ──
alt = run_scenario(**{**S1, "clearance_height": 0.07})
dev = np.max(np.abs(alt["com_pos"] - ref["com_pos"]))
assert dev > 0, "❌ 측정이 고장났거나 인자가 무시되고 있다"
print(f"✅ 음성 대조군 통과: clearance 0.05→0.07 시 max|Δ| = {dev:.3e}")