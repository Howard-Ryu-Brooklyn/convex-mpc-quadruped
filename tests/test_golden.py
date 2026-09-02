import sys
from pathlib import Path

import numpy as np
import numpy.testing as npt
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "sim_main"))

from scenarios import SCENARIOS          # noqa: E402
from scenario_runner import run_scenario # noqa: E402

BASELINE_DIR = Path(__file__).resolve().parent.parent / "baselines"
RTOL, ATOL = 1e-9, 1e-12
SIGNALS = ("com_pos", "com_vel", "rpy", "omega", "grf", "feet_W", "contact")


# ── Tier A: 궤적 회귀 (엄격, 리팩토링 검증용) ──
@pytest.mark.parametrize("name", ["S0_standing", "S1_trot_fwd"])
def test_matches_baseline(name):
    path = BASELINE_DIR / f"{name}.npz"
    if not path.exists():
        pytest.skip(f"기준선 없음: {path} — capture_baseline.py를 먼저 실행")

    ref = np.load(path)
    got = run_scenario(**SCENARIOS[name])

    assert got["completed"], f"{name}: {got['diverged_at_s']}s에서 발산"

    for key in SIGNALS:
        assert got[key].shape == ref[key].shape, (
            f"[{name}] {key} shape 불일치: {got[key].shape} != {ref[key].shape}"
        )
        npt.assert_allclose(got[key], ref[key], rtol=RTOL, atol=ATOL,
                            err_msg=f"[{name}] {key} 회귀 발생")

# ── 물리 상수 ──
TOL_N = 1.0   # OSQP 수렴 허용오차 흡수 [N]. 총 지지력 ~421N 대비 0.24%.


def _assert_healthy(h, name, z_tol=0.05, rp_tol_deg=15.0):
    """물리적으로 살아있는 궤적인지 검사. 여러 시나리오에서 재사용."""
    assert h["completed"], f"[{name}] 수치 발산 @ {h['diverged_at_s']}s"
    assert np.all(np.isfinite(h["com_pos"])), f"[{name}] CoM에 NaN/Inf"

    z = h["com_pos"][:, 2]
    assert np.all(np.abs(z - 0.34) < z_tol), \
        f"[{name}] 높이 이탈: [{z.min():.3f}, {z.max():.3f}]"

    rp = np.abs(h["rpy"][:, :2])
    assert np.all(rp < np.deg2rad(rp_tol_deg)), \
        f"[{name}] 자세 이탈: {np.rad2deg(rp.max()):.1f}deg"


def test_s1_physical_invariants():
    import config as cfg
    h = run_scenario(**SCENARIOS["S1_trot_fwd"])
    _assert_healthy(h, "S1")

    # 3초에 1 m/s면 3m. 정착 시간을 감안해 80%를 하한으로.
    assert h["com_pos"][-1, 0] > 0.8 * 3.0, f"전진 부족: {h['com_pos'][-1, 0]:.2f}m"

    # 코드가 구현한 것은 원뿔이 아니라 사각뿔(피라미드)이다.
    #   |fx| <= μ·fz  AND  |fy| <= μ·fz   (독립 제약)
    # 따라서 ||f_xy|| 는 최대 √2·μ·fz 까지 허용된다. 이는 진짜 마찰원뿔의
    # 외부 근사이며, 실기 미끄러짐의 알려진 원인이다 (Step 11에서 검증).
    fx, fy, fz = h["grf"][:, 0, :], h["grf"][:, 1, :], h["grf"][:, 2, :]
    assert np.all(np.abs(fx) <= cfg.MU_FRICTION * fz + TOL_N), "마찰 피라미드 위반 (x축)"
    assert np.all(np.abs(fy) <= cfg.MU_FRICTION * fz + TOL_N), "마찰 피라미드 위반 (y축)"
    assert np.all(fz >= -TOL_N), "수직항력 음수 — 지면이 발을 당기고 있음"

    # 정상 보행이면 평균 수직항력은 중력과 균형을 이뤄야 한다.
    mg = abs(cfg.m * cfg.gz)
    assert abs(fz.sum(axis=1).mean() - mg) < 0.1 * mg, "평균 수직항력이 중력과 불일치"


@pytest.mark.xfail(reason="결함 1(Ac 전치) — ψ≈90°에서 roll/pitch 피드백 부호 반전", strict=True)
def test_s3_yaw_stays_bounded():
    h = run_scenario(**SCENARIOS["S3_yaw"])
    _assert_healthy(h, "S3", z_tol=0.05, rp_tol_deg=8.0)