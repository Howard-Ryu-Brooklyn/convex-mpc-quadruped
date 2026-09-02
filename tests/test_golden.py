import sys
from pathlib import Path

import numpy as np
import numpy.testing as npt
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "sim_main"))

from scenarios import SCENARIOS          # noqa: E402
from scenario_runner import run_scenario # noqa: E402

BASELINE_DIR = Path(__file__).resolve().parent.parent / "baselines"
# 신호별 허용오차. 물리량마다 스케일과 수치 노이즈 바닥이 다르다.
#   판정식: |a-b| <= atol + rtol*|b|
TOLERANCES = {
    #             rtol,   atol
    "com_pos":  (1e-9,  1e-12),   # [m]     피코미터
    "com_vel":  (1e-9,  1e-12),   # [m/s]
    "rpy":      (1e-9,  1e-12),   # [rad]
    "omega":    (1e-9,  1e-12),   # [rad/s]
    "feet_W":   (1e-9,  1e-12),   # [m]
    "grf":      (1e-9,  1e-6),    # [N]  ← 스윙/정지 다리에는 솔버 수렴 오차로
                                  #   ~1e-4 N의 잔류력이 남는다. 이 값에 rtol=1e-9를
                                  #   걸면 허용치가 1e-13이 되는데, 400N급 연산의
                                  #   배정밀도 오차 바닥(~1e-13)과 같아 물리적으로
                                  #   만족 불가능하다. 400N에서 1e-6 N 미만의 변화는
                                  #   어차피 의미가 없다.
    "contact":  (0.0,   0.0),     # 0/1 정수 — 정확히 일치해야 한다
}


# ── Tier A: 궤적 회귀 (엄격, 리팩토링 검증용) ──
# golden 마커: `make check`에서 제외된다. 기준선 비교는 "코드가 정상인가"가
# 아니라 "이 변경을 받아들일 것인가"를 묻는 별도의 질문이기 때문이다.
@pytest.mark.golden
@pytest.mark.parametrize("name", ["S0_standing", "S1_trot_fwd"])
def test_matches_baseline(name):
    path = BASELINE_DIR / f"{name}.npz"
    if not path.exists():
        pytest.skip(f"기준선 없음: {path} — capture_baseline.py를 먼저 실행")

    ref = np.load(path)
    got = run_scenario(**SCENARIOS[name])

    assert got["completed"], f"{name}: {got['diverged_at_s']}s에서 발산"

    for key, (rtol, atol) in TOLERANCES.items():
        assert got[key].shape == ref[key].shape, (
            f"[{name}] {key} shape 불일치: {got[key].shape} != {ref[key].shape}"
        )
        npt.assert_allclose(got[key], ref[key], rtol=rtol, atol=atol,
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
    """높이를 제외한 물리 불변량.

    높이는 test_s1_height_regulation으로 분리되어 있다. 단언이 순차적이면
    앞의 것이 뒤의 것을 가리므로(높이 실패가 자세 실패를 은폐했던 전례),
    독립적으로 판정되어야 하는 성질은 테스트를 나눈다.
    테스트의 입도가 곧 진단의 해상도다.
    """
    import config as cfg
    h = run_scenario(**SCENARIOS["S1_trot_fwd"])

    assert h["completed"], f"[S1] 수치 발산 @ {h['diverged_at_s']}s"
    assert np.all(np.isfinite(h["com_pos"])), "[S1] CoM에 NaN/Inf"

    rp = np.abs(h["rpy"][:, :2])
    assert np.all(rp < np.deg2rad(15)), f"[S1] 자세 이탈: {np.rad2deg(rp.max()):.1f}deg"

    assert h["com_pos"][-1, 0] > 0.8 * 3.0, f"[S1] 전진 부족: {h['com_pos'][-1, 0]:.2f}m"

    fx, fy, fz = h["grf"][:, 0, :], h["grf"][:, 1, :], h["grf"][:, 2, :]
    assert np.all(np.abs(fx) <= cfg.MU_FRICTION * fz + TOL_N), "[S1] 마찰 피라미드 위반 (x)"
    assert np.all(np.abs(fy) <= cfg.MU_FRICTION * fz + TOL_N), "[S1] 마찰 피라미드 위반 (y)"
    assert np.all(fz >= -TOL_N), "[S1] 수직항력 음수"

    mg = abs(cfg.m * cfg.gz)
    assert abs(fz.sum(axis=1).mean() - mg) < 0.1 * mg, "[S1] 평균 수직항력이 중력과 불일치"


def test_s1_height_regulation():
    """높이 유지. 결함 8(MPC 토크암) 회귀 감시."""
    h = run_scenario(**SCENARIOS["S1_trot_fwd"])
    z = h["com_pos"][:, 2]
    assert np.all(np.abs(z - 0.34) < 0.05), f"[S1] 높이 이탈: [{z.min():.3f}, {z.max():.3f}]"


def test_s3_yaw_stays_bounded():
    """선회 시 자세 안정성. 결함 1(Ac 전치) + 결함 2(r_feet_traj)의 회귀 감시.

    두 결함 모두 ψ=0에서 오차가 0이라 S1(직진)으로는 절대 잡히지 않는다.
    이 테스트가 이 두 버그를 지키는 유일한 파수꾼이다.
    """
    h = run_scenario(**SCENARIOS["S3_yaw"])
    _assert_healthy(h, "S3", z_tol=0.05, rp_tol_deg=8.0)