"""기준선과 현재 코드의 **품질**을 나란히 비교한다.

check_s1_delta.py 는 '얼마나 바뀌었나'를 답한다. 이 스크립트는
'좋아졌나 나빠졌나'를 답한다. 동작을 바꾸는 수정 뒤에는 둘 다 필요하다.

변화량이 크다는 것은 그 자체로 좋지도 나쁘지도 않다. 기준선을 받아들일지는
품질 지표를 보고 정한다.

실행: python scripts/compare_to_baseline.py [시나리오키 ...]
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "sim_main"))

import config as cfg  # noqa: E402
from scenario_runner import run_scenario  # noqa: E402
from scenarios import SCENARIOS  # noqa: E402


def metrics(h, nominal_height, target_distance_m=0.0):
    z = h["com_pos"][:, 2]
    rp = np.abs(h["rpy"][:, :2])
    fx, fy, fz = h["grf"][:, 0, :], h["grf"][:, 1, :], h["grf"][:, 2, :]
    mg = abs(cfg.m * cfg.gz)
    cone = np.maximum(np.abs(fx), np.abs(fy)) - cfg.MU_FRICTION * fz
    return {
        "높이 편차 최대 [m]": float(np.abs(z - nominal_height).max()),
        "높이 RMS [m]": float(np.sqrt(np.mean((z - nominal_height) ** 2))),
        "roll/pitch 최대 [deg]": float(np.rad2deg(rp.max())),
        "roll/pitch RMS [deg]": float(np.rad2deg(np.sqrt(np.mean(rp**2)))),
        "|yaw| 최대 [deg]": float(np.rad2deg(np.abs(h["rpy"][:, 2]).max())),
        "|v| 최대 [m/s]": float(np.linalg.norm(h["com_vel"], axis=1).max()),
        "전진 거리 [m]": float(h["com_pos"][-1, 0] - h["com_pos"][0, 0]),
        "전진 거리 오차 최대 [m]": float(abs(
            (h["com_pos"][-1, 0] - h["com_pos"][0, 0]) - target_distance_m)),
        "측면 이탈 최대 [m]": float(np.abs(h["com_pos"][:, 1]).max()),
        # ⚠️ 이 값은 30Hz 로그 표본의 평균이지 시간 평균이 아니다. 힘이 접촉
        #    전이에서 계단처럼 바뀌므로 표본 평균은 편향된다. 실제로 이벤트
        #    구동 도입 후 0.9991 -> 0.9939 로 움직였지만 높이 편차는 4.35mm 로
        #    변함이 없다 - 0.61% 지지력 부족이 실재했다면 3초에 27cm 가라앉는다.
        #    즉 이 변화는 표본화 아티팩트다. TODO: runner 에서 전 스텝 시간
        #    평균을 계산해 넘긴다.
        "평균 수직력 / mg": float(fz.sum(axis=1).mean() / mg),
        # 양수 = 위반. OSQP 수렴 오차(~1e-1 N) 수준이면 무해하다.
        "마찰 피라미드 위반 최대 [N]": float(max(cone.max(), 0.0)),
    }


# 잡음 바닥(noise floor) — 이 아래의 변화는 물리적으로 의미가 없다.
# 판단 기준은 '유효숫자가 몇 자리냐'가 아니라 '이 차이가 실기에서 구별
# 가능한 양이냐'다. 상대 비교(ratio)만 쓰면 0 근처에서 반드시 거짓 경보가
# 난다 — golden 테스트에서 rtol 만으로 near-zero 신호를 다룰 수 없어
# 신호별 atol 을 넣었던 것과 정확히 같은 문제다.
#   1e-4 m   = 0.1 mm   (엔코더/IMU 분해능 아래)
#   1e-2 deg             (자세 추정 노이즈 아래)
#   1e-3 m/s             (속도 추정 노이즈 아래)
#   1e-1 N               (OSQP 수렴 오차 수준)
NOISE_FLOOR = {
    "[m/s]": 1e-3,   # "[m]" 보다 먼저 검사되어야 한다 (부분 문자열 포함 관계)
    "[m]":   1e-4,
    "[deg]": 1e-2,
    "[N]":   1e-1,
    "/ mg":  1e-4,
}


#: 이상값이 '작을수록 좋다'가 아니라 '특정 값이어야 하는' 지표.
#: 평균 수직력/mg 는 1.0 에서 멀어지는 것이 나쁜 것이지 작아지는 것이 나쁜
#: 게 아니다. 아래 판정은 |x - target| 로 본다.
#:
#: 이 항목이 없던 동안 '평균 수직력 / mg' 는 이름에 편차/RMS/최대 가 없어서
#: lower_is_better 가 False 였고, 따라서 **어떤 값이 나와도 항상 "=" 로
#: 찍혔다.** 0.9991 -> 0.9939 (지지력 0.61% 부족) 도 조용히 통과했다.
#: 비교 도구가 무엇을 평가하지 않고 있는지 아는 것도 도구의 일부다.
TARGET_VALUE = {"평균 수직력 / mg": 1.0}


def _noise_floor(metric_name: str) -> float:
    """지표 이름의 단위 표기로 잡음 바닥을 고른다. 모르면 0 (항상 비교)."""
    for unit, floor in NOISE_FLOOR.items():
        if unit in metric_name:
            return floor
    return 0.0


def compare(name):
    path = ROOT / "baselines" / f"{name}.npz"
    if not path.exists():
        print(f"[{name}] 기준선 없음: {path}")
        return

    ref = dict(np.load(path))
    got = run_scenario(**SCENARIOS[name])
    nominal = cfg.leg_length_straight / 2

    # 시나리오의 목표 전진 거리 = v_des_x * duration
    sc = SCENARIOS[name]
    target = sc.get("v_des_x", 0.0) * sc["duration_s"]
    m_ref, m_got = metrics(ref, nominal, target), metrics(got, nominal, target)

    print(f"\n═══ {name} ═══")
    print(f"{'지표':<28}{'기준선':>14}{'현재':>14}{'변화':>12}")
    print("─" * 68)
    for k in m_ref:
        a, b = m_ref[k], m_got[k]
        if max(abs(a), abs(b)) < _noise_floor(k):
            # 두 값 다 잡음 바닥 아래 → 비율이 몇이든 비교 자체가 무의미하다.
            mark = "  —"
        elif k in TARGET_VALUE:
            # 목표값에서 얼마나 멀어졌는가로 판정한다.
            t = TARGET_VALUE[k]
            da, db = abs(a - t), abs(b - t)
            if max(da, db) < _noise_floor(k):
                mark = "  —"
            elif da == 0 or abs(db / da - 1.0) < 0.02:
                mark = "  ="
            else:
                mark = "  ✅ 개선" if db < da else "  ⚠️ 악화"
        elif abs(a) < 1e-12:
            mark = "  —"
        else:
            ratio = b / a
            # 작을수록 좋은 지표만 화살표를 붙인다
            lower_is_better = any(w in k for w in ("편차", "RMS", "최대", "이탈", "위반"))
            if not lower_is_better or abs(ratio - 1.0) < 0.02:
                mark = "  ="
            else:
                mark = "  ✅ 개선" if ratio < 1 else "  ⚠️ 악화"
        print(f"{k:<28}{a:>14.6f}{b:>14.6f}{mark:>12}")


if __name__ == "__main__":
    for name in (sys.argv[1:] or ["S0_standing", "S1_trot_fwd", "S3_yaw"]):
        compare(name)
