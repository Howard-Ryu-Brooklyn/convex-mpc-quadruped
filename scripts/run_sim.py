#!/usr/bin/env python3
"""시나리오를 SRB / MuJoCo 로 돌리고, 진단값을 찍고, 뷰어로 재생한다.

두 플랜트는 다른 질문에 답한다
    SRB     외부 변화 없이 MPC 알고리즘 자체가 맞는가
    MuJoCo  그 선형화 가정이 실제 물리에서 얼마나 강인한가
    --plant both 로 나란히 돌리면 그 차이가 곧 '선형화의 대가'다.

왜 '재생'인가 (라이브 구동이 아니라)
    물리는 9000 Hz, 화면은 60 Hz 다. 한 루프에 묶으면 뷰어가 물리 속도를
    지배하고, 그러면 눈으로 보는 시뮬레이션과 테스트가 돌리는 시뮬레이션이
    서로 다른 것이 된다. 먼저 헤드리스로 끝까지 돌리고 나중에 재생한다.

macOS 주의
    passive 뷰어는 반드시 `mjpython` 으로 띄워야 한다 (Cocoa 가 메인 스레드를
    요구한다). 감지해서 안내한다.

예시
    python   scripts/run_sim.py --list
    python   scripts/run_sim.py G_bound --plant both
    mjpython scripts/run_sim.py G_gallop --view --speed 0.3 --loop
    mjpython scripts/run_sim.py --gait bounding --vx 2.0 --duration 2 --view
"""
from __future__ import annotations

import argparse
import sys
import time
import unicodedata
from typing import Any

import numpy as np

from quadruped_mpc.control.gait_planning import GAIT_PARAMS
from quadruped_mpc.experiments.mujoco_runner import DEFAULT_XML, HOME_QPOS
from quadruped_mpc.experiments.mujoco_runner import run_scenario_mujoco
from quadruped_mpc.experiments.scenario_runner import run_scenario
from quadruped_mpc.experiments.scenarios import GALLERY, REGRESSION, SCENARIOS, ScenarioSpec


# ── CLI ───────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scenario", nargs="?", default="G_trot",
                    help="시나리오 이름 (--list 로 목록)")
    ap.add_argument("--plant", choices=["srb", "mujoco", "both"], default="mujoco")
    ap.add_argument("--list", action="store_true", help="시나리오·보행모드 목록만 찍고 끝")

    g = ap.add_argument_group("시나리오 덮어쓰기 (지정한 것만 바뀐다)")
    g.add_argument("--gait", choices=sorted(GAIT_PARAMS))
    g.add_argument("--vx", type=float, help="전진 속도 [m/s]")
    g.add_argument("--vy", type=float, help="측방 속도 [m/s]")
    g.add_argument("--wz", type=float, help="요레이트 [deg/s]")
    g.add_argument("--duration", type=float, help="지속 시간 [s]")
    g.add_argument("--clearance", type=float, help="스윙 최고 높이 [m]")

    v = ap.add_argument_group("재생")
    v.add_argument("--view", nargs="?", const="auto", default=None,
                   choices=["auto", "srb", "mujoco"],
                   help="뷰어로 재생. both 일 때 기본은 mujoco")
    v.add_argument("--speed", type=float, default=1.0, help="재생 배속")
    v.add_argument("--loop", action="store_true", help="반복 재생")
    v.add_argument("--log-hz", type=int, default=100,
                   help="로깅 주기 = 재생 프레임률 (기본 100)")
    ap.add_argument("--quiet", action="store_true")
    return ap.parse_args()

def build_spec(args: argparse.Namespace) -> ScenarioSpec:
    """이름으로 고르고, 플래그로 덮어쓴다.

    주석(annotation)이 없으면 dict 병합 리터럴이 dict[str, object] 로 추론되어
    ScenarioSpec 의 키 오타 검출이 이 자리에서만 조용히 사라진다.
    """
    if args.scenario not in SCENARIOS:
        raise SystemExit(f"알 수 없는 시나리오: {args.scenario}\n"
                         f"가능: {', '.join(SCENARIOS)}")
    spec: ScenarioSpec = {**SCENARIOS[args.scenario], "log_hz": args.log_hz}
    if args.gait is not None:
        spec["gait_name"] = args.gait
    if args.vx is not None:
        spec["v_des_x"] = args.vx
    if args.vy is not None:
        spec["v_des_y"] = args.vy
    if args.wz is not None:
        spec["omega_z_deg_s"] = args.wz
    if args.duration is not None:
        spec["duration_s"] = args.duration
    if args.clearance is not None:
        spec["clearance_height"] = args.clearance
    return spec

def print_catalog() -> None:
    print("\n  [ 보행 모드 ]  gait_planning.GAIT_PARAMS")
    print(f"    {'이름':<14s} {'duty':>5s} {'주기':>6s}  {'위상 오프셋 (FR FL RR RL)':<26s} 설명")
    for name, p in GAIT_PARAMS.items():
        off = " ".join(f"{o:.2f}" for o in p["phase_offset"])
        print(f"    {name:<14s} {p['duty_cycle']:>5.2f} {p['cycle_time']:>5.2f}s"
              f"  {off:<26s} {p['description']}")

    for title, group, note in [
        ("회귀용", REGRESSION, "골든이 물려 있다 — 바꾸면 make accept 필요"),
        ("관찰용", GALLERY,    "아무 테스트도 없다 — 마음껏 바꿔라"),
    ]:
        print(f"\n  [ 시나리오 · {title} ]  {note}")
        print(f"    {'이름':<15s} {'게이트':<13s} {'vx':>5s} {'vy':>5s} {'wz':>6s} {'시간':>6s}")
        for name, sc in group.items():
            print(f"    {name:<15s} {sc.get('gait_name', ''):<13s}"
                  f" {sc.get('v_des_x', 0.0):>5.1f} {sc.get('v_des_y', 0.0):>5.1f}"
                  f" {sc.get('omega_z_deg_s', 0.0):>5.0f}° {sc.get('duration_s', 3.0):>5.1f}s")
    print()


# ── 보고 ──────────────────────────────────────────────────────────────
def _w(text: str) -> int:
    """터미널에서 차지하는 칸 수. 한글·한자는 두 칸이다.

    str.ljust / f"{x:<12s}" 는 문자 **개수**로 채운다. 한글이 섞이면 실제
    렌더 폭과 어긋나 표가 밀린다. 라벨이 한글인 표를 만들 때마다 나오는
    문제라, 폭 계산을 한 곳에 둔다.
    """
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in text)


def _pad(text: str, width: int, right: bool = False) -> str:
    """렌더 폭 기준으로 채운다."""
    fill = " " * max(0, width - _w(text))
    return fill + text if right else text + fill

def metrics(res: dict[str, Any]) -> dict[str, str]:
    """판정하지 않는다. 숫자만 낸다."""
    pos, vel, rpy = res["com_pos"], res["com_vel"], res["rpy"]
    m = {
        "완주":            "예" if res["completed"] else f"발산 {res['diverged_at_s']:.2f}s",
        "전진 거리 [m]":    f"{pos[-1, 0] - pos[0, 0]:+.3f}",
        "측방 이탈 [m]":    f"{pos[-1, 1] - pos[0, 1]:+.3f}",
        "평균 vx [m/s]":    f"{vel[:, 0].mean():+.3f}",
        "높이 변동 [mm]":   f"{np.ptp(pos[:, 2]) * 1e3:.1f}",
        "roll/pitch RMS°":  f"{np.rad2deg(np.sqrt((rpy[:, :2] ** 2).mean())):.2f}",
        "|yaw| 최대 °":     f"{np.rad2deg(np.abs(rpy[:, 2]).max()):.2f}",
        "스윙 위상 최대":   f"{float(res['max_s']):.3f}",
        "이벤트 재해":      f"{int(res['n_event_solves'])}",
    }
    if "max_torque_nm" in res:          # MuJoCo 전용
        m["최대 토크 [N·m]"]  = f"{float(res['max_torque_nm']):.1f}"
        m["  지지"]           = f"{float(res['max_torque_stance_nm']):.1f}"
        m["  스윙"]           = f"{float(res['max_torque_swing_nm']):.1f}"
        # 정상값은 '최대 스윙 속도 x 스윙 dt' 다 — 지령은 스윙 틱(1kHz)마다만
        # 바뀌므로 물리 dt 가 아니다. trot 1 m/s 에서 3 mm 근처가 정상이고,
        # 결함 13 당시에는 17 mm 였다. 몇 배로 벗어나는지가 신호다.
        m["지령 점프 [mm]"]   = f"{float(res['max_cmd_jump_m']) * 1e3:.2f}"
        m["yaw 스텝 °"]       = f"{np.rad2deg(float(res['max_yaw_step'])):.2f}"
    return m

def report(title: str, results: dict[str, dict[str, Any]]) -> None:
    cols = list(results)
    keys: list[str] = []
    for r in results.values():
        for k in metrics(r):
            if k not in keys:
                keys.append(k)
    table = {c: metrics(results[c]) for c in cols}

    w = max(_w(k) for k in keys) + 2
    col = 16
    line = "─" * (w + col * len(cols) + 2)
    print(f"\n{line}\n  {title}\n{line}")
    print("  " + _pad("", w) + "".join(_pad(c, col, right=True) for c in cols))
    for k in keys:
        print("  " + _pad(k, w)
              + "".join(_pad(table[c].get(k, "—"), col, right=True) for c in cols))
    print()
    if len(cols) == 2:
        print("  두 열의 차이가 곧 '선형화와 이상화의 대가'다. "
              "SRB 에서만 되는 것은 아직 되는 것이 아니다.\n")


# ── 재생 ──────────────────────────────────────────────────────────────
def srb_qpos(res: dict[str, Any]) -> np.ndarray:
    """SRB 결과를 MuJoCo 뷰어가 먹을 수 있는 qpos 열로 되돌린다.

    **시각화 전용 근사다.** SRB 의 com_pos 는 질량중심이고, MuJoCo 자유관절의
    qpos[0:3] 은 trunk **바디 원점**이다. 둘의 차이(수 cm)는 여기서 무시한다 —
    그림을 보려는 것이지 물리를 재현하려는 것이 아니기 때문이다.
    이 구분을 흐린 것이 model_audit 이 감시하는 규약 불일치의 전형이다.
    """
    import mujoco

    com, R, q = res["com_pos"], res["R_W_B"], res["q"]
    n = len(com)
    qpos = np.tile(HOME_QPOS.astype(float), (n, 1))
    qpos[:, 0:3] = com
    quat = np.empty(4)
    for k in range(n):
        mujoco.mju_mat2Quat(quat, np.ascontiguousarray(R[k]).reshape(9))
        qpos[k, 3:7] = quat
        qpos[k, 7:19] = q[k].T.reshape(12)     # (3,4) -> [leg][joint]
    return qpos

def replay(qpos: np.ndarray, frame_dt: float, speed: float, loop: bool,
           label: str) -> None:
    import mujoco
    import mujoco.viewer

    model = mujoco.MjModel.from_xml_path(str(DEFAULT_XML))
    data = mujoco.MjData(model)
    if qpos.shape[1] != model.nq:
        raise SystemExit(f"qpos 폭 불일치: 기록 {qpos.shape[1]} vs 모델 nq {model.nq}")

    try:
        handle = mujoco.viewer.launch_passive(model, data)
    except RuntimeError as e:
        if sys.platform == "darwin":
            raise SystemExit("\n  macOS 에서는 뷰어를 mjpython 으로 띄워야 한다:\n"
                             f"    mjpython {' '.join(sys.argv)}\n") from e
        raise

    print(f"  재생 [{label}]  {len(qpos)} 프레임 @ {1 / frame_dt:.0f} Hz"
          f" × {speed:g} 배속{' (반복)' if loop else ''}.  창을 닫으면 끝난다.")
    with handle as viewer:
        while viewer.is_running():
            t0 = time.time()
            for k in range(len(qpos)):
                if not viewer.is_running():
                    break
                data.qpos[:] = qpos[k]
                mujoco.mj_forward(model, data)   # 파생량만 갱신, 적분 없음
                viewer.sync()
                lag = t0 + (k + 1) * frame_dt / speed - time.time()
                if lag > 0:
                    time.sleep(lag)
            if not loop:
                break


# ── main ──────────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()
    if args.list:
        print_catalog()
        return

    spec = build_spec(args)
    tag = (f"{spec.get('gait_name')}  vx={spec.get('v_des_x', 0.0):g}"
           f"  vy={spec.get('v_des_y', 0.0):g}"
           f"  wz={spec.get('omega_z_deg_s', 0.0):g}deg/s"
           f"  {spec.get('duration_s', 3.0):g}s")

    results: dict[str, dict[str, Any]] = {}
    if args.plant in ("srb", "both"):
        print(f"\n  SRB    실행 중 … {tag}")
        results["SRB"] = run_scenario(**spec)
    if args.plant in ("mujoco", "both"):
        print(f"  MuJoCo 실행 중 … {tag}")
        results["MuJoCo"] = run_scenario_mujoco(**spec, verbose=not args.quiet)

    report(f"{args.scenario}   {tag}", results)

    if args.view is None:
        return
    which = args.view
    if which == "auto":
        which = "mujoco" if "MuJoCo" in results else "srb"
    if which == "mujoco" and "MuJoCo" not in results:
        raise SystemExit("--view mujoco 인데 MuJoCo 를 돌리지 않았다 (--plant 확인)")
    if which == "srb" and "SRB" not in results:
        raise SystemExit("--view srb 인데 SRB 를 돌리지 않았다 (--plant 확인)")

    qpos = (results["MuJoCo"]["qpos"] if which == "mujoco"
            else srb_qpos(results["SRB"]))
    replay(qpos, 1.0 / args.log_hz, args.speed, args.loop,
           which + ("  (근사 재구성)" if which == "srb" else ""))

if __name__ == "__main__":
    main()
