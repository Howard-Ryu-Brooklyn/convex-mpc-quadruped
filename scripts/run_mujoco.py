#!/usr/bin/env python3
"""MuJoCo 로 시나리오를 돌리고, 진단값을 찍고, 뷰어로 재생한다.

왜 '재생'인가 (라이브 구동이 아니라)
    물리는 9000 Hz 로 돌고 화면은 60 Hz 면 충분하다. 두 주기를 한 루프에
    묶으면 뷰어가 물리 속도를 지배하게 되고, 그러면 여기서 보는 것과
    테스트가 돌리는 것이 **다른 시뮬레이션**이 된다.
    그래서 먼저 헤드리스로 끝까지 돌려 qpos 를 로깅하고, 그 다음 재생한다.
    덕분에 배속·반복·되감기가 공짜로 되고, mujoco_runner 에는 뷰어 코드가
    한 줄도 들어가지 않는다.

macOS 주의
    MuJoCo 의 passive 뷰어는 macOS 에서 반드시 `mjpython` 으로 실행해야 한다
    (Cocoa 가 메인 스레드를 요구한다). 이 스크립트는 그 상황을 감지해서
    알려준다.

    python  scripts/run_mujoco.py               # 돌리고 숫자만
    mjpython scripts/run_mujoco.py --view       # 돌리고 재생까지
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "sim_main"))

from mujoco_runner import DEFAULT_XML, run_scenario_mujoco   # noqa: E402
from scenarios import SCENARIOS, ScenarioSpec                # noqa: E402


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scenario", nargs="?", default="S1_trot_fwd",
                    choices=sorted(SCENARIOS), help="시나리오 이름")
    ap.add_argument("--duration", type=float, default=None,
                    help="시나리오 기본값을 덮어쓴다 [s]")
    ap.add_argument("--view", action="store_true", help="끝난 뒤 뷰어로 재생")
    ap.add_argument("--speed", type=float, default=1.0, help="재생 배속")
    ap.add_argument("--loop", action="store_true", help="반복 재생")
    ap.add_argument("--log-hz", type=int, default=100,
                    help="로깅 주기. 재생 프레임률이 된다 (기본 100)")
    ap.add_argument("--quiet", action="store_true")
    return ap.parse_args()


def report(name: str, res: dict) -> None:
    """숫자를 한 화면에 모아 놓는다. 판정은 사람이 한다."""
    ok = bool(res["completed"])
    t = res["t"]
    print()
    print("=" * 62)
    print(f"  {name}   {'완주' if ok else '발산 @ %.3f s' % res['diverged_at_s']}"
          f"   ({t[-1]:.2f} s, {len(t)} 프레임)")
    print("=" * 62)

    rpy, pos, vel = res["rpy"], res["com_pos"], res["com_vel"]
    rows = [
        ("전진 거리",        f"{pos[-1, 0] - pos[0, 0]:+.3f} m"),
        ("측방 이탈",        f"{pos[-1, 1] - pos[0, 1]:+.3f} m"),
        ("높이 변동 (p-p)",  f"{np.ptp(pos[:, 2]) * 1e3:.1f} mm"),
        ("평균 전진 속도",   f"{vel[:, 0].mean():+.3f} m/s"),
        ("roll/pitch RMS",   f"{np.rad2deg(np.sqrt((rpy[:, :2] ** 2).mean())):.2f} deg"),
        ("|yaw| 최대",       f"{np.rad2deg(np.abs(rpy[:, 2]).max()):.2f} deg"),
    ]
    print("\n  [ 추종 ]")
    for k, v in rows:
        print(f"    {k:<18s} {v:>14s}")

    lim = 250.0   # config.TAU_MAX 와 같은 값. 넘으면 엔진이 잘라낸다.
    tq = [
        ("최대 토크 (전체)", float(res["max_torque_nm"]),        lim),
        ("  지지 다리",      float(res["max_torque_stance_nm"]), lim),
        ("  스윙 다리",      float(res["max_torque_swing_nm"]),  lim),
    ]
    print("\n  [ 토크 ]  한계 250 N·m")
    for k, v, l in tq:
        flag = "  ⚠ 초과" if v > l else ""
        print(f"    {k:<18s} {v:>10.1f} N·m{flag}")
    print(f"    {'발생 시각':<18s} {float(res['t_at_max_torque_s']):>10.3f} s")

    print("\n  [ 가정 감시 ]")
    diag = [
        ("스윙 위상 최대",   f"{float(res['max_s']):.3f}",              "1.0 을 크게 넘으면 베지에 외삽"),
        ("이벤트 재해 횟수", f"{int(res['n_event_solves'])}",           "접촉 전이마다 1 회가 정상"),
        ("yaw 스텝 최대",    f"{np.rad2deg(float(res['max_yaw_step'])):.2f} deg", "180 에 근접하면 unwrap 위험"),
        ("지령 점프 최대",   f"{float(res['max_cmd_jump_m']) * 1e3:.2f} mm",      "결함 13 회귀 감시 (정상 ~0.2)"),
        ("목표 점프 최대",   f"{float(res['max_target_jump_m']) * 1e3:.2f} mm",   "Raibert 갱신 계단"),
    ]
    for k, v, why in diag:
        print(f"    {k:<18s} {v:>12s}   {why}")
    print()


def replay(qpos: np.ndarray, frame_dt: float, speed: float, loop: bool) -> None:
    import mujoco
    import mujoco.viewer

    model = mujoco.MjModel.from_xml_path(str(DEFAULT_XML))
    data = mujoco.MjData(model)

    if qpos.shape[1] != model.nq:
        raise SystemExit(f"qpos 폭이 안 맞는다: 기록 {qpos.shape[1]} vs 모델 nq {model.nq}")

    try:
        handle = mujoco.viewer.launch_passive(model, data)
    except RuntimeError as e:
        if "mjpython" in str(e).lower() or sys.platform == "darwin":
            raise SystemExit(
                "\nmacOS 에서는 뷰어를 mjpython 으로 띄워야 한다:\n"
                f"    mjpython {' '.join(sys.argv)}\n") from e
        raise

    print(f"  재생: {len(qpos)} 프레임 @ {1/frame_dt:.0f} Hz × {speed:.2f} 배속"
          f"{' (반복)' if loop else ''}.  창을 닫으면 끝난다.")
    with handle as viewer:
        while viewer.is_running():
            t0 = time.time()
            for k in range(len(qpos)):
                if not viewer.is_running():
                    break
                data.qpos[:] = qpos[k]
                mujoco.mj_forward(model, data)     # 파생량(xpos 등)만 갱신, 적분 없음
                viewer.sync()
                lag = t0 + (k + 1) * frame_dt / speed - time.time()
                if lag > 0:
                    time.sleep(lag)
            if not loop:
                break


def main() -> None:
    args = parse_args()
    # 주석(annotation)이 없으면 dict 병합 리터럴이 dict[str, object] 로
    # 추론되어 ScenarioSpec 의 키 오타 검출이 이 한 줄에서만 사라진다.
    spec: ScenarioSpec = {**SCENARIOS[args.scenario], "log_hz": args.log_hz}
    if args.duration is not None:
        spec["duration_s"] = args.duration

    print(f"\n  {args.scenario} 실행 중 … ({spec})")
    res = run_scenario_mujoco(**spec, verbose=not args.quiet)
    report(args.scenario, res)

    if args.view:
        replay(res["qpos"], 1.0 / args.log_hz, args.speed, args.loop)


if __name__ == "__main__":
    main()
