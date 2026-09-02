"""기준선 캡처.

⚠️ 기준선을 갱신할 때는 '왜 바뀌어야 하는지'를 커밋 메시지에 반드시 남긴다.
   이유 없는 기준선 갱신 = 테스트를 끈 것과 같다.
"""
import sys
import numpy as np
from pathlib import Path

from scenarios import SCENARIOS
from scenario_runner import run_scenario

BASELINE_DIR = Path(__file__).resolve().parent.parent / "baselines"


def capture(name: str) -> Path:
    if name not in SCENARIOS:
        raise KeyError(f"알 수 없는 시나리오: {name}. 가능: {list(SCENARIOS)}")

    result = run_scenario(**SCENARIOS[name])

    if not result["completed"]:
        raise RuntimeError(
            f"{name}: {result['diverged_at_s']:.3f}s에서 발산. 발산하는 시나리오는 기준선이 될 수 없다."
        )

    BASELINE_DIR.mkdir(exist_ok=True)
    out = BASELINE_DIR / f"{name}.npz"
    arrays = {k: v for k, v in result.items() if isinstance(v, np.ndarray)}
    np.savez_compressed(out, **arrays)
    print(f"✅ {name}: {out}  ({out.stat().st_size / 1024:.1f} KB, {len(result['t'])} 샘플)")
    return out


if __name__ == "__main__":
    names = sys.argv[1:] or ["S0_standing", "S1_trot_fwd"]
    for n in names:
        capture(n)