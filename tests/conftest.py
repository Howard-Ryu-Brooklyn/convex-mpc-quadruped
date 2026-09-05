"""pytest 공통 설정. pytest 가 테스트 모듈보다 먼저 자동으로 읽는다.

Step 1-6 에서 정식 패키지(src/quadruped_mpc/)로 옮기면 sys.path 조작은 사라진다.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "sim_main"))

from scenario_runner import run_scenario  # noqa: E402
from scenarios import SCENARIOS  # noqa: E402


@pytest.fixture(scope="session")
def scenario_results():
    """시나리오 실행 결과를 세션당 한 번만 계산해 모든 테스트가 공유한다.

    이 fixture 가 없으면 같은 시뮬레이션이 반복 실행된다. 실제로 S1 은 5 번,
    S3 는 3 번 돌고 있었고 (golden, 물리 불변량, 높이, 접촉 마스크, 지면 관통)
    불변량을 추가할수록 선형으로 늘어난다.

    ⚠️ 반환된 dict 는 모든 테스트가 공유하므로 **절대 수정하지 말 것**.
       한 테스트가 배열을 건드리면 다음 테스트가 오염된 데이터를 본다.
       세션 스코프 fixture 를 쓸 때 가장 흔한 사고다.
    """
    cache: dict[str, dict] = {}

    def get(name: str) -> dict:
        if name not in cache:
            cache[name] = run_scenario(**SCENARIOS[name])
        return cache[name]

    return get
