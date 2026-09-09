"""저장소 안의 데이터 경로.

**코드가 자기 위치를 아는 지점은 이 파일 하나뿐이다.** 예전에는 스크립트마다
`ROOT = Path(__file__).resolve().parent.parent` 를 각자 계산했고, 그래서 파일을
한 칸 옮기는 순간 조용히 다른 곳을 가리켰다. 같은 사실을 두 곳에서 계산하면
반드시 갈라진다 — 이 저장소가 결함 5 로 배운 것이다.

설치해서 쓰는 경우(pip install 로 site-packages 에 들어간 경우) 저장소의
models/ 와 baselines/ 는 따라오지 않는다. 그때는 환경변수로 지정한다.

    export QUADRUPED_MPC_MODELS=/path/to/models
    export QUADRUPED_MPC_BASELINES=/path/to/baselines
"""
from __future__ import annotations

import os
from pathlib import Path

#: src/quadruped_mpc/paths.py -> src/quadruped_mpc -> src -> 저장소 루트
REPO_ROOT = Path(__file__).resolve().parents[2]

MODELS_DIR = Path(os.environ.get("QUADRUPED_MPC_MODELS", REPO_ROOT / "models"))
BASELINE_DIR = Path(os.environ.get("QUADRUPED_MPC_BASELINES", REPO_ROOT / "baselines"))

#: 기본 로봇 모델. MIT Cheetah 3 (mujoco_menagerie 유래, Apache-2.0).
DEFAULT_MODEL_XML = MODELS_DIR / "mit_cheetah3" / "scene.xml"
