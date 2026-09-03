"""pytest 공통 설정. pytest 가 테스트 모듈보다 먼저 자동으로 읽는다.

sim_main 을 import 경로에 넣는 작업을 여기 한 곳에 모은다.
Step 1-6 에서 정식 패키지(src/quadruped_mpc/)로 옮기면 이 파일은 사라진다.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "sim_main"))
