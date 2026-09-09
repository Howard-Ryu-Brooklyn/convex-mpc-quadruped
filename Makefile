.PHONY: help install check check-all types accept list run view clean

help:  ## 이 목록
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | sed 's/:.*##/ —/'

install:  ## 편집 가능 설치 (개발 의존성 포함)
	python -m pip install -e ".[mujoco,notebook,dev]"
	nbstripout --install --attributes .gitattributes

check:  ## 코드가 물리적으로 정상인가 (여기 빨간불만 진짜 문제)
	pytest -m "not golden" -q

check-all:  ## 물리 + 기준선 전체
	pytest -v

types:  ## 타입 검사 (검사기를 안 돌리는 타입은 주석이다)
	mypy

accept:  ## 변경을 의도한 것으로 받아들이고 기준선을 갱신한다
	python scripts/capture_baseline.py S0_standing S1_trot_fwd S3_yaw
	pytest -v

list:  ## 시나리오와 보행 모드 목록
	python scripts/run_sim.py --list

run:  ## 돌리고 숫자만 (S=시나리오 P=srb|mujoco|both)
	python scripts/run_sim.py $(or $(S),G_trot) --plant $(or $(P),both)

view:  ## 돌리고 뷰어로 재생 (macOS 는 mjpython 필요). S=시나리오 X=배속
	mjpython scripts/run_sim.py $(or $(S),G_trot) --view --loop --speed $(or $(X),1.0)

clean:  ## 캐시와 빌드 산출물 제거
	rm -rf .pytest_cache .mypy_cache build dist src/*.egg-info
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
