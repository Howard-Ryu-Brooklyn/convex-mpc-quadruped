.PHONY: check check-all accept help

help:
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | sed 's/:.*##/ —/'

check:  ## 코드가 물리적으로 정상인가 (여기 빨간불만 진짜 문제)
	pytest -m "not golden" -q

check-all:  ## 물리 + 기준선 전체
	pytest -v

accept:  ## 변경을 의도한 것으로 받아들이고 기준선을 갱신한다
	python sim_main/capture_baseline.py S0_standing S1_trot_fwd
	pytest -v
