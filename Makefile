.PHONY: help install check check-all types accept list run view gif clean

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

gif:  ## 녹화 -> README GIF. IN= [SS=시작s] [T=길이s] [CROP=w:h:x:y] [W=720] [FPS=20] [OUT=]
	@test -n "$(IN)" || (echo "사용법: make gif IN=~/Desktop/화면\\ 기록.mov"; exit 1)
	@mkdir -p $(dir $(GIF_OUT))
	@# 2-패스 팔레트 방식. 한 번에 뽑으면 256색을 프레임마다 다시 고르느라
	@# 잔디와 그림자에 밴딩이 생긴다.
	ffmpeg -loglevel error $(GIF_TRIM) -i "$(IN)" \
	  -vf "$(GIF_CROP)fps=$(GIF_FPS),scale=$(GIF_W):-1:flags=lanczos,palettegen=stats_mode=diff" \
	  -y /tmp/quadruped-palette.png
	ffmpeg -loglevel error $(GIF_TRIM) -i "$(IN)" -i /tmp/quadruped-palette.png \
	  -lavfi "$(GIF_CROP)fps=$(GIF_FPS),scale=$(GIF_W):-1:flags=lanczos[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=3" \
	  -y "$(GIF_OUT)"
	@ls -lh "$(GIF_OUT)"
	@echo "5 MB 를 넘으면: T= 로 짧게(용량은 프레임 수에 비례), CROP= 으로"
	@echo "빈 영역 제거, W=560, FPS=15 순으로 줄여라. 추적 카메라는 배경이"
	@echo "매 프레임 바뀌어 GIF 프레임간 압축이 거의 듣지 않는다."

GIF_OUT ?= docs/media/trot.gif
GIF_FPS ?= 20
GIF_W   ?= 720
# macOS 의 '선택 부분 기록' 은 선택 영역 표시선을 프레임에 남길 때가 있다.
# CROP=w:h:x:y 로 가장자리를 몇 px 잘라낸다. 예: CROP=1912:1070:4:4
COMMA    := ,
# crop 은 스케일 **전** 원본 해상도에 적용된다. iw/ih 식을 쓰면 해상도를
# 몰라도 된다:  CROP=iw*0.88:ih*0.70:iw*0.06:ih*0.22
GIF_CROP  = $(if $(CROP),crop=$(CROP)$(COMMA),)
# -ss/-t 를 -i 앞에 둔다 (입력 시킹 — 훨씬 빠르다).
GIF_TRIM  = $(if $(SS),-ss $(SS),) $(if $(T),-t $(T),)

clean:  ## 캐시와 빌드 산출물 제거
	rm -rf .pytest_cache .mypy_cache build dist src/*.egg-info
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
