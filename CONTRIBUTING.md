# 기여 안내

## 환경

```bash
python -m venv .venv && source .venv/bin/activate      # Python 3.11+
make install
```

`make install` 은 `pip install -e ".[mujoco,notebook,dev]"` 와 `nbstripout --install`
을 함께 한다. **후자를 건너뛰지 마라** — 노트북 출력(base64 PNG)이 그대로 커밋되면
저장소가 순식간에 수십 MB 커진다. 실제로 이 저장소가 그렇게 됐던 적이 있다.

`nbstripout --install` 은 `.git/config` 에 필터 명령을 박는데, 기본값이 **현재
가상환경의 절대경로**다. 그러면 다른 환경(원격 셸, CI, 다른 사람의 머신)에서
필터가 조용히 실행되지 않는다. 설치 후 아래처럼 경로 독립으로 바꿔 두는 것을 권한다.

```bash
git config filter.nbstripout.clean 'python -m nbstripout'
git config diff.ipynb.textconv    'python -m nbstripout -t'
```

## 무엇을 먼저 돌리는가

| 명령 | 언제 |
|---|---|
| `make check` | 항상. 여기 빨간불만 진짜 문제다 (~5 초) |
| `make types` | 타입을 건드렸을 때. 검사기를 안 돌리는 타입은 주석이다 |
| `make check-all` | PR 을 올리기 전. 골든 회귀까지 |
| `make accept` | **물리를 의도적으로 바꿨을 때만.** 기준선을 다시 뜬다 |

### 골든은 왜 CI 의 게이트가 아닌가

CI(`.github/workflows/ci.yml`)는 `pytest -m "not golden"` 과 `mypy` 만 돌린다.
골든이 플랫폼을 건너 재현되는지는 **측정의 문제이지 가정의 문제가 아니다.**

### 측정 결과

기준선은 macOS/arm64 에서 떴다. 그것이 다른 CPU·다른 BLAS·다른 numpy 에서도
같은 궤적을 내는가 — 재본 결과다.

| 환경 | numpy | scipy | osqp | BLAS | 결과 |
|---|---|---|---|---|---|
| macOS arm64 · py3.13.4 · homebrew venv | 2.5.1 | 1.18.0 | 1.1.3 | — | **기준선을 뜬 곳** |
| macOS arm64 · py3.12.10 · miniforge | — | — | — | — | ✅ |
| Linux x86-64 · py3.13.15 · GH Actions | 2.5.3 | 1.18.1 | 1.1.3 | scipy-openblas 0.3.34 | ✅ |
| Linux x86-64 · py3.11.16 · GH Actions | 2.4.6 | 1.17.1 | 1.1.3 | scipy-openblas 0.3.31 | ✅ |
| macOS arm64 · py3.13.15 · GH Actions | 2.5.3 | 1.18.1 | 1.1.3 | **accelerate** | ✅ |

모두 `rtol=1e-9`. 눈여겨볼 것은 마지막 두 줄이다 — **OpenBLAS 와 Apple
Accelerate 는 서로 다른 구현이다.** 커널도, 벡터화도, 누산 순서도 다르다.
여기에 numpy 2.4.6 / 2.5.3, 인터프리터 3.11~3.13, x86-64 와 arm64 가 겹쳐도
궤적이 피코미터 수준에서 같다. OSQP 의 ADMM 반복이 결정론적이고, 이 문제가
배정밀도 안에서 충분히 잘 조건화되어 있다는 뜻이다.

### 그래서 골든을 주 CI 로 올릴 것인가 — 아직 아니다

위 표에서 **osqp 는 다섯 줄 전부 1.1.3 이다.** 즉 지금까지 바꿔 본 것은
BLAS·numpy·CPU·인터프리터이지 **QP 솔버 자체가 아니다.** 궤적을 결정하는
것은 솔버의 반복이므로, osqp 2.x 가 오면 이 표는 아무것도 말해 주지 않는다.
numpy 3.x 도 마찬가지다.

한 번의 초록은 "이 조합에서 통과했다"이지 "앞으로 통과한다"가 아니다.
의존성 상한을 올릴 때마다 프로브를 다시 돌리고, 초록을 몇 번 더 본 뒤에
게이트로 승격한다. 그때 이 표가 그 판단의 근거가 된다.

`.github/workflows/golden-probe.yml` 을 수동 실행(workflow_dispatch)하면
답과 툴체인이 **run 페이지의 job 목록 아래** 에 표로 렌더된다
(job 을 클릭해 들어간 로그 화면이 아니다).
**결과를 위 표에 옮겨 적어라 — job summary 는 90 일 뒤 사라진다.**

그때까지 골든은 로컬 게이트다 — PR 전에 `make check-all`.

`make accept` 는 되돌리기 어려운 명령이다. 골든이 깨졌을 때 첫 반응이
`make accept` 라면, 그 순간 회귀 테스트는 없는 것과 같아진다.
**먼저 `scripts/compare_to_baseline.py` 로 무엇이 얼마나 바뀌었는지 보고,
그 변화가 의도한 것인지 설명할 수 있을 때만** 갱신한다.

## 테스트를 쓰는 법

단언이 **독립적으로 검증 가능한 성질**인지, **구현을 옮겨 적은 것**인지 스스로 묻는다.

- 좋음: `FK(IK(p)) == p`, 무감쇠 에너지 보존, "스윙 다리의 힘은 정확히 0",
  `θ → -θ` 일 때 `τ → -τ`
- 나쁨: "네 다리의 관절각이 서로 같다" — 실제로 이 테스트가 있었고,
  구현이 `q1 = 0` 을 네 다리에 복사했기 때문에 **통과했다.**
  테스트가 버그를 굳히고 있었다.

새 테스트를 쓰면 **그 테스트가 실패할 수 있다는 것을 먼저 증명한다**(음성 대조군).
일부러 값을 틀리게 넣어 빨간불을 한 번 본 뒤에 커밋한다. 항상 통과하는 테스트는
없는 테스트보다 나쁘다 — 있다는 착각을 주기 때문이다.

허용오차는 감으로 정하지 않는다. `scripts/measure_reproducibility.py` 로 같은 설정을
여러 번 돌려 잡음 바닥을 재고, 거기에 여유를 곱한다.

## 코드 규약

- **프레임과 단위를 이름에 넣는다.** `r_feet` 아님 → `r_feet_W`. `theta` → `theta_rad`.
- **절대 위치와 상대 벡터는 다른 필드로.** 단위가 같으면(m) 대입이 조용히 성립한다.
- **부호 상수를 넘기지 말고 대상을 넘긴다.** `compute_leg_ik(p, l_hip=±h)` 가 아니라
  `compute_leg_ik(p, leg)` — 부호는 함수가 배치에서 파생한다.
- **계산은 하는데 아무도 안 쓰는 값을 두지 않는다.** 지우거나 검증한다.
  플랫폼이 바뀌는 순간 검증되지 않은 채로 활성화된다.
- **조용한 가정은 숫자로 노출한다.** `max_phase_seen`, `max_abs_step_rad`,
  `n_event_solves`, `max_cmd_jump_m` 처럼 사후에 검사 가능한 값으로.
- **알려진 부채는 `xfail(strict=True)` 로 박제한다.** 고쳐지는 순간 XPASS 로 알려준다.
  타입도 같다 — `warn_unused_ignores = true` 라 불필요해진 `# type: ignore` 는
  그 자체가 오류가 된다.

## 의존 방향

```
core  ◀──  control  ◀──  plants  ◀──  experiments  ◀──  viz
```

`core` 는 `control` 을 모르고, `control` 은 `plants` 를 모른다. 러너만이 셋을
조립한다. 이 방향을 거스르는 import 를 넣기 전에, 정말 그 자리에 있어야 하는
코드인지 다시 본다.

가장 자주 깨지는 형태는 **그림이 알고리즘 모듈에 끼어드는 것**이다. 실제로
`control/gait_planning.py` 안에 matplotlib 을 쓰는 함수가 있었다. 아래가 비어
있는지 확인하면 된다.

```bash
grep -rn matplotlib src/quadruped_mpc/{core,control,plants,experiments}
```

## 커밋

- 무엇을 바꿨는지가 아니라 **왜 바꿨는지**를 쓴다. diff 가 전자를 이미 말한다.
- 결함을 고쳤으면 **어떻게 찾았는지**를 남긴다. 다음 사람이 같은 방법을 쓴다.
- 규칙을 우회했으면(필터를 껐거나, 검사를 건너뛰었거나) **우회했다는 사실 자체를**
  커밋 메시지에 남긴다. 우회는 우회한 자리에 흔적을 남기지 않는다.

## README 의 그림 만들기

```bash
which ffmpeg || brew install ffmpeg
make view S=G_trot                      # 카메라가 몸통을 따라간다
# Tab / Shift+Tab 으로 UI 패널을 접는다 (F1 이 단축키 목록)
# Cmd+Shift+5 -> 선택 부분 기록 -> 4~6 초 -> 정지
make gif IN=~/Desktop/화면\ 기록\ ....mov
```

- 재생 카메라는 **기본이 몸통 추적**이다. 자유 카메라(`--free-cam`)로 두면
  1 m/s 로 걷는 로봇이 몇 초 만에 화면 밖으로 나간다. 거리는 `--cam-dist`.
- 4~6 초면 trot 주기(0.5 s) 기준 8~12 보다. 걸음이 반복된다는 것이 보이면
  충분하고, 그 이상은 파일만 커진다.
- macOS 의 '선택 부분 기록' 은 **선택 영역 표시선을 프레임에 남길 때가 있다.**
  `CROP=w:h:x:y` 로 가장자리를 몇 px 잘라낸다.
### 용량

**추적 카메라를 켜면 GIF 가 크게 불어난다.** GIF 는 프레임 사이에 변하지
않은 픽셀을 건너뛰어 압축하는데, 카메라가 로봇을 따라가면 배경이 매 프레임
스크롤해서 그 압축이 통째로 무력해진다. 고정 카메라 4.4 MB 짜리가 추적
카메라에서 18 MB 가 됐다. 그림은 좋아지고 파일은 나빠지는 맞교환이다.

줄이는 순서 — **효과가 큰 것부터**:

1. `T=5` — 용량은 프레임 수에 비례한다. 가장 확실하다. 재녹화 없이 자른다.
2. `CROP=` — 창 타이틀바, 검은 여백, 빈 하늘을 걷어낸다. 해상도를 몰라도
   되게 `iw`/`ih` 식을 쓴다: `CROP=iw*0.88:ih*0.70:iw*0.06:ih*0.22`
3. `W=640` 또는 `560`
4. `FPS=15`

`SS=` 로 시작 지점도 고를 수 있다 — 처음 1~2 초는 보통 카메라를 맞추는
구간이라 버리는 게 낫다.

5 MB 를 넘기지 않는 것을 목표로 한다. GitHub 는 더 큰 것도 띄우지만,
README 첫 화면에서 수 MB 를 내려받게 하는 것은 예의가 아니다.
