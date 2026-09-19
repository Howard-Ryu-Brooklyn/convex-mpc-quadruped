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

지금까지 측정된 것 (둘 다 macOS/arm64):

| | 환경 |
|---|---|
| 기준선을 뜬 곳 | Python 3.13.4, homebrew venv, numpy 2.5.1, osqp 1.1.3 |
| 재현한 곳 | Python 3.12.10, miniforge base, 다른 numpy·osqp 빌드 |

**rtol=1e-9 로 통과했다.** 인터프리터와 빌드가 다른데도 피코미터 수준에서 같다.

측정되지 **않은** 것: Linux/x86-64. numpy 가 링크하는 BLAS 와 OSQP 휠이 다르다.
통과할 수도 있다. `.github/workflows/golden-probe.yml` 이 그것을 재기 위해 있고,
수동 실행(workflow_dispatch)이다 — 답을 모르는 질문을 주 워크플로의 배지에
올리면, 아무도 조치할 수 없는 빨간불이 생긴다. **그런 빨간불은 없는 검사보다
나쁘다.** 옆에 있는 진짜 빨간불까지 함께 무시되기 때문이다.

프로브를 돌렸으면 **답을 이 표에 적어라.** 그것이 이 절의 용도다.

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
