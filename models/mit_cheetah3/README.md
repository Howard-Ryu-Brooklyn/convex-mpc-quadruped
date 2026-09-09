# `mit_cheetah3` — 이름과 내용이 반씩 다른 모델

**동역학은 MIT Cheetah 3, 겉모습은 Unitree A1이다.** 이 폴더를 처음 보는 사람이
가장 먼저 오해하는 지점이라 맨 위에 적어 둔다.

| 무엇 | 출처 |
|---|---|
| 몸통 질량 43 kg, 관성 `diag(0.41, 2.1, 2.1)` | MIT Cheetah 3 (Di Carlo et al., IROS 2018) |
| 다리 길이·관절 한계·토크 한계 250 N·m | 이 프로젝트에서 직접 작성 |
| `assets/*.obj`, `assets/*.png` (시각 메시) | Unitree A1 description, BSD-3-Clause |

시각 메시는 **그림에만 쓰인다.** 충돌 지오메트리와 관성은 전부 `mit_cheetah3.xml`
안에 명시돼 있고 메시에서 파생되지 않는다. 따라서 화면에 보이는 다리 두께와
시뮬레이션이 푸는 다리 질량은 서로 무관하다.

## 이 모델이 SRB 플랜트와 일부러 다른 점

`quadruped_mpc.plants.model_audit` 가 두 모델을 대조해서, **의도한 차이만**
통과시키고 나머지는 예외를 던진다. 의도한 차이는 8개이고 그중 중요한 둘:

- **마찰 원뿔**: MPC는 사각뿔(선형 제약), MuJoCo는 `cone="elliptic"`.
  MPC가 푸는 제약이 실제 제약의 외부 근사라, 대각 방향에서 미끄러질 수 있다.
- **관절 감쇠·마찰**: MuJoCo에만 있다(`damping`, `frictionloss`, `armature`).
  SRB에는 없다. 이 차이가 곧 "이상적 모델이 무시한 것"의 크기다.

전체 목록과 각각의 이유는 `model_audit.py` 를 보라. 감사에 없는 차이가 생기면
`UnexpectedModelMismatch` 로 즉시 실패한다 — 모델이 조용히 바뀌는 것을 막는 장치다.

## 라이선스

메시와 텍스처는 Unitree Robotics의 BSD 3-Clause (같은 폴더의 `LICENSE`).
MJCF와 그 밖의 모든 것은 저장소 루트의 MIT. 자세한 것은 루트의 `NOTICE`.
