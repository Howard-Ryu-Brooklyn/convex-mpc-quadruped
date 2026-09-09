# 파이썬 시뮬레이션 환경·Git 치트시트


> 원본: `Cheat-Sheet.ipynb` — 저장소 정리 과정에서 노트북을 버리고 내용만 옮겼다.


# 💻 파이썬 시뮬레이션 환경 구축 및 Git 버전 관리 종합 가이드

> **목적**: 본 문서는 4족보행 로봇 최적제어 시뮬레이션 프로젝트의 개발 환경(파이썬 가상환경) 세팅과, 매번 컴퓨터 재부팅 시 확인해야 할 체크리스트, 그리고 Git/GitHub 소스코드 업로드 과정을 다른 자료 없이 이 문서 하나만 보고 해결할 수 있도록 정리한 치트시트(Cheat-sheet)입니다.

---

## 1. 🐍 파이썬 가상환경 (`.venv`) 구축 및 관리

### ⚙️ [최초 1회] 개발 환경 구축 초기 세팅
프로젝트 폴더로 이동한 뒤, 독립된 가상환경을 생성하고 필수 패키지들을 설치하는 과정입니다.

```bash
# 1. 프로젝트 폴더로 이동
cd ~/quadruped-simulation

# 2. 가상환경(.venv) 폴더 생성
python3 -m venv .venv

# 3. 터미널 가상환경 활성화 (Mac OS 기준)
source .venv/bin/activate

# 4. 주피터 노트북에 가상환경 커널 등록 (최초 1회만 수행하면 주피터 내에서 자동 연결)
python -m ipykernel install --user --name=quadruped-env --display-name "Quadruped Simulation (venv)"

# 5. pip 및 빌드 도구 최신화
pip install --upgrade pip setuptools wheel

# 6. 핵심 제어, 최적화 및 시각화 라이브러리 설치
pip install numpy scipy matplotlib casadi osqp qpsolvers ipykernel
```

### 🔄 [데일리 루틴] 컴퓨터 재부팅 시 활성화 가이드
컴퓨터를 다시 켰을 때, 터미널 환경에서 패키지를 실행하거나 추가 도구를 설치하려면 반드시 가상환경을 다시 활성화해야 합니다.

* **주피터 노트북 실행 시:** 커널이 `Quadruped Simulation (venv)`로 설정되어 있다면 별도 터미널 설정 없이 즉시 사용 가능합니다.
* **터미널에서 명령 수행 시:** 아래 명령어로 가상환경을 반드시 켜줍니다.
  ```bash
  # 프로젝트 폴더 이동 및 가상환경 활성화
  cd ~/quadruped-simulation && source .venv/bin/activate
  ```

---

## 2. 💾 새로운 코드 Git 업로드 가이드 (Routine Workflow)

코드를 수정했거나 새로운 파일(예: 새로운 제어기 소스, 그래프 이미지 등)을 추가한 뒤 GitHub에 안전하게 동기화하는 고정 루틴입니다.

### 🛠️ Git 업로드 표준 5단계 명령어

#### 1단계 : 현재 변경 상태 확인 (내가 뭘 고쳤지?)
```bash
git status
```
* 빨간색 문자열로 수정된 파일(`modified`)과 새로 추가된 파일(`Untracked`) 목록이 정확히 표시되는지 확인합니다.

#### 2단계 : 장바구니에 파일 담기 (Staging)
```bash
# 변경된 파일 '전체'를 한 번에 스테이징할 때
git add .

# 또는 특정 파일(예: 주피터 노트북 파일)만 콕 집어서 담을 때
git add simulation_notebook.ipynb
```

#### 3단계 : 로컬 저장소에 버전 기록 (Commit)
```bash
git commit -m "feat: 예측 행렬 A_qp, B_qp 직접 생성 및 QP formulation 수식 구현"
```
* 메시지 접두사 룰(`feat:`, `fix:`, `docs:`)을 지켜 작성한 핵심 기능 내용을 명확히 기록합니다.

#### 4단계 : 원격 저장소 최신화 및 충돌 방지 (⭐ 가장 중요)
```bash
git pull origin main
```
* 내 코드를 Push하기 전에, 원격 서버(GitHub)에 업데이트된 이력을 먼저 로컬로 땡겨와 병합하는 필수 안전 장치입니다.

#### 5단계 : GitHub 서버로 최종 업로드 (Push)
```bash
git push origin main
```
* 로컬 저장소의 커밋 이력이 최종적으로 GitHub 원격 저장소에 완벽하게 반영됩니다.
