# 액티브러닝 실험 조건 최적화 시스템

> 본 문서의 모든 수치는 저장소에서 실제 명령을 실행해 얻었으며, 각 항목에 획득 명령을 병기했다.
> 확인되지 않은 항목은 추측하지 않고 "확인 불가"로 남겼다.

---

## 개요

| 항목 | 내용 | 획득 명령 |
|---|---|---|
| **기간** | 2026-07-15 ~ 2026-07-30 (16일) | `git log --reverse --format='%ad' --date=iso \| head -1`, `git log -1 --date=iso` |
| **총 커밋** | 36 (전량 2026-07) | `git rev-list --count HEAD`, `git log --format='%ad' --date=format:'%Y-%m' \| sort \| uniq -c` |
| **누적 변경량** | +3,431 / −1,790 (순증 +1,641) | `git log --pretty=tformat: --numstat \| awk '{a+=$1;r+=$2} END{print a,r,a-r}'` |
| **현재 코드 규모** | `app.py` 1,154줄 + `origin_charts.py` 468줄 = **1,622줄** | `wc -l $(git ls-files '*.py')` |
| **상태** | 운영 중 (배포 완료, 실제 실험 데이터 투입됨) | 저장소 내 실제 데이터 `FAPbI3_Jiwon_Data .xlsx` 존재 |
| **배포 URL** | nbedl-exp-assistant.streamlit.app | 사용자 제공 · `git remote -v` → `github.com/jun0land/NBEDL-Exp-Assistant` |
| **배포 방식** | Streamlit Community Cloud (main 브랜치 자동 배포) | 배포 로그 (`Cloning repository... branch: 'main', main module: 'app.py'`) |

### 커밋 분포 (일자별)

```
07-15  19커밋   초기 구축 + UI 반복 (5da6dc7 "시스템 1차 완성본 업로드" → 7fef7f5)
07-16   3커밋   사용 설명서 기능
07-17   2커밋   배경 이미지 최적화 (PNG 8.5MB → WebP 82KB)
07-22   4커밋   다중 목표 지표 + MOBO 도입
07-23   1커밋   UI
07-25   2커밋   Origin 스타일 차트 모듈 분리 (origin_charts.py 신규 468줄)
07-27   3커밋   차트 기능 확장
07-30   2커밋   CPU-only torch / 연산 부하 대응
```
*획득: `git log --reverse --format='%ad|%h|%s' --date=format:'%m-%d %H:%M' --shortstat`*

---

## 문제 정의 — 기존 전수탐색·감 의존 방식의 병목

페로브스카이트(FAPbI₃) 광검출기 공정 최적화에서, 실험 1회는 박막 제작 → 소자 측정 → 지표 산출까지 물리적 시간과 재료를 소모한다. 여기에 세 가지 병목이 있었다.

1. **탐색 공간 대비 실험 예산 부족.** 실제 운영 데이터 기준 공정 변수 `안티솔벤트 Drop`은 Integer 0~60초 범위다. 격자 전수탐색은 1변수만으로도 61조건, 반복측정 2회를 곱하면 122회 실험이 된다. 실제 축적된 데이터는 **8행**(4조건 × 2반복)에 불과했다.
   *획득: `Config_Vars` 시트 `Type=Integer (정수), Min=0, Max=60` / `Data` 시트 8행 — `zipfile`로 xlsx 직접 파싱*

2. **목표 지표가 다수이고 서로 트레이드오프 관계.** 실제 목표는 바이어스·파장 조합별 Responsivity 4종 — `R(+1 V)_940`, `R(+1 V)_850`, `R(-1 V)_530`, `R(-1 V)_470` (전부 A/W, 전부 Maximize). 운영 중 최대 8개까지 확장됐다. 하나를 올리면 다른 게 내려가는 구조에서 "감"으로는 균형점을 잡을 수 없다.
   *획득: `Target_Vars` 시트 파싱 결과 4행 / 사용자 보고 "목표지표를 8개까지"*

3. **반복 측정의 산포를 사람이 눈으로 걸러냈다.** 동일 조건 반복(`5s_1`, `5s_2` 같은 샘플명 규칙)에서 튀는 값을 배제하는 기준이 주관적이었다.

---

## 사용한 AI 도구·모델·인터페이스

| 구분 | 내용 | 근거 |
|---|---|---|
| 도구 | **Claude Code** (CLI/IDE 에이전트) | `~/.claude/` 세션 기록 존재 |
| 모델 | `claude-sonnet-5` (설정값), 세션 말미 `opus`로 전환 | `~/.claude/settings.json` → `{"model": "claude-sonnet-5", ...}` |
| 인터페이스 | VS Code 확장 | 세션 환경 메타데이터 |
| 세션 규모 | **1개 세션 · 1,551 레코드 · 17MB** (user 405 / assistant 700) | `~/.claude/projects/c--Users-mintj-NBEDL-Exp-Assistant/*.jsonl` 파싱 |
| 세션 기간 | 2026-07-22 ~ 2026-08-03 (활동 5일) | 동 jsonl 타임스탬프 |
| 툴 호출 | Bash 171 / Read 77 / Edit 52 / Grep 23 / Write 15 / TaskOutput 17 | 동 jsonl `tool_use` 집계 |

### Claude Code 커스터마이징 현황 (전부 미사용)

| 항목 | 상태 | 획득 명령 |
|---|---|---|
| `CLAUDE.md` | **없음** | `ls CLAUDE.md` → No such file |
| 프로젝트 `.claude/` | **빈 디렉터리** | `ls -laR .claude` → 파일 0개 |
| 커스텀 슬래시 커맨드 | **없음** | `ls ~/.claude/commands` → 디렉터리 자체 없음 |
| 커스텀 스킬 | **없음** | `ls ~/.claude/skills` → 없음 |
| 서브에이전트 | **없음** | `ls ~/.claude/agents` → 없음 |
| 훅(hooks) | **없음** | `~/.claude.json` 파싱 → `hooks: {}` (전역·프로젝트 모두 비어 있음) |
| MCP 서버 | **없음** | `~/.claude.json` 파싱 → `mcpServers: {}` |
| 전역 설정 | `model`, `autoUpdatesChannel`, `theme` 3개 키뿐 | `cat ~/.claude/settings.json` (86 bytes) |

> **해석:** 프레임워크 설정 없이 순수 대화형 세션만으로 구축했다. 이는 재현성·온보딩 관점에서는 개선 여지(= `CLAUDE.md`로 도메인 컨텍스트를 고정하지 않아 매 세션 재설명 비용 발생)로 읽힌다.

### AI 관여 구간 분리

세션 최초 레코드는 2026-07-22 18:09(KST)다. 이를 기준으로 커밋을 나누면:

| 구간 | 커밋 수 | 성격 |
|---|---|---|
| 세션 이전 (07-15 ~ 07-22 16:20) | **24** | 초기 구축·UI 반복 |
| 세션 이후 (07-22 20:01 ~ 07-30) | **12** | 다중목표 최적화 엔진·차트 모듈·성능 대응 |

세션 구간 코드 증가량: **+955 / −122** (`origin_charts.py` 468줄 신규 포함)
*획득: `git log --since='2026-07-22 18:00' --oneline | wc -l`, `git diff --stat a8e84ce HEAD`*

---

## 동기

이 시스템은 두 개의 상위 도구가 만든 데이터의 **종착점**으로 설계됐다.

측정 장비(Keithley)에서 나온 원시 I-V 데이터는 `photodetector-app`이 파싱해 소자 성능지표(Responsivity, Detectivity)를 계산한다. 그런데 그 지표들이 **"그래서 다음에 어떤 공정 조건으로 찍어야 하나"**로 이어지지 않고 엑셀에 쌓이기만 했다. 지표는 잘 계산되는데 그 지표를 **실험 설계 의사결정으로 환류시키는 고리**가 비어 있었던 것이 직접적 동기다.

부수적으로, `origin_charts.py`가 `photodetector-app`의 Origin 그래프 규격(10×8인치=960×768px, simple_white, 검정 mirror 축, `exponentformat="E"`, Myriad Pro, ORIGIN_COLORS 24색)을 그대로 이식한 것도 같은 맥락 — 논문용 산출물 형식을 도구 간에 통일해 재작업을 없애려는 의도다.
*획득: `origin_charts.py` 1~11행 docstring, `pd_app/figure.py`·`pd_app/constants.py` 대조*

---

## 워크플로우

```
[Keithley 장비]  I-V 원시 측정 (.xls, AnodeV/AnodeI 컬럼)
      │
      ▼
[photodetector-app]  파싱 → 파장별 성능지표 계산
      │              R = (|I_light| − |I_dark|) / (E_e[W/cm²] × Area[cm²])
      │              D* = R·√A / √(2q·I_dark)
      │              내보내기 컬럼: "R_{-1V} (A/W)", "D*_{-1V} (Jones)" …
      ▼
[NBEDL Exp Assistant]  ← 본 시스템
      │
      ├─ ① Setup: 공정 변수(탐색 대상) + 목표 지표(방향·단위) + 환경 변수 정의
      ├─ ② 신규 실험 입력: 샘플명 + 환경값 + 공정조건 + 목표 실측값 (8자리 소수)
      ├─ ③ DB 관리: 전체 이력 표 + "학습 적용" 체크로 수동 배제
      ├─ ④ AI 대시보드: 목표 개수에 따라 자동 분기 → 차기 실험 후보 3건 추천
      └─ ⑤ Origin 스타일 분포 그래프 + PNG/JPG/CSV 내보내기 (300dpi)
      │
      ▼
[Excel 4시트 저장]  Data / Config_Vars / Target_Vars / Config_Meta
      │
      └──── 재업로드 시 설정·데이터 전체 복원 → 다음 사이클로 순환
```

*획득: `app.py` 탭 구조(`st.tabs(["📝 신규 실험 입력", "🗂️ 데이터베이스 관리", "🤖 AI 최적화 대시보드"])`), `build_excel_bytes()` 723~734행, `load_excel_data()` 735행~*

---

## 핵심 알고리즘 (입력 → surrogate → acquisition → 추천 조건 출력)

### 경로 분기 구조

목표 지표 개수에 따라 **3개 경로로 자동 분기**한다. 분기 임계값은 상수로 노출돼 있다.

```python
MOBO_HYPERVOLUME_MAX_OBJECTIVES = 4  # app.py:603
```

| 목표 개수 | Surrogate | Acquisition | 라이브러리 | 코드 위치 |
|---|---|---|---|---|
| **1개** | GP | **EI** (Expected Improvement) | scikit-optimize | `app.py:1047` |
| **2~4개** | `SingleTaskGP` × N → `ModelListGP` | **qLogNEHVI** (Log Noisy Expected Hypervolume Improvement) | BoTorch | `app.py:683` |
| **5개 이상** | 동일 | **qLogNEI + 체비셰프 스칼라화** (ParEGO 방식) | BoTorch | `app.py:701` |

### ① 입력 전처리

```python
# app.py:562 process_robust_data / :580 process_robust_data_multi
grouped = df.groupby(feature_cols)          # 동일 공정조건끼리 묶음
if len(y_vals) >= 3:                        # 반복 3회 이상일 때만 이상치 판정
    q1, q3 = np.percentile(y_vals, [25, 75])
    lower, upper = q1 - 1.5*iqr, q3 + 1.5*iqr
    valid_y = [y for y in y_vals if lower <= y <= upper]
    if not valid_y: valid_y = y_vals        # 전부 이상치면 원복 (데이터 소실 방지)
robust_y.append(np.mean(valid_y))           # 조건당 대표값 = 강건 평균
```

### ② Surrogate

- **단일목표:** `Optimizer(dimensions=ai_spaces, base_estimator="GP", acq_func="EI", random_state=None)`
- **다중목표:** 목표별로 독립 GP를 세우고 리스트로 결합
  ```python
  models = [SingleTaskGP(X_norm, Y_adj[:, i:i+1], outcome_transform=Standardize(m=1))
            for i in range(Y_adj.shape[-1])]          # app.py:668
  model = ModelListGP(*models)
  mll = SumMarginalLogLikelihood(model.likelihood, model)
  fit_gpytorch_mll(mll)                               # app.py:672
  ```
  - X 정규화: `normalize(X_raw, bounds=bounds_raw)` (박스 경계 기준 0~1)
  - Y 표준화: `Standardize(m=1)` (GP outcome transform)
  - 방향 통일: `sign = ±1`로 Minimize 목표의 부호를 반전해 전부 최대화 기준으로 맞춤 (`app.py:663`)

### ③ Acquisition

**2~4개 목표 — 정확한 하이퍼볼륨 기반**
```python
ref_point = Y_adj.min(dim=0).values - 0.1 * y_range   # 관측 최소값에서 10% 여유
acq = qLogNoisyExpectedHypervolumeImprovement(
    model=model, ref_point=ref_point.tolist(), X_baseline=X_norm,
    sampler=SobolQMCNormalSampler(torch.Size([64])), prune_baseline=True)
candidates_norm, _ = optimize_acqf(
    acq, bounds=standard_bounds, q=3,
    num_restarts=5, raw_samples=128,
    options={"batch_limit": 5, "maxiter": 200}, sequential=True)
```

**5개 이상 — ParEGO 스칼라화 (Knowles 2006)**
```python
for i in range(n_candidates):                         # 후보마다 다른 가중치
    gen = torch.Generator().manual_seed(i)
    weights = torch.rand(n_obj, generator=gen, dtype=torch.double)
    weights = weights / weights.sum()                 # 합=1로 정규화
    scalarized_obj = GenericMCObjective(
        get_chebyshev_scalarization(weights=weights, Y=Y_adj))
    acq = qLogNoisyExpectedImprovement(
        model=model, X_baseline=X_norm, objective=scalarized_obj,
        sampler=sampler, prune_baseline=True)
    cand, _ = optimize_acqf(acq, bounds=standard_bounds, q=1, ...)
```
후보마다 랜덤 가중치가 다르므로 파레토 프론트의 **서로 다른 지점**을 겨냥한다 → 하이퍼볼륨 계산 없이도 트레이드오프 다양성을 유지.

### ④ 출력

- 차기 실험 후보 **3건**, 각 후보의 공정 조건값 + **예측 목표값**(GP posterior mean, 원 단위 역변환)
- `unnormalize()` → `decode_x()`로 Integer는 반올림, Categorical은 인덱스→라벨 복원
- 직전 결과와 동일하면 "AI 수렴 상태 판단" 안내 표시 (목표별로 따로 기억: `prev_next_points_by_target`)

### 탐색 변수 정의

| 타입 | 범위 지정 | 기본값 | 인코딩 |
|---|---|---|---|
| `Real (실수)` | Min/Max (`number_input`, step 미지정) | 0.0 / 10.0 | 그대로 |
| `Integer (정수)` | Min/Max (`step=1`) | 0 / 100 | 최적화 후 `int(round())` |
| `Categorical (범주)` | 쉼표 구분 옵션 문자열 | — | 옵션 인덱스 0..k-1로 연속 인코딩 후 반올림 복원 |

**제약조건:** 박스 경계(Min~Max)만 존재. 결합 제약(선형/비선형)은 **미구현**
*획득: `grep -c "inequality_constraints\|equality_constraints\|nonlinear_inequality" app.py` → **0건***

### 목적함수 정의

- 지표별로 `Maximize`/`Minimize` 개별 선택 (`Target_Vars` 시트의 `Direction` 컬럼)
- **가중치: 사용자 지정 없음.** ParEGO 경로에서만 내부적으로 랜덤 체비셰프 가중치 사용
- 정규화: 최적화용은 `Standardize(m=1)`(Y) + `normalize()`(X). 시각화용은 별도로 min-max 0~1 (`origin_charts._normalize`), 이는 UI에서 끌 수 있음

### Cold start 처리

```python
if len(valid_df) < 2:                                  # app.py:1022, 1111
    st.warning("정밀 분석을 위해 최소 2개 이상의 유효 데이터가 필요합니다.")
```
- 단일목표 경로의 초기 랜덤 표본 수(`n_initial_points`)는 **명시 지정 없음** → skopt 기본값 사용. 정확한 기본값은 로컬에 skopt 미설치라 **확인 불가**
- 추가 가드: 유효 데이터가 전부 변수 범위 밖이면 학습셋이 비어 skopt 내부 `np.argmin([])`가 죽으므로, 사전 차단 후 원인 메시지 표시 (`app.py:1041~1052`)

### 재현성 확보 장치

| 경로 | 시드 | 재현성 |
|---|---|---|
| 다중목표 (MOBO) | `torch.manual_seed(0)` (`app.py:628`) + `Generator().manual_seed(i)` (`app.py:697`) | **재현 가능** |
| 단일목표 (skopt) | `random_state=None` (`app.py:1047`) | **재현 불가** — 실행마다 결과 달라짐 |

> 이 비대칭은 의도된 설계가 아니라 **미정리 상태**로 보인다. 단일목표 경로에도 시드를 고정하는 편이 일관적이다.

**상태 저장:** Excel 4시트 왕복(내보내기/불러오기)으로 설정·데이터 전체 복원. `@st.cache_data(max_entries=3)`로 매 리런 재생성 방지 (`app.py:723`)

---

## 설계 트레이드오프와 실패 기록

### 1. 목표 8개에서 연산 폭발 → CPU 스로틀 (가장 큰 실패)

**증상.** Streamlit Cloud에서 스로틀 제재:
> *"Your app has been throttled — we've temporarily reduced its CPU to keep the platform healthy for everyone."*

**원인 규명.** 추측 대신 목표 개수를 늘려가며 실측했다.

| 목표 개수 | qLogNEHVI 실행 시간 |
|---|---|
| 2개 | **1.95초** |
| 4개 | **19.57초** (10배) |
| 8개 | **8분 경과 후에도 미완료** (측정 중단) |

하이퍼볼륨은 목표 차원이 늘수록 파레토 분할 계산이 조합적으로 폭발한다. MC 샘플 수·재시작 횟수를 줄이는 정도로는 해결되지 않는 **알고리즘 자체의 한계**였다.

**대응.** 목표 5개 이상이면 하이퍼볼륨 계산이 아예 없는 ParEGO 스칼라화로 자동 전환.

| 목표 개수 | 전환 후 | 경로 |
|---|---|---|
| 4개 | 27.62초 | qLogNEHVI (유지) |
| 5개 | **5.98초** | ParEGO |
| 8개 | **7.54초** | ParEGO |

**트레이드오프.** ParEGO는 파레토 프론트를 정확히 넓히는 것이 아니라 랜덤 가중 방향으로 근사한다 — **탐색 품질 일부를 포기하고 실행 가능성을 얻은 것.** 목표 4개 이하에서는 정확한 방식을 유지해 손해를 최소화했다.

**도입 커밋:** `21f4332` "lose weight" (2026-07-30 17:48, +65/−18)
*획득: `git show 21f4332 -- app.py`*

주요 diff:
```diff
-    from botorch.acquisition.multi_objective.monte_carlo import qNoisyExpectedHypervolumeImprovement
+    from botorch.acquisition.multi_objective.logei import qLogNoisyExpectedHypervolumeImprovement
+    from botorch.acquisition.logei import qLogNoisyExpectedImprovement
+    from botorch.acquisition.objective import GenericMCObjective
+    from botorch.utils.multi_objective.scalarization import get_chebyshev_scalarization
...
+MOBO_HYPERVOLUME_MAX_OBJECTIVES = 4
...
-    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([128]))
+    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([64]))
-        num_restarts=10, raw_samples=256,
+            num_restarts=5, raw_samples=128,
+    if n_obj <= MOBO_HYPERVOLUME_MAX_OBJECTIVES:   ... qLogNEHVI
+    else:                                          ... ParEGO 루프
```

동시에 적용한 부수 대응 2건:
- `torch.set_num_threads(1)` — 컨테이너 CPU 쿼터가 1코어 안팎인데 torch는 호스트 전체 코어만큼 스레드를 띄워 경합이 발생, 피크 사용률을 부풀림
- qNEHVI → **qLogNEHVI** 전환 — BoTorch 자체가 수치 불안정을 경고(`NumericsWarning ... strongly recommended to simply replace`)하던 것을 해소

> **측정 신뢰도 캐비앳(정직한 한계):** 로컬 측정 시 `Failed to compile fused qLogEHVI C++ extension: Ninja is required ... To get ~3x speedup` 경고가 있었다. 즉 **로컬 qLogNEHVI 수치는 순수 Python 폴백 기준**이며, ninja가 설치된 환경(배포 로그상 `ninja==1.13.0` 포함)에서는 최대 3배 빠를 수 있다. 다만 3배를 적용해도 목표 8개의 미완료 상황은 뒤집히지 않는다.

### 2. torch가 CUDA 풀스택으로 설치돼 배포 비대화

**증상.** 앱이 뜨지 않는데 로그에 트레이스백이 없음(= 예외가 아니라 프로세스 강제 종료 정황).

**원인.** 배포 로그에서 GPU 패키지가 대량 설치된 것을 확인:
```
+ torch==2.13.0    + cuda-toolkit==13.0.3.0   + nvidia-cublas==13.1.1.3
+ cuda-bindings    + nvidia-cudnn-cu13        + nvidia-cufft ... (nvidia-* 10여 개)
```
Streamlit Community Cloud에는 GPU가 없는데 PyPI 기본 인덱스가 CUDA 포함 풀버전을 설치 → 수 GB로 불어나 메모리 한계 초과.

**대응.** `requirements.txt`에 CPU 전용 휠 인덱스 명시 (커밋 `f3da375`)
```
--extra-index-url https://download.pytorch.org/whl/cpu
torch
botorch
```
**검증:** 재설치 결과 `torch-2.13.0+cpu`, `nvidia-*`/`cuda-*` 패키지 **0개**

**부수 실패:** 이 주석을 처음에 한글로 작성했다가 Windows 로캘(cp949)에서 `UnicodeDecodeError`로 pip이 죽음 → requirements.txt는 ASCII로만 유지하도록 수정.

### 3. `학습_적용` 체크박스 컬럼 dtype 오염 → 앱 전체 크래시

**증상.** `StreamlitAPIException` (배포 로그 트레이스백, `app.py:902 st.data_editor`)

**원인.** Excel 왕복 중 `학습_적용` 컬럼에 `NaN`/문자열 `"TRUE"`/숫자 `1` 등이 섞이면 `CheckboxColumn`이 타입 불일치 예외를 던진다. 더 나쁜 건 **`st.tabs()`는 보이지 않는 탭 코드도 매 리런 실행**하므로, 해당 탭을 열지 않아도 앱 전체가 죽고 그 와중에 누른 Excel 다운로드도 정상 파일을 못 받는다.

**대응.** `coerce_bool_col()` 방어 함수 추가, 대시보드 진입 시마다 정규화 (커밋 `d4ae286`)
```python
def coerce_bool_col(series):        # app.py:539
    if pd.api.types.is_bool_dtype(series): return series.fillna(True)
    ...  # NaN→True, "false"/"0"/""→False, 그 외 문자열→True
```
표시 직전뿐 아니라 **필터링 전에도** 통과시켜야 `df["학습_적용"] == True` 비교가 조용히 틀리는 것을 막는다.

### 4. `Target_Vars` 시트와 `Data` 컬럼 순서 불일치

**증상.** "마지막 목표지표가 맨 위에 나온다"는 사용자 보고. 코드에는 순서를 뒤집는 로직이 없어 재현 테스트를 2회 했으나 정상 — 실제 파일을 받아 원인 확인.

**원인.** 실제 파일에서 두 시트의 순서가 정확히 역순이었다.
- `Data` 컬럼: 940 → 850 → 530 → 470 (최초 생성 시 고정, 이후 불변)
- `Target_Vars` 행: 470 → 530 → 850 → 940 (엑셀에서 사람이 보기 좋게 재정렬)

**대응.** 순서를 단순히 뒤집는 대신, 화면 표시를 **항상 `Data` 컬럼 순서에 동기화** (커밋 `4e3b7e6`)
```python
_data_cols = list(st.session_state.df_data.columns)        # app.py:894
st.session_state.target_vars.sort(
    key=lambda tv: _data_cols.index(tv["Name"]) if tv["Name"] in _data_cols else len(_data_cols))
```

### 5. 소수점 절삭으로 실측값 손실

`0.10400548` 같은 Responsivity 값이 Streamlit `number_input` 기본 포맷(소수 2자리)으로 잘릴 수 있었다 → `format="%.8f", step=0.000001` 명시, MOBO 예측 표시도 `.3g` → `.6f`로 확장.

---

## 정량 결과

### 검증된 수치

| 지표 | 개선 전 | 개선 후 | 측정 방법 |
|---|---|---|---|
| **목표 8개 추천 연산** | 8분+ 미완료 | **7.54초** | `run_mobo` 직접 호출 타이밍 (합성 데이터 30행 × 목표 8개 × 변수 3개) |
| 목표 5개 추천 연산 | (동 경로에서 폭발) | **5.98초** | 동일 |
| 목표 2개 / 4개 | 1.95초 / 19.57초 | 유지 (정확 경로) | 동일 |
| **배포 의존성** | CUDA 풀스택 (nvidia-* 10여 개) | `torch-2.13.0+cpu`, GPU 패키지 **0개** | `pip install -r requirements.txt` 후 설치 목록 대조 |
| 배경 이미지 | PNG 4096px / 8.5MB (base64 ≈11MB) | WebP 2048px / **82KB** (PSNR 40.5dB) | `app.py:59~63` 주석 + `git ls-files` |

### 탐색 효율 (구조적 근거)

- 격자 전수탐색: `안티솔벤트 Drop` 0~60초(Integer) 단독으로 **61조건**, 반복 2회 시 122회 실험
- 실제 축적 데이터: **8행** (4조건 × 2반복)
- 즉 전수탐색 대비 **약 7%의 실험량**으로 4개 목표 동시 최적화 사이클을 운용

> **확인 불가:** 이 시스템 도입으로 실제 목표 성능에 도달하기까지 **몇 회의 실험이 절감됐는지**는 실험 결과 로그(도입 전후 비교군)가 저장소에 없어 확인 불가. 위 수치는 탐색 공간 대비 투입 실험량의 비율일 뿐 성능 도달 속도를 뜻하지 않는다.

### 개발 생산성

| 지표 | 값 |
|---|---|
| 총 개발 기간 | 16일 (2026-07-15 ~ 07-30) |
| 총 커밋 | 36 |
| 최종 코드 | 1,622줄 (Python 2개 모듈) |
| AI 세션 구간 코드 증가 | +955 / −122 |
| AI 세션 구간 커밋 | 12 / 36 (33%) |

---

## 객관적 증거

### 저장소 구성 (`git ls-files`)
```
.gitignore
.streamlit/config.toml          # enableStaticServing = true
app.py                          # 1,154줄 — UI + 최적화 엔진
logo.png
origin_charts.py                #   468줄 — Origin 스타일 차트 + 제외데이터 + 내보내기
requirements.txt                #    12줄
static/fonts/MyriadPro-Regular.otf
static/liquid_bg.webp
```

### `requirements.txt` 전문 (버전 포함)
```
streamlit
pandas
numpy
openpyxl
scikit-optimize
streamlit-extras
plotly

# CPU-only torch (Streamlit Community Cloud has no GPU; default PyPI torch pulls
# in the full CUDA stack, which is several GB and can blow past memory limits).
--extra-index-url https://download.pytorch.org/whl/cpu
torch
botorch
```
> 소스에는 버전 핀이 없다. 실제 해결된 버전은 배포 로그 기준:
> `streamlit==1.60.0`, `botorch==0.18.1`, `torch==2.13.0`, `gpytorch==1.15.2`,
> `scikit-optimize==0.10.2`, `pandas==3.0.5`, `numpy==2.5.1`, `plotly==6.9.0`,
> `openpyxl==3.1.5`, `scikit-learn==1.9.0`, `scipy==1.18.0`, `streamlit-extras==1.6.0`
> (Python 3.14.6 환경, uv로 86패키지 설치)
>
> **리스크:** 버전 미고정 상태라 상위 라이브러리 업데이트 시 재현성이 깨질 수 있다.

### 주요 모듈 역할

| 파일 | 역할 |
|---|---|
| `app.py` | Setup/Dashboard 2화면 + 3탭. 데이터 전처리(IQR), 단일/다중목표 최적화 엔진, Excel 4시트 I/O, Liquid Glass CSS, 사용 설명서 드로어 |
| `origin_charts.py` | 공정변수별 Origin 규격 Plotly 그래프(포인트+다항 추세선), 제외 데이터 익스팬더, 클라이언트 사이드 PNG/JPG 300dpi 내보내기, Myriad Pro `@font-face` 번들 |

### 데이터 스키마 (실제 운영 파일 `FAPbI3_Jiwon_Data .xlsx` 직접 파싱)

**시트 4종:** `Data`, `Config_Vars`, `Target_Vars`, `Config_Meta`

| 시트 | 컬럼 |
|---|---|
| `Data` | `학습_적용`(bool), `샘플명`, `온도 (°C)`, `습도 (%)`, `안티솔벤트 Drop`, `R(+1 V)_940`, `R(+1 V)_850`, `R(-1 V)_530`, `R(-1 V)_470` — 8행 |
| `Config_Vars` | `Old_Name`, `Name`, `Unit`, `Type`, `Min`, `Max`, `Options` → 1행 (`안티솔벤트 Drop`, sec, Integer, 0~60) |
| `Target_Vars` | `Old_Name`, `Name`, `Direction`, `Unit` → 4행 (전부 `Maximize`, `A/W`) |
| `Config_Meta` | `Exp_Name`=`FAPbI3_Jiwon`, `Passive_Vars`=`온도 (°C),습도 (%)` |

*획득: `zipfile`로 xlsx 직접 열어 `xl/workbook.xml`·`xl/sharedStrings.xml` 파싱*

**컬럼 역할 구분 3종:**
- **공정 변수(config_vars)** — AI가 탐색하는 대상. 범위·타입 지정
- **환경 변수(passive_vars)** — 기록만 하고 최적화에 미사용 (온도/습도)
- **목표 지표(target_vars)** — 최적화 대상. 방향·단위 지정

**조건당 반복 데이터 처리:** 동일 공정조건을 `groupby`로 묶어 반복 3회 이상이면 IQR 1.5배 밖 배제 후 평균, 2회 이하는 전량 평균. 샘플명은 `5s_1`/`5s_2`처럼 조건+반복 인덱스 규칙으로 관리.

### 상위 도구 연계 (증거 기반)

`origin_charts.py` docstring이 이식 출처를 명시:
> *"photodetector-app 의 오리진 형식을 그대로 이식했다: 그래프 크기·축 스타일 pd_app/figure.py, 팔레트 pd_app/constants.ORIGIN_COLORS, 클라이언트 사이드 이미지 내보내기 pd_app/ui/summary.py"*

`photodetector-app` 실측 확인:
- 입력: 구형 **Keithley `.xls`** (`AnodeV`/`AnodeI` 컬럼) — `pd_app/parsing.py:56`
- 출력: `R_{-1V} (A/W)`, `D*_{-1V} (Jones)` 등 파장별 성능지표 CSV — `pd_app/ui/summary.py:275`
- 계산식 (`summary.py:135~165`):
  ```python
  i_ph1 = abs(i_light1) - i_dark_abs1
  R1 = i_ph1 / (ee_w * area_cm2)                                   # Responsivity [A/W]
  D1 = R1 * math.sqrt(area_cm2) / math.sqrt(2.0 * _Q_ELECTRON * i_dark_abs1)  # D* [Jones]
  ```

→ 본 시스템의 목표 지표 `R(+1 V)_940` 등이 이 파이프라인의 산출물임이 명명 규칙·단위(A/W)로 확인된다.

**관련 저장소 세션 규모** (`du -sh ~/.claude/projects/*`):
```
NBEDL-Exp-Assistant   17M    ← 본 시스템
Kiethly-4200         4.0M    ← 장비(Keithley 4200) 측
raman-mapping        2.2M
photodetector-app    1.6M    ← 상위 도구
equip-system          20K
```
*단, `C:\Users\mintj\Kiethly-4200` 디렉터리는 현재 경로에 없어 내용 **확인 불가** (세션 기록만 존재).*

### 성능 측정 재현 방법

```bash
# 목표 개수를 2/4/5/8로 바꿔가며 run_mobo 직접 호출, time.perf_counter()로 계측
# 조건: 합성 데이터 30행, 공정변수 3개(Real 2 + Real 1), n_candidates=3,
#       torch.set_num_threads(1), CPU-only torch 2.13.0
```

---

## 한계 및 개선 여지 (정직한 기록)

1. **단일목표 경로 재현성 미확보** — `random_state=None`이라 같은 데이터로도 실행마다 추천이 달라진다. 다중목표는 시드 고정돼 있어 일관성이 없다.
2. **의존성 버전 미고정** — `requirements.txt`에 핀이 없어 상위 업데이트 시 동작이 바뀔 수 있다.
3. **결합 제약 미지원** — 박스 경계만 가능. "A+B ≤ 100" 같은 공정 제약을 표현할 수 없다.
4. **`CLAUDE.md` 부재** — 도메인 컨텍스트가 코드 주석에만 있어 세션마다 재설명 비용이 발생한다.
5. **ParEGO 전환 임계값(4)의 근거는 로컬 1회 측정** — 배포 환경(ninja 유무, CPU 쿼터)에 따라 최적 임계값은 다를 수 있다.
6. **효과 검증 데이터 부재** — 도입 전후 실험 횟수 비교군이 없어 실질 절감 효과는 확인 불가.
7. **Categorical 근사 처리** — 인덱스 연속화 후 반올림 방식이라 `optimize_acqf_mixed` 대비 이론적 정확도가 낮다 (코드 주석에 트레이드오프 명시됨).

---

*문서 생성일: 2026-08-03 · 저장소 HEAD: `21f4332` (2026-07-30) · 작업 트리 clean*
