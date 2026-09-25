# Bench CSV 컬럼정의서

> local-llm-lab 벤치마크 결과(`results/bench.csv`)의 단일 진실 공급원.
> 새 런타임(llama.cpp, MLX)을 추가할 때도 이 문서를 먼저 갱신한다.

- **schema_version**: 1.2.0
- **대상 스크립트**: `bench/hf_bench.py`, `bench/bench-memory.sh`

---

## 1. 통합 스키마

### 실행 식별

| 컬럼 | 타입 | 설명 |
|---|---|---|
| `run_id` | string | 이 행(프롬프트 1개)의 고유 식별자 (UUID) |
| `session_id` | string | 같은 프로세스 실행에서 나온 행들을 묶는 식별자 (프로세스당 1회 생성, 그 실행의 모든 행에 동일하게 채움) |
| `timestamp` | ISO8601 | 실행 시각 |
| `git_commit` | string | 실행 당시 커밋 SHA (short). git 레포가 아니면 `nogit` |
| `dirty_flag` | boolean | 커밋 이후 로컬 변경 여부 (`git diff --quiet`). **`git_commit`과 분리된 컬럼** — SHA는 항상 순수하게 유지 |
| `cond` | enum | 측정 조건 태그. `baseline` \| `no_sync` \| `no_warmup`. 1-3 검증 실험에서 같은 `bench.csv` 안의 행들을 조건별로 필터링하기 위함. **주의: 현재는 단일 값이라 `no_sync`+`no_warmup` 동시 적용 조건은 표현 불가 — 필요해지면 `no_sync_no_warmup`처럼 값을 추가하거나 `sync`/`warmup` 두 개의 별도 boolean 컬럼으로 다시 쪼개는 걸 고려** |

### 실행 환경

| 컬럼 | 타입 | 설명 |
|---|---|---|
| `runtime` | enum | `hf` \| `ollama` \| `llamacpp` \| `mlx` |
| `runtime_version` | string | 런타임 버전 (`transformers` 버전, `ollama_version` 등) |
| `model_id` | string | 모델 식별자 (예: `Qwen/Qwen2.5-0.5B-Instruct`) |
| `quant` | string \| null | 양자화 방식. **정책 미정 — 2절 참고** |
| `dtype` | string \| null | 부동소수점 정밀도. **정책 미정 — 2절 참고** |
| `device` | string | `mps` \| `cpu` \| `cuda` 등 |
| `n_ctx` | int | 컨텍스트 윈도우 크기 |

### 프롬프트 / 토큰

| 컬럼 | 타입 | 설명 |
|---|---|---|
| `prompt_id` | string | `prompts.json`의 `id` (예: `p16`, `p1024`) |
| `prompt_tokens` | int | 실측 프롬프트 토큰 수 (system prompt 포함) |
| `gen_tokens` | int | 생성된 토큰 수 (`min_new_tokens == max_new_tokens`로 고정된 값) |

### 성능 지표

| 컬럼 | 타입 | 설명 | 정의 |
|---|---|---|---|
| `ttft_ms` | float | Time To First Token | **런타임마다 구성이 다르다 — 아래 주의 필독.** 모델 로딩 시간(`load_ms`)은 양쪽 다 제외 |
| `decode_tok_s` | float | 디코딩 속도 | `gen_tokens / (디코딩 소요시간)`. **prompt 처리 시간 제외한 순수 디코딩 구간만** |
| `total_s` | float | 전체 소요 시간(초) | TTFT + 디코딩 전체 |

#### ⚠ `ttft_ms`는 런타임 간 비교에 쓰지 않는다 — 결정 완료

| 런타임 | 실제로 재는 것 |
|---|---|
| `hf` | `max_new_tokens=1` **별도 호출의 wall-clock** = prefill + 디코딩 1스텝 + 프레임워크 오버헤드 |
| `ollama` | `prompt_eval_duration` = **prefill만** |

디코딩 1스텝과 호출 오버헤드만큼 Ollama 값이 구조적으로 작다. 프롬프트가 짧을수록
(`p16`) 전체 TTFT 대비 이 격차의 비중이 크고, `p1024`로 갈수록 prefill이 지배해서 줄어든다.

**결정:** 각 런타임은 자기가 직접 보고하는 값을 그대로 쓴다. 합성값
(`prompt_eval_duration + eval_duration/eval_count` 등)으로 억지로 맞추지 않는다 —
첫 토큰의 디코딩 시간은 평균과 다르고, 만들어낸 값이라 근거가 약해진다.

대신 **분석 범위를 좁힌다**:

- ✅ 런타임 **내부**의 프롬프트 길이 스케일링 (`p16` → `p1024`에서 TTFT가 어떻게 자라는지)
- ✅ 같은 런타임 안에서의 조건 비교 (`cond`, `dtype`, `quant`별)
- ❌ 런타임 **간** TTFT 절대값·순위 비교

**강제 장치:** `scripts/plot_variance.py`의 `assert_metric_comparable()`이 `ttft_ms`를
그릴 때 데이터에 런타임이 2개 이상이면 `ValueError`로 중단한다. `--runtime hf`처럼
하나를 지정해야 한다. 문서 규칙만으로는 새는데, `--metric` 기본값이 `ttft_ms`이고
Ollama 행도 `cond=baseline`이라 가드가 없으면 기본 호출에서 조용히 섞인다.

`peak_rss_gb`도 런타임별 구성이 다르지만(HF는 `ru_maxrss` 누적 최대, Ollama는 샘플링
최댓값) 차단이 아니라 경고만 한다.

> 이 정의 차이는 약점이 아니라 측정 설계의 일부다. "TTFT를 런타임 간에 비교하지 않은
> 이유"는 그 자체로 설명 가치가 있으니 리포트에 남긴다.

### 메모리

| 컬럼 | 타입 | 설명 |
|---|---|---|
| `peak_rss_gb` | float | 프로세스 최고 RSS. **주의: "로드 전/후 스냅샷 차이"가 아니라 실제 관측 최고점**이어야 함 — 3절 참고 |
| `runtime_alloc_gb` | float | 런타임이 자체 보고하는 할당량 (MPS allocator 등) |

### 메타

| 컬럼 | 타입 | 설명 |
|---|---|---|
| `host` | string | **익명** 머신 라벨 (예: `apple-m2-pro-16gb`). 실제 hostname을 쓰지 않는다 — 5절 참고 |
| `os` | string | `uname -s` 결과 |
| `notes` | string | 이상 케이스, 수동 메모. 스키마 밖 값은 여기에 JSON으로 압축 가능 |

---

## 2. `quant` / `dtype` 정규화 및 필터링 정책 — 결정 완료

**원칙**: 한 행에 `dtype`과 `quant`가 동시에 채워지지 않는다. 손실 없는 원본 정밀도는 `dtype`에, 실제 양자화가 적용된 경우는 `quant`에 넣는다.

### 정규화 매핑

| 원본 (Ollama 태그) | `dtype` | `quant` |
|---|---|---|
| `f16` | `fp16` | `null` |
| `f32` (있는 경우) | `fp32` | `null` |
| `q8_0` | `null` | `q8_0` |
| `q4_0` | `null` | `q4_0` |
| `q4_K_M` 등 기타 양자화 태그 | `null` | 원본 태그 그대로 |

| HF 행 | `dtype` | `quant` |
|---|---|---|
| fp16 / fp32 실행 | 그대로 | `null` |

이 규칙으로 `WHERE dtype='fp16'`을 걸면 HF와 Ollama의 f16 행이 동시에 잡혀서, 런타임 무관하게 "같은 정밀도끼리" 비교가 가능해진다.

### 필터링 규칙 — 비교군 두 개로 분리

리포트/README에서 절대 하나의 순위표로 합치지 않는다. 원인(엔진 차이 vs 양자화 차이)이 섞이기 때문.

**① 정밀도 비교군** — 런타임 엔진 자체의 속도 차이를 보는 목적
```sql
WHERE quant IS NULL
GROUP BY dtype
```
runtime 무관하게 `dtype`으로만 그룹핑. **주의**: quant 변수만 통제된 것이지, HF(eager 실행)와 Ollama(자체 서빙 엔진) 사이의 구조적 차이는 여전히 안 걸러진다. 리포트에 이 한계를 명시할 것.

**② 양자화 트레이드오프군** — 양자화가 속도/메모리에 미치는 영향을 보는 목적
```sql
WHERE runtime='ollama' AND quant IS NOT NULL
```
같은 `model_id`의 `dtype='fp16', quant IS NULL` 행을 기준점(baseline)으로 대조.

**③ 섹션 분리**: 최종 리포트에서 "엔진 비교" 섹션과 "양자화 비교" 섹션을 물리적으로 나눠서 제시한다.

---

## 3. 런타임별 원본 → 통합 컬럼 매핑

### HF (`hf_bench.py`)

직접 구현하므로 통합 컬럼명을 그대로 출력하면 됨. 매핑표 불필요.

### Ollama (`bench-memory.sh`) — v2 전체 재감사

원본 30개 컬럼 전체를 다시 확인한 결과. `model_vram_mb`→`runtime_alloc_gb`, `processor`→`device`는
v1에서 "제외"로 잘못 분류했던 걸 바로잡은 것.

| 원본 컬럼 | 통합 컬럼 | 처리 |
|---|---|---|
| `timestamp` | `timestamp` | 그대로 |
| `host` | `host` | 그대로 |
| `ollama_version` | `runtime_version` | 이름만 변경 |
| `model` | `model_id` | 이름만 변경 |
| `gen_tokens` | `gen_tokens` | 그대로 |
| `prompt_tokens` | `prompt_tokens` | 그대로. **1-5에서 HF 실측치와 일치 검증 필수** |
| `total_ms` | `total_s` | `/ 1000` |
| `num_ctx` | `n_ctx` | 정수 필수. **HF도 빈 값이 아니다** — `args.n_ctx`(기본 2048)를 쓴다. 단 의미가 다름: HF는 토크나이저 truncation 상한(메모리 할당과 무관), Ollama는 KV 캐시 선할당 크기. 같은 값으로 맞춰 쓰되 메모리 컬럼 해석 시 이 비대칭을 함께 읽을 것 |
| `model_vram_mb` | `runtime_alloc_gb` | **v1에서 제외로 잘못 분류.** Ollama 자체 보고 GPU 할당량 — HF의 `driver_allocated_memory()`와 개념적으로 대응 |
| `processor` | `device` | **v1에서 제외로 잘못 분류.** GPU/CPU 실행 여부 — HF의 `device`(mps/cpu) 자리를 채워야 함 |
| `ollama_rss_baseline/loaded_mb` | `peak_rss_gb` | 로드 전/후 **스냅샷**이라 디코딩 중 실제 peak을 못 잡을 수 있음 (4절 하단 재확인 필요 항목 참고) |
| `sys_used_*_mb`, `sys_wired_*_mb`, `sys_anon_*_mb` | — | **제외.** 시스템 전체 메모리 통계, 프로세스 범위와 안 맞음 |
| `model_size_mb` | — | **제외.** 디스크상 모델 파일 크기, 런타임 메모리 사용량 아님 |
| `mem_total_mb` | — | **제외.** 머신 전체 물리 메모리(시스템 사양). `host`로 머신 식별 가능하므로 매 행 반복 저장 불필요 |
| `load_ms` | — | **제외.** 단, `ttft_ms` 검증 시 혼입 여부 확인용으로만 참고 |
| `keep_alive` | — | **제외.** `load_ms` 해석 시 참고용 |
| `run` | — | **제외.** 같은 조건 반복 실행 순번으로 추정되나, 반복 측정 편차 분석 계획 없음. `timestamp`로 순서 유추 가능하므로 버림 |
| (없음) | `run_id` | 신규 생성 (행마다 UUID) |
| (없음) | `git_commit`, `dirty_flag` | `bench-memory.sh`에 로직 신규 추가 |
| (없음) | `runtime` | 상수 `"ollama"` |
| (없음) | `prompt_id` | `prompts.json` 순회 시 `.id` 주입 |
| (없음) | `os` | `uname -s` 신규 추가 |

### ⚠ 검증 필요 (이름만 바꾸면 안 됨)

| 원본 컬럼 | 통합 컬럼 | 확인할 것 |
|---|---|---|
| `tokens_per_sec` | `decode_tok_s` | **해소.** `eval_count / eval_duration`으로 순수 디코딩 구간을 쓰고 있어 개념적으로 동일. 분자 비대칭(HF는 2~N번째, Ollama는 1~N번째 토큰 평균)은 N=128에서 1/N ≈ 0.8% 수준이라 그대로 두고 문서화만 — `ollama-adapter-spec.md` §3-E |
| `prompt_eval_ms` | `ttft_ms` | **해소(정의 차이 확정).** prefill-only이며 HF와 구성이 다르다. 런타임 간 비교를 하지 않는 것으로 결정 — 1절 `ttft_ms` 주의 항목 참고. `load_ms` 혼입 여부는 `p1024` 실측으로 별도 확인 (`ollama-adapter-spec.md` §4-2, §5) |
| `ollama_rss_baseline/loaded_mb` | `peak_rss_gb` | **해소 예정.** 스냅샷이라 실제 peak을 못 잡는 게 확인됨. 생성 구간 샘플링으로 바꾸기로 결정 — `ollama-adapter-spec.md` §3-F. HF의 `ru_maxrss`도 로드 시점에 정점을 찍어 사실상 "로드 후 정착 메모리"임이 실측으로 확인됨 |

---

## 4. 제외 컬럼

통합 스키마엔 자리가 없어서 CSV에는 넣지 않음. 원본 로그는 그대로 보존.

- `sys_used_*`, `sys_wired_*`, `sys_anon_*` — 시스템 전체 메모리 통계, 프로세스 범위와 안 맞음
- `model_size_mb` — 디스크상 모델 파일 크기, 런타임 메모리 사용량 아님
- `mem_total_mb` — 머신 전체 물리 메모리(시스템 사양), `host`로 대체 가능
- `load_ms`, `keep_alive` — TTFT/메모리 해석 시 참고용으로만, 컬럼으로는 안 남김
- `run` — 반복 실행 순번, 편차 분석 계획 없어 버림

**제외 사유**: 스키마의 목적이 "런타임 간 공통 비교"인데, 한쪽 런타임에만 있는 컬럼이 늘어나면 표가 지저분해지고 다른 런타임 행이 전부 빈 값이 됨.

`model_vram_mb`(→`runtime_alloc_gb`), `processor`(→`device`)는 v1에서 여기 있었으나 v2에서 매핑 대상으로 이동 — 3절 참고.

---

## 5. `host` 컬럼 규약

이 레포는 제출 시점에 public으로 전환된다. `host`에 실제 머신 이름이 들어가면
사용자 실명이 그대로 공개된다.

**규약:** 모든 런타임이 `bench/hostlabel.sh` **한 곳**에서 라벨을 얻는다.
해석 순서는 `BENCH_HOST` 환경변수 → `bench/hostlabel.sh` 실행 결과 → `unknown-host`.

- 라벨 형식: `<chip>-<ram>gb` (예: `apple-m2-pro-16gb`). 공백·쉼표는 `_`로 치환해 CSV 단일 토큰 유지
- 실패 시 **hostname으로 폴백하지 않는다.** 익명화가 조용히 풀리는 것보다 `unknown-host`가 낫다
- 라벨 생성 로직을 런타임별로 재구현하지 않는다. 값이 미묘하게 달라지면 런타임 간 조인이 깨진다
- `bench-memory.sh`는 라벨이 빈 문자열이면 **즉시 종료**한다 (과거에 경로 오류로
  `host`가 빈 값인 채 수집된 적이 있다)

> `hostlabel.sh`는 `scripts/`에서 `bench/`로 옮겼다. 호출자(`hf_bench.py`,
> `bench-memory.sh`)가 모두 `bench/`에 있어 경로 해석이 단순해지고, 각자
> 자기 파일 기준 상대 경로로 찾을 수 있다.

---

## Changelog

- `1.0.0` — 초기 통합 스키마 확정. `quant`/`dtype` 정규화·필터링 정책 결정 완료.
- `1.0.1` — Ollama 30개 원본 컬럼 전체 재감사. `model_vram_mb`→`runtime_alloc_gb`, `processor`→`device` 매핑 확정 (v1에서 제외로 오분류했던 것 정정). `run` 컬럼 제외 확정. `tokens_per_sec`/`prompt_eval_ms`/`ollama_rss_*` 세 항목은 여전히 검증 필요 상태로 남음.
- `1.2.0` — **컬럼 구성 변경 없음(24개 유지), 의미·규약만 확정.** ① `ttft_ms`의 런타임별
  정의 차이를 명시하고 런타임 간 비교를 금지 (1절 주의 항목). `plot_variance.py`에
  `assert_metric_comparable()` 가드로 강제. ② `host` 컬럼 규약 신설 (5절) —
  `hostlabel.sh`를 `scripts/`→`bench/`로 이동, `hf_bench.py`가 `socket.gethostname()`
  대신 이 스크립트를 호출하도록 변경. ③ 3절의 검증 필요 3항목에 결론 기재,
  `num_ctx → n_ctx`의 "HF는 빈 값" 오기 정정. **하위 호환 유지** — 기존
  `bench.csv`를 그대로 이어서 쓸 수 있다 (단 `host` 값은 변경 시점 전후로 달라진다)
- `1.1.0` — `session_id`, `cond` 컬럼 추가 (22→24개). 1-3 검증 실험에서 여러 프로세스 실행을 묶고(`session_id`) 측정 조건을 구분(`cond`)하기 위함 — 기존에는 `notes`에 문자열로 태깅하는 방식을 검토했으나 쿼리 편의성을 위해 정식 컬럼으로 분리. **하위 호환 깨짐**: 이 버전 이전에 수집된 `bench.csv`(22컬럼)는 새 헤더와 컬럼 수가 안 맞으므로 새 파일로 다시 시작함 (기존 데이터는 `docs/hf-notes.md`의 베이스라인 변동폭 섹션에 분석·보존됨).