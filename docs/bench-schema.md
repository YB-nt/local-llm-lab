# Bench CSV 컬럼정의서

> local-llm-lab 벤치마크 결과(`results/bench.csv`)의 단일 진실 공급원.
> 새 런타임(llama.cpp, MLX)을 추가할 때도 이 문서를 먼저 갱신한다.

- **schema_version**: 1.0.1
- **대상 스크립트**: `bench/hf_bench.py`, `bench/bench-memory.sh`

---

## 1. 통합 스키마

### 실행 식별

| 컬럼 | 타입 | 설명 |
|---|---|---|
| `run_id` | string | 이 실행의 고유 식별자 (UUID 또는 순번) |
| `timestamp` | ISO8601 | 실행 시각 |
| `git_commit` | string | 실행 당시 커밋 SHA (short). git 레포가 아니면 `nogit` |
| `dirty_flag` | boolean | 커밋 이후 로컬 변경 여부 (`git diff --quiet`). **`git_commit`과 분리된 컬럼** — SHA는 항상 순수하게 유지 |

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
| `ttft_ms` | float | Time To First Token | `max_new_tokens=1` 런으로 근사. **모델 로딩 시간(`load_ms`) 제외** |
| `decode_tok_s` | float | 디코딩 속도 | `gen_tokens / (디코딩 소요시간)`. **prompt 처리 시간 제외한 순수 디코딩 구간만** |
| `total_s` | float | 전체 소요 시간(초) | TTFT + 디코딩 전체 |

### 메모리

| 컬럼 | 타입 | 설명 |
|---|---|---|
| `peak_rss_gb` | float | 프로세스 최고 RSS. **주의: "로드 전/후 스냅샷 차이"가 아니라 실제 관측 최고점**이어야 함 — 3절 참고 |
| `runtime_alloc_gb` | float | 런타임이 자체 보고하는 할당량 (MPS allocator 등) |

### 메타

| 컬럼 | 타입 | 설명 |
|---|---|---|
| `host` | string | 실행 머신 이름 |
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
| `num_ctx` | `n_ctx` | 그대로 |
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
| `tokens_per_sec` | `decode_tok_s` | `eval_ms`(순수 디코딩 시간, Ollama API의 `eval_duration`)를 분모로 쓰는지 확인. 맞다면 HF `decode_tok_s`와 같은 정의라 이름만 바꾸면 됨. `total_ms` 기반이면 재계산 필요 |
| `prompt_eval_ms` | `ttft_ms` | `load_ms`(모델 로딩)가 섞여 들어가는지. `keep_alive` 설정에 따라 매 요청마다 로드가 발생하면 TTFT가 부풀려짐 |
| `ollama_rss_baseline/loaded_mb` | `peak_rss_gb` | 로드 전/후 **스냅샷**이라 디코딩 중 실제 peak(특히 `p1024`처럼 KV 캐시가 커지는 케이스)를 못 잡을 수 있음. 이름 그대로 쓸지, "post-load RSS"로 정의를 재확인할지 결정 필요 |

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

## Changelog

- `1.0.0` — 초기 통합 스키마 확정. `quant`/`dtype` 정규화·필터링 정책 결정 완료.
- `1.0.1` — Ollama 30개 원본 컬럼 전체 재감사. `model_vram_mb`→`runtime_alloc_gb`, `processor`→`device` 매핑 확정 (v1에서 제외로 오분류했던 것 정정). `run` 컬럼 제외 확정. `tokens_per_sec`/`prompt_eval_ms`/`ollama_rss_*` 세 항목은 여전히 검증 필요 상태로 남음.