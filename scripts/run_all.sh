#!/usr/bin/env bash
#
# run_all.sh - Day 1 하네스 전체를 한 번의 명령으로 순차 실행한다.
#
# ⚠ 병렬 실행하지 않는다. 이유는 둘 다 타협 불가다:
#   1. hf_bench.py와 bench-memory.sh가 같은 results/bench.csv에 append한다.
#      동시에 쓰면 행이 섞여 깨진다.
#   2. 벤치마크다. 두 프로세스가 같은 GPU·통합 메모리를 두고 경쟁하면 측정값이
#      전부 오염된다. 워밍업/프롬프트 캐시/keep_alive 규율이 단독 점유를 전제한다.
#
# 이 스크립트의 진짜 역할은 "한 번에 돌리기"가 아니라 **통제 변수 일치 보장**이다.
# --n-ctx / --gen-tokens를 한 곳에서 정해 HF와 Ollama 양쪽에 같은 값으로 넘긴다.
# 손으로 따로 돌리면 이 둘이 어긋나기 쉽고, 어긋나면 비교가 통째로 무효다
# (notes/runtime-comparability.md §3 통제 변수).
#
# Usage:
#   ./scripts/run_all.sh [options]
#
# Options:
#   -r, --repeats N      스윕 반복 횟수 (프로세스 재실행, default: 1)
#   -c, --n-ctx N        양쪽 공통 컨텍스트 크기 (default: 2048)
#   -g, --gen-tokens N   양쪽 공통 생성 토큰 수 (default: 128)
#       --notes TEXT     HF 행의 notes 태그 (';' 구분, 쉼표 금지)
#       --host URL       Ollama base URL (default: http://localhost:11434)
#       --skip-drift     모델 드리프트 검사 건너뜀
#       --skip-hf        HF 수집 건너뜀
#       --skip-ollama    Ollama 수집 건너뜀
#       --force          드리프트가 있어도 계속 진행
#   -n, --dry-run        실행할 명령만 출력
#   -h, --help           이 도움말
#
# Examples:
#   ./scripts/run_all.sh --dry-run
#   ./scripts/run_all.sh --repeats 5 --notes "quiet_env"
#   ./scripts/run_all.sh --skip-hf --repeats 3

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT" || exit 1

# ----------------------------------------------------------------------------
# 측정 대상 매트릭스 — 여기만 고치면 된다
# ----------------------------------------------------------------------------
# "<hf_repo>|<dtype>"
HF_MATRIX=(
  "Qwen/Qwen2.5-0.5B-Instruct|fp16"
  "Qwen/Qwen2.5-0.5B-Instruct|fp32"
  "Qwen/Qwen2.5-1.5B-Instruct|fp16"
  "Qwen/Qwen2.5-1.5B-Instruct|fp32"
)

# Ollama 태그. F32 태그는 보통 배포되지 않으므로 매트릭스가 비대칭인 게 정상이다.
OLLAMA_MATRIX=(
  "qwen2.5:0.5b-instruct-fp16"
  "qwen2.5:1.5b-instruct-fp16"
)

# "<model_key>|<variant>" — 측정 전에 동일성을 재검증할 쌍
IDENTITY_PAIRS=(
  "qwen2.5-0.5b-instruct|fp16"
)

# ----------------------------------------------------------------------------
# Defaults / args
# ----------------------------------------------------------------------------
REPEATS=1
N_CTX=2048
GEN_TOKENS=128
NOTES=""
HOST="http://localhost:11434"
SKIP_DRIFT=0
SKIP_HF=0
SKIP_OLLAMA=0
FORCE=0
DRY_RUN=0

usage() { sed -n '2,38p' "$0" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }

while [ $# -gt 0 ]; do
  case "$1" in
    -r|--repeats)     REPEATS="$2"; shift 2 ;;
    -c|--n-ctx)       N_CTX="$2"; shift 2 ;;
    -g|--gen-tokens)  GEN_TOKENS="$2"; shift 2 ;;
    --notes)          NOTES="$2"; shift 2 ;;
    --host)           HOST="${2%/}"; shift 2 ;;
    --skip-drift)     SKIP_DRIFT=1; shift ;;
    --skip-hf)        SKIP_HF=1; shift ;;
    --skip-ollama)    SKIP_OLLAMA=1; shift ;;
    --force)          FORCE=1; shift ;;
    -n|--dry-run)     DRY_RUN=1; shift ;;
    -h|--help)        usage 0 ;;
    *)                echo "Unknown option: $1" >&2; usage 1 ;;
  esac
done

case "$NOTES" in
  *,*) echo "ERROR: --notes 에 쉼표를 쓸 수 없다 (CSV가 깨진다). ';' 로 구분할 것." >&2; exit 1 ;;
esac

mkdir -p logs
STAMP="$(date +%Y%m%d-%H%M%S)"
LOG="logs/run_all-${STAMP}.log"

# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
FAILURES=0
say()   { printf '%s\n' "$*" | tee -a "$LOG"; }
phase() { say ""; say "=============================================================="; say "  $*"; say "=============================================================="; }
ok()    { say "  [OK]   $*"; }
warn()  { say "  [WARN] $*"; }
fail()  { say "  [FAIL] $*"; FAILURES=$((FAILURES+1)); }

run() {
  say ""
  say "  \$ $*"
  if [ "$DRY_RUN" -eq 1 ]; then return 0; fi
  if "$@" >>"$LOG" 2>&1; then
    ok "완료"
    return 0
  else
    fail "종료 코드 $? — 로그: $LOG"
    return 1
  fi
}

csv_rows() { [ -s results/bench.csv ] && echo $(( $(wc -l < results/bench.csv) - 1 )) || echo 0; }

say "=============================================================="
say "  run_all.sh  |  $(date '+%Y-%m-%d %H:%M:%S')"
say "  n_ctx=$N_CTX  gen_tokens=$GEN_TOKENS  repeats=$REPEATS"
say "  통제 변수는 양쪽에 같은 값으로 전달된다"
say "  로그: $LOG"
[ "$DRY_RUN" -eq 1 ] && say "  *** DRY RUN — 명령만 출력한다 ***"
say "=============================================================="

# ----------------------------------------------------------------------------
# Phase 0 — 선행 검사
# ----------------------------------------------------------------------------
phase "Phase 0 — 선행 검사"

if curl -sf "$HOST/api/version" >/dev/null 2>&1; then
  ok "Ollama 도달 ($(curl -s "$HOST/api/version" | jq -r '.version // "?"'))"
else
  if [ "$SKIP_OLLAMA" -eq 1 ]; then
    warn "Ollama에 닿지 않음 — --skip-ollama 이므로 계속"
  else
    fail "Ollama에 닿지 않음 ($HOST). 서버를 띄우거나 --skip-ollama."
    exit 1
  fi
fi

# CSV 헤더 상태. 비었거나 24컬럼과 일치해야 한다 (CC가 넣은 가드와 같은 판정).
EXPECTED_HEADER="$(uv run python -c 'import sys; sys.path.insert(0,"bench"); from schema import FIELDNAMES; print(",".join(FIELDNAMES))' 2>/dev/null)"
if [ -z "$EXPECTED_HEADER" ]; then
  fail "schema.py에서 FIELDNAMES를 읽지 못했다"
  exit 1
fi
if [ ! -s results/bench.csv ]; then
  ok "results/bench.csv 비어있음 — 스크립트가 헤더를 생성한다"
else
  ACTUAL_HEADER="$(head -n 1 results/bench.csv | tr -d '\r')"
  if [ "$ACTUAL_HEADER" = "$EXPECTED_HEADER" ]; then
    ok "results/bench.csv 헤더 일치 (기존 $(csv_rows)행)"
  else
    fail "results/bench.csv 헤더가 schema.py와 다르다. append하면 컬럼이 밀린다."
    say "       기대: $EXPECTED_HEADER"
    say "       실제: $ACTUAL_HEADER"
    exit 1
  fi
fi

# dirty_flag는 측정 행에 그대로 기록된다. 경고만.
if git --no-optional-locks diff --quiet 2>/dev/null; then
  ok "워킹트리 clean — dirty_flag=False로 기록된다"
else
  warn "워킹트리 dirty — 모든 행이 dirty_flag=True로 기록된다"
fi

ROWS_BEFORE="$(csv_rows)"

# ----------------------------------------------------------------------------
# Phase 1 — 모델 동일성 드리프트
# ----------------------------------------------------------------------------
if [ "$SKIP_DRIFT" -eq 1 ]; then
  phase "Phase 1 — 드리프트 검사 (건너뜀)"
else
  phase "Phase 1 — 모델 동일성 드리프트"
  say "  기록된 pin(HF revision / Ollama digest)이 현재와 같은지 확인한다."
  say "  드리프트가 있으면 지금 수집하는 데이터는 models.yaml이 주장하는 것과"
  say "  다른 가중치에 대한 측정이 된다."
  if run uv run python scripts/check_model_drift.py --host "$HOST"; then
    :
  else
    if [ "$FORCE" -eq 1 ]; then
      warn "드리프트 감지 — --force 이므로 계속 진행"
    else
      say ""
      say "  드리프트가 있다. 다음 중 하나를 하고 다시 실행할 것:"
      say "    - capture_model_identity.py 로 재검증 후 models.yaml 갱신"
      say "    - 의도한 변경이 아니라면 원래 가중치로 되돌리기"
      say "    - 그래도 수집하려면 --force"
      exit 1
    fi
  fi
fi

# ----------------------------------------------------------------------------
# Phase 2 — HF 수집
# ----------------------------------------------------------------------------
if [ "$SKIP_HF" -eq 1 ]; then
  phase "Phase 2 — HF 수집 (건너뜀)"
else
  phase "Phase 2 — HF 수집"
  say "  한 프로세스 = 한 스윕 (활성 프롬프트 전체 1회). 반복은 프로세스 재실행."
  for rep in $(seq 1 "$REPEATS"); do
    for entry in "${HF_MATRIX[@]}"; do
      MODEL="${entry%%|*}"
      DTYPE="${entry##*|}"
      say ""
      say "--- [rep $rep/$REPEATS] HF  $MODEL  $DTYPE ---"
      if [ -n "$NOTES" ]; then
        run uv run python bench/hf_bench.py \
          --model-id "$MODEL" --dtype "$DTYPE" \
          --n-ctx "$N_CTX" --gen-tokens "$GEN_TOKENS" --notes "$NOTES"
      else
        run uv run python bench/hf_bench.py \
          --model-id "$MODEL" --dtype "$DTYPE" \
          --n-ctx "$N_CTX" --gen-tokens "$GEN_TOKENS"
      fi
    done
  done
fi

# ----------------------------------------------------------------------------
# Phase 3 — Ollama 수집
# ----------------------------------------------------------------------------
if [ "$SKIP_OLLAMA" -eq 1 ]; then
  phase "Phase 3 — Ollama 수집 (건너뜀)"
else
  phase "Phase 3 — Ollama 수집"
  say "  --fresh 기본값이 시작 시 언로드하므로 프로세스마다 콜드 프롬프트 캐시."
  for rep in $(seq 1 "$REPEATS"); do
    for TAG in "${OLLAMA_MATRIX[@]}"; do
      say ""
      say "--- [rep $rep/$REPEATS] Ollama  $TAG ---"
      if curl -s "$HOST/api/tags" | jq -e --arg m "$TAG" \
           '[(.models // [])[] | select(.name == $m)] | length > 0' >/dev/null 2>&1; then
        run ./bench/bench-memory.sh "$TAG" \
          --num-ctx "$N_CTX" --gen-tokens "$GEN_TOKENS" --host "$HOST"
      else
        warn "미설치, 건너뜀:  ollama pull $TAG"
      fi
    done
  done
fi

# ----------------------------------------------------------------------------
# Phase 4 — 요약 · 통제 변수 검증
# ----------------------------------------------------------------------------
phase "Phase 4 — 요약"

if [ "$DRY_RUN" -eq 1 ]; then
  say "  (dry-run — 요약 생략)"
else
  ROWS_AFTER="$(csv_rows)"
  say "  results/bench.csv: ${ROWS_BEFORE}행 → ${ROWS_AFTER}행  (+$((ROWS_AFTER - ROWS_BEFORE)))"
  say ""
  uv run python - <<'PY' 2>&1 | tee -a "$LOG"
import csv, collections, pathlib
p = pathlib.Path("results/bench.csv")
rows = list(csv.DictReader(p.open())) if p.stat().st_size else []
if not rows:
    print("  (행 없음)"); raise SystemExit
def dist(k): return dict(collections.Counter(r[k] for r in rows))
print(f"  runtime : {dist('runtime')}")
print(f"  cond    : {dist('cond')}")
print(f"  dtype   : {dist('dtype')}")
print(f"  quant   : {dist('quant')}")
print(f"  device  : {dist('device')}")
print()
# 통제 변수 일치 — 어긋나면 런타임 간 비교가 무효다
for col in ("n_ctx", "gen_tokens"):
    vals = sorted({r[col] for r in rows})
    flag = "OK" if len(vals) == 1 else "!! 불일치 — 런타임 간 비교 불가"
    print(f"  {col:11}: {vals}  {flag}")
mixed = [r for r in rows if r["device"] == "mixed"]
if mixed:
    print(f"  !! device=mixed 행 {len(mixed)}개 — 엔진 비교에서 제외 대상")
bad = [r for r in rows if bool(r["dtype"]) == bool(r["quant"])]
if bad:
    print(f"  !! dtype/quant 상호배타 위반 {len(bad)}행")
hf = sum(1 for r in rows if r["runtime"] == "hf")
ol = sum(1 for r in rows if r["runtime"] == "ollama")
print()
print(f"  DoD (ROADMAP 1-5): hf {hf}행 (>=10 필요), ollama {ol}행 (>0 필요) "
      f"-> {'충족' if hf >= 10 and ol > 0 else '미충족'}")
PY
fi

say ""
if [ "$FAILURES" -eq 0 ]; then
  say "  실패 0건. 로그: $LOG"
else
  say "  실패 ${FAILURES}건 — 로그 확인: $LOG"
fi
say ""
exit "$FAILURES"
