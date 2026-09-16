#!/usr/bin/env bash
#
# bench-memory.sh - Ollama 메모리/처리량 벤치마크 하네스.
#
# macOS-focused (uses `top` / `ps` / `vm_stat`), talks to the Ollama HTTP API.
#
# 두 종류의 CSV를 남긴다 (docs/ollama-adapter-spec.md):
#   1. benchmarks/bench-results.csv  - 이 스크립트 고유의 30컬럼 상세 로그 (그대로 유지)
#   2. results/bench.csv             - HF 행과 같은 파일에 섞이는 통합 24컬럼 스키마
#      (컬럼 정의: bench/schema.py FIELDNAMES). 대체가 아니라 추가 출력이다.
#
# 한 프로세스 실행 = 프롬프트 스윕 1회 (bench/prompts.json의 active 프롬프트 전체를
# 순서대로 1번씩). 반복 측정은 프로세스를 다시 실행해서 한다 - 같은 프로세스 안에서
# 같은 프롬프트를 두 번 요청하면 Ollama가 프롬프트 캐시를 재사용해 두 번째 prefill
# 값이 무효가 된다 (spec §3-G). 그래서 `-n/--runs` 같은 내부 반복 옵션은 없다.
#
# Usage:
#   ./bench-memory.sh MODEL [options]
#
# Options:
#   -c, --num-ctx N          Context window size, options.num_ctx (default: 2048)
#   -g, --gen-tokens N       Forced generation length, options.num_predict (default: 128)
#   -k, --keep-alive DUR     Ollama keep_alive for the model (default: 5m)
#       --cond NAME          cond 컬럼 오버라이드 (default: baseline)
#       --host URL           Ollama base URL (default: http://localhost:11434)
#       --prompts-path PATH  bench/prompts.json 경로 (default: <repo>/bench/prompts.json)
#       --csv PATH           원본 30컬럼 CSV (default: <repo>/benchmarks/bench-results.csv)
#       --unified-csv PATH   통합 24컬럼 CSV (default: <repo>/results/bench.csv)
#       --log PATH           사람이 읽는 로그 (default: <repo>/benchmarks/bench-memory.log)
#       --no-fresh           Do NOT unload the model first; measure current state
#       --no-unload          Leave the model loaded when finished
#       --settle SECONDS     Wait after (un)load before measuring (default: 3)
#   -h, --help               Show this help
#
# Examples:
#   ./bench-memory.sh qwen2.5:0.5b-instruct-fp16
#   ./bench-memory.sh qwen2.5:0.5b-instruct-fp16 -c 8192 -k 10m
#
set -uo pipefail

# ----------------------------------------------------------------------------
# Defaults
# ----------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

HOST="http://localhost:11434"
KEEP_ALIVE="5m"
NUM_CTX="2048"
GEN_TOKENS="128"
COND="baseline"
PROMPTS_JSON="$REPO_ROOT/bench/prompts.json"
BENCH_DIR="$REPO_ROOT/benchmarks"
CSV="$BENCH_DIR/bench-results.csv"; LOG="$BENCH_DIR/bench-memory.log"
UNIFIED_CSV="$REPO_ROOT/results/bench.csv"
SETTLE=3
FRESH=1
UNLOAD=1

# 벤치마크 세트(bench/prompts.json) 밖의 프롬프트여야 한다 (spec §3-C).
# system 필드 없이, num_predict=4로 짧게 태워서 디코딩 커널만 데운다.
WARMUP_PROMPT="This warmup request exists only to prime decode kernels before the benchmark sweep and is intentionally excluded from bench/prompts.json."
WARMUP_TOKENS=4

# bench/schema.py의 FIELDNAMES(단일 진실 공급원)와 이 스크립트가 행을 조립하는
# 순서가 같은지 런타임에 검증한다 (§1, 체크리스트: "값이 한 칸 밀리는 실수 확인").
EXPECTED_FIELD_ORDER="run_id,session_id,timestamp,git_commit,dirty_flag,cond,runtime,runtime_version,model_id,quant,dtype,device,n_ctx,prompt_id,prompt_tokens,gen_tokens,ttft_ms,decode_tok_s,total_s,peak_rss_gb,runtime_alloc_gb,host,os,notes"

# ----------------------------------------------------------------------------
# Args
# ----------------------------------------------------------------------------
usage() {
  awk 'NR>1 && /^#/{sub(/^# ?/,""); print; next} NR>1{exit}' "$0"
  exit "${1:-0}"
}

[ $# -eq 0 ] && usage 1
MODEL=""
while [ $# -gt 0 ]; do
  case "$1" in
    -c|--num-ctx)     NUM_CTX="$2"; shift 2 ;;
    -g|--gen-tokens)  GEN_TOKENS="$2"; shift 2 ;;
    -k|--keep-alive)  KEEP_ALIVE="$2"; shift 2 ;;
    --cond)           COND="$2"; shift 2 ;;
    --host)           HOST="${2%/}"; shift 2 ;;
    --prompts-path)   PROMPTS_JSON="$2"; shift 2 ;;
    --csv)            CSV="$2"; shift 2 ;;
    --unified-csv)    UNIFIED_CSV="$2"; shift 2 ;;
    --log)            LOG="$2"; shift 2 ;;
    --no-fresh)       FRESH=0; shift ;;
    --no-unload)      UNLOAD=0; shift ;;
    --settle)         SETTLE="$2"; shift 2 ;;
    -h|--help)        usage 0 ;;
    -*)               echo "Unknown option: $1" >&2; usage 1 ;;
    *)                if [ -z "$MODEL" ]; then MODEL="$1"; else echo "Unexpected arg: $1" >&2; usage 1; fi; shift ;;
  esac
done
[ -z "$MODEL" ] && { echo "ERROR: MODEL is required" >&2; usage 1; }

command -v jq   >/dev/null || { echo "ERROR: jq is required" >&2; exit 1; }
command -v curl >/dev/null || { echo "ERROR: curl is required" >&2; exit 1; }
command -v uv   >/dev/null || { echo "ERROR: uv is required (schema.py lookup)" >&2; exit 1; }
[ -f "$PROMPTS_JSON" ] || { echo "ERROR: not found: $PROMPTS_JSON" >&2; exit 1; }

# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
# 원본 30컬럼 CSV용 타임스탬프 - 기존 포맷 그대로 유지 (콜론 없는 오프셋).
ts_legacy() { date +"%Y-%m-%dT%H:%M:%S%z"; }
# 통합 24컬럼 CSV용 타임스탬프 - HF의 datetime.isoformat()과 맞추기 위해 오프셋에
# 콜론을 넣는다. macOS(BSD) date는 %:z를 지원하지 않으므로 후처리한다 (spec §4-1).
ts_iso() { date +"%Y-%m-%dT%H:%M:%S%z" | sed -E 's/([0-9]{2})([0-9]{2})$/\1:\2/'; }
say() { printf '%s\n' "$*" | tee -a "$LOG"; }

# System-wide memory, parsed from `vm_stat` (page-granular, instant).
# Prints: "<used_mb> <free_mb> <wired_mb> <compressed_mb> <anon_mb>"
sys_mem() {
  vm_stat 2>/dev/null | awk '
    /page size of/ { match($0,/[0-9]+/); ps=substr($0,RSTART,RLENGTH) }
    /Pages free:/                    { free=$3+0 }
    /Pages speculative:/             { spec=$3+0 }
    /Pages inactive:/                { inact=$3+0 }
    /Pages purgeable:/               { purg=$3+0 }
    /Pages wired down:/              { wired=$4+0 }
    /Pages occupied by compressor:/  { comp=$5+0 }
    /Anonymous pages:/               { anon=$3+0 }
    END {
      mb = ps / 1048576
      avail = free + spec + inact + purg
      hw = "'"$MEM_TOTAL_MB"'" + 0
      used = hw - (avail*mb)
      printf "%.0f %.0f %.0f %.0f %.0f", used, avail*mb, wired*mb, comp*mb, anon*mb
    }'
}

# Total RSS (MB) of every process whose command line contains "ollama".
ollama_rss_mb() {
  local pids csv kb
  pids=$(pgrep -f 'ollama' 2>/dev/null)
  [ -z "$pids" ] && { echo 0; return; }
  csv=$(printf '%s' "$pids" | tr '\n' ',' | sed 's/,$//')
  kb=$(ps -o rss= -p "$csv" 2>/dev/null | awk '{s+=$1} END{printf "%.0f", s+0}')
  awk -v k="$kb" 'BEGIN{printf "%.0f", k/1024}'
}

# Ollama's own accounting for the loaded model via /api/ps (MB-rounded; legacy CSV only).
# Prints TAB-separated: "<total_mb>\t<vram_mb>\t<processor>\t<ctx>" (or "0\t0\t-\t-").
ollama_ps_model() {
  curl -s "$HOST/api/ps" | jq -r --arg m "$MODEL" '
    ((.models // []) | map(select(.name == $m or .model == $m)) | .[0]) as $x
    | if $x == null then [0, 0, "-", "-"]
      else
        (($x.size // 0)) as $sb
        | (($x.size_vram // 0)) as $vb
        | [ ($sb/1048576 | floor),
            ($vb/1048576 | floor),
            (if $sb == 0 then "-"
             elif $vb >= $sb then "100% GPU"
             elif $vb == 0 then "100% CPU"
             else (($vb*100/$sb) | floor | tostring) + "% GPU / " + ((100 - ($vb*100/$sb)) | floor | tostring) + "% CPU" end),
            ($x.context_length // ($x.details.context_length) // "-") ]
      end | @tsv'
}

model_loaded() {
  curl -s "$HOST/api/ps" | jq -e --arg m "$MODEL" \
    '[(.models // [])[] | select(.name==$m or .model==$m)] | length > 0' >/dev/null
}

# Body builders for /api/generate.
load_body() {  # $1 = keep_alive
  jq -nc --arg m "$MODEL" --arg k "$1" --argjson ctx "$NUM_CTX" \
    '{model:$m, prompt:"", stream:false, keep_alive:$k, options:{num_ctx:$ctx}}'
}
unload_body() {
  jq -nc --arg m "$MODEL" '{model:$m, prompt:"", keep_alive:"0"}'
}
warmup_body() {
  jq -nc --arg m "$MODEL" --arg p "$WARMUP_PROMPT" --arg k "$KEEP_ALIVE" \
    --argjson ctx "$NUM_CTX" --argjson np "$WARMUP_TOKENS" \
    '{model:$m, prompt:$p, stream:false, keep_alive:$k, options:{num_ctx:$ctx, num_predict:$np}}'
}
# system 필드로 bench/prompts.json의 _meta.system_prompt.text를 그대로 전달한다.
# 빠뜨리면 "system prompt 없음"이 아니라 Ollama가 모델 내장 기본값을 주입해
# prompt_tokens가 조용히 달라진다 (spec §3-B, 실측으로 정정됨).
sweep_body() {  # $1 = prompt text
  jq -nc --arg m "$MODEL" --arg p "$1" --arg s "$SYS_PROMPT" --arg k "$KEEP_ALIVE" \
    --argjson ctx "$NUM_CTX" --argjson np "$GEN_TOKENS" '
    {model:$m, prompt:$p, system:$s, stream:false, keep_alive:$k,
     options:{num_ctx:$ctx, num_predict:$np}}'
}

unload_model() {
  curl -s "$HOST/api/generate" -d "$(unload_body)" >/dev/null
  for _ in $(seq 1 20); do model_loaded || break; sleep 0.5; done
}

# 생성 요청 1건을 실행하면서 그 구간 동안 ollama 프로세스 합산 RSS를 200ms 간격으로
# 샘플링해 최댓값을 남긴다 (spec §3-F). 로드 직후 스냅샷 1회가 아니라 생성 구간
# 중 실제 최고점을 재는 것이 목적이므로, 스냅샷 복사가 아니라 매 프롬프트마다
# 새로 샘플링한다 - 프롬프트별로 값이 달라야 정상이다.
# 결과는 전역 변수 GEN_RESP / PEAK_RSS_MB에 남긴다.
run_measured_generate() {  # $1 = prompt text
  local body rss_tmp sampler_pid
  body=$(sweep_body "$1")
  rss_tmp=$(mktemp)
  echo 0 > "$rss_tmp"
  ( while :; do
      cur=$(ollama_rss_mb)
      prev=$(cat "$rss_tmp" 2>/dev/null)
      if [[ "$cur" =~ ^[0-9]+$ ]] && [[ "$prev" =~ ^[0-9]+$ ]] && [ "$cur" -gt "$prev" ]; then
        echo "$cur" > "$rss_tmp"
      fi
      sleep 0.2
    done ) &
  sampler_pid=$!
  GEN_RESP=$(curl -s "$HOST/api/generate" -d "$body")
  kill "$sampler_pid" 2>/dev/null
  wait "$sampler_pid" 2>/dev/null
  PEAK_RSS_MB=$(cat "$rss_tmp" 2>/dev/null)
  [[ "$PEAK_RSS_MB" =~ ^[0-9]+$ ]] || PEAK_RSS_MB=0
  rm -f "$rss_tmp"
}

# ----------------------------------------------------------------------------
# Preflight
# ----------------------------------------------------------------------------
curl -sf "$HOST/api/version" >/dev/null || { echo "ERROR: Ollama not reachable at $HOST" >&2; exit 1; }
OLLAMA_VER=$(curl -s "$HOST/api/version" | jq -r '.version // "?"')

# host 라벨은 bench/hostlabel.sh 한 곳에서만 생성한다 - hf_bench.py도 같은 스크립트를
# 호출하므로 두 런타임의 host 값이 문자열까지 일치하고 런타임 간 조인이 가능하다.
# 실제 hostname은 쓰지 않는다 (사용자 실명이 들어갈 수 있고 이 레포는 public으로 전환된다).
HOSTLABEL_SH="$SCRIPT_DIR/hostlabel.sh"
HOSTNAME_S="${BENCH_HOST:-$("$HOSTLABEL_SH" 2>/dev/null)}"
HOSTNAME_S=$(printf '%s' "$HOSTNAME_S" | sed 's/[[:space:],]\{1,\}/_/g')   # keep CSV single-token
# 빈 값을 조용히 넘기지 않는다 - 스크립트 경로가 틀려 host가 빈 문자열로 쌓인 적이 있다.
[ -z "$HOSTNAME_S" ] && {
  echo "ERROR: host label is empty ($HOSTLABEL_SH 실행 실패)." >&2
  echo "       BENCH_HOST 환경변수로 지정하거나 hostlabel.sh 경로를 확인하세요." >&2
  exit 1
}
MEM_TOTAL_MB=$(( $(sysctl -n hw.memsize) / 1048576 ))
OS_NAME=$(uname -s)

# bench/schema.py의 FIELDNAMES가 단일 진실 공급원이다 - 컬럼명을 하드코딩하지 않는다.
FIELDS=$(cd "$REPO_ROOT" && uv run python -c \
  'import sys; sys.path.insert(0,"bench"); from schema import FIELDNAMES; print(",".join(FIELDNAMES))')
[ -z "$FIELDS" ] && { echo "ERROR: failed to read FIELDNAMES from bench/schema.py" >&2; exit 1; }
if [ "$FIELDS" != "$EXPECTED_FIELD_ORDER" ]; then
  echo "ERROR: bench/schema.py의 FIELDNAMES 순서가 이 스크립트의 가정과 다릅니다." >&2
  echo "  schema.py : $FIELDS" >&2
  echo "  script    : $EXPECTED_FIELD_ORDER" >&2
  echo "  스크립트의 행 조립 순서를 schema.py에 맞춰 함께 고쳐야 합니다." >&2
  exit 1
fi

# bench/prompts.json: active 프롬프트만 사용, 0개면 중단 (HF와 동일).
SYS_PROMPT=$(jq -r '._meta.system_prompt.text' "$PROMPTS_JSON")
[ -z "$SYS_PROMPT" ] && { echo "ERROR: ${PROMPTS_JSON}: _meta.system_prompt.text is empty" >&2; exit 1; }
ACTIVE_IDS=$(jq -r '.prompts[] | select(.active==true) | .id' "$PROMPTS_JSON")
ACTIVE_COUNT=$(printf '%s\n' "$ACTIVE_IDS" | sed '/^$/d' | wc -l | tr -d ' ')
[ "$ACTIVE_COUNT" -eq 0 ] && { echo "ERROR: ${PROMPTS_JSON}: no active prompts found" >&2; exit 1; }
UNIQUE_COUNT=$(printf '%s\n' "$ACTIVE_IDS" | sort -u | wc -l | tr -d ' ')
[ "$UNIQUE_COUNT" -ne "$ACTIVE_COUNT" ] && { echo "ERROR: ${PROMPTS_JSON}: duplicate active prompt ids" >&2; exit 1; }

mkdir -p "$(dirname "$CSV")" "$(dirname "$LOG")" "$(dirname "$UNIFIED_CSV")"
if [ ! -f "$CSV" ]; then
  echo "timestamp,host,ollama_version,model,run,gen_tokens,prompt_tokens,tokens_per_sec,load_ms,prompt_eval_ms,eval_ms,total_ms,mem_total_mb,sys_used_baseline_mb,sys_used_loaded_mb,sys_used_delta_mb,sys_wired_baseline_mb,sys_wired_loaded_mb,sys_wired_delta_mb,sys_anon_baseline_mb,sys_anon_loaded_mb,sys_anon_delta_mb,ollama_rss_baseline_mb,ollama_rss_loaded_mb,ollama_rss_delta_mb,model_size_mb,model_vram_mb,processor,num_ctx,keep_alive" > "$CSV"
fi
# 기존 파일이 있으면 첫 줄을 FIELDS와 문자열까지 비교한다. 다른 스키마의 헤더
# (예: 이 파일 자체의 30컬럼 레거시 형식이 실수로 통합 CSV 경로에 쓰인 경우) 위에
# 이 스키마의 행을 그냥 append하면 컬럼이 밀려 pandas가 에러 없이 조용히 잘못된
# 값을 읽는다 (notes/integration-test-2026-09-16.md에서 실측 재현됨) - 그 조용한
# 손상을 막기 위해 먼저 큰 소리로 막는다. tr -d '\r'로 CRLF/LF 어느 쪽으로 쓰인
# 헤더든 허용한다 - hf_bench.py의 csv.DictWriter는 기본 CRLF로 쓴다.
if [ -s "$UNIFIED_CSV" ]; then
  EXISTING_HEADER=$(head -n 1 "$UNIFIED_CSV" | tr -d '\r')
  if [ "$EXISTING_HEADER" != "$FIELDS" ]; then
    echo "ERROR: ${UNIFIED_CSV}의 기존 헤더가 FIELDNAMES와 다릅니다." >&2
    echo "  기존: $EXISTING_HEADER" >&2
    echo "  기대: $FIELDS" >&2
    echo "  이 파일에 다른 스키마의 헤더가 남아있는 것으로 보입니다 - 이 위에 새" >&2
    echo "  행을 이어 쓰면 컬럼이 밀려 조용히 손상됩니다. 파일을 확인해 헤더를" >&2
    echo "  바로잡거나, --unified-csv로 새 경로를 지정하세요." >&2
    exit 1
  fi
else
  printf '%s\n' "$FIELDS" > "$UNIFIED_CSV"
fi

# 실행 전체에서 한 번만 정해지는 값들 (session-level).
SESSION_ID=$(uuidgen | tr 'A-Z' 'a-z')
GIT_COMMIT=$(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null)
if [ -z "$GIT_COMMIT" ]; then
  GIT_COMMIT="nogit"
  DIRTY_FLAG="False"
else
  if git -C "$REPO_ROOT" diff --quiet 2>/dev/null; then DIRTY_FLAG="False"; else DIRTY_FLAG="True"; fi
fi

say "======================================================================"
say "  Ollama memory benchmark  |  $(ts_legacy)"
say "  host=$HOSTNAME_S  ollama=$OLLAMA_VER  ram=${MEM_TOTAL_MB}MB"
say "  model=$MODEL  session=$SESSION_ID  num_ctx=$NUM_CTX  gen_tokens=$GEN_TOKENS  keep_alive=$KEEP_ALIVE  cond=$COND"
say "======================================================================"

# ----------------------------------------------------------------------------
# 1. Baseline (before load)
# ----------------------------------------------------------------------------
if [ "$FRESH" -eq 1 ]; then
  if model_loaded; then say "[fresh] model already resident -> unloading first"; unload_model; fi
else
  model_loaded && say "[warn] --no-fresh and model is ALREADY loaded; 'before' numbers include it"
fi
sleep "$SETTLE"

read -r SYS_USED_BASE SYS_FREE_BASE SYS_WIRED_BASE SYS_COMP_BASE SYS_ANON_BASE <<<"$(sys_mem)"
RSS_BASE=$(ollama_rss_mb)
say ""
say "-- BEFORE LOAD ------------------------------------------------"
say "   system memory used : ${SYS_USED_BASE} MB   (available ${SYS_FREE_BASE} MB)"
say "   wired / compressed  : ${SYS_WIRED_BASE} MB / ${SYS_COMP_BASE} MB   anon ${SYS_ANON_BASE} MB"
say "   ollama processes RSS: ${RSS_BASE} MB"

# ----------------------------------------------------------------------------
# 2. Load model
# ----------------------------------------------------------------------------
say ""
say "-- LOADING MODEL -------------------------------------------------------"
LOAD_RESP=$(curl -s "$HOST/api/generate" -d "$(load_body "$KEEP_ALIVE")")
if ! printf '%s' "$LOAD_RESP" | jq -e '.done' >/dev/null 2>&1; then
  say "ERROR: load failed: $(printf '%s' "$LOAD_RESP" | jq -r '.error // .' 2>/dev/null || printf '%s' "$LOAD_RESP")"
  exit 1
fi
LOAD_MS=$(printf '%s' "$LOAD_RESP" | jq -r '((.load_duration // 0)/1e6) | floor')
sleep "$SETTLE"

read -r SYS_USED_LOAD SYS_FREE_LOAD SYS_WIRED_LOAD SYS_COMP_LOAD SYS_ANON_LOAD <<<"$(sys_mem)"
RSS_LOAD=$(ollama_rss_mb)
IFS=$'\t' read -r M_SIZE_MB M_VRAM_MB M_PROC M_CTX <<<"$(ollama_ps_model)"

SYS_DELTA=$(( SYS_USED_LOAD - SYS_USED_BASE ))
WIRED_DELTA=$(( SYS_WIRED_LOAD - SYS_WIRED_BASE ))
ANON_DELTA=$(( SYS_ANON_LOAD - SYS_ANON_BASE ))
RSS_DELTA=$(( RSS_LOAD - RSS_BASE ))

say ""
say "-- AFTER LOAD -------------------------------------------------"
say "   system memory used : ${SYS_USED_LOAD} MB   (delta ${SYS_DELTA} MB, available ${SYS_FREE_LOAD} MB)"
say "   wired / compressed  : ${SYS_WIRED_LOAD} MB (delta ${WIRED_DELTA} MB) / ${SYS_COMP_LOAD} MB   anon ${SYS_ANON_LOAD} MB (delta ${ANON_DELTA} MB)"
say "   >> best load estimate: wired+${WIRED_DELTA}MB | ollama RSS+${RSS_DELTA}MB | api/ps ${M_SIZE_MB}MB"
say "   ollama processes RSS: ${RSS_LOAD} MB   (delta ${RSS_DELTA} MB)"
say "   ollama /api/ps      : size=${M_SIZE_MB} MB  vram=${M_VRAM_MB} MB  processor=${M_PROC}  ctx=${M_CTX}"
say "   model load_duration : ${LOAD_MS} ms"

# ----------------------------------------------------------------------------
# 2b. Unified-schema session-level facts (spec §2-1, §2-2, §2-3)
# ----------------------------------------------------------------------------
# /api/show: quantization_level -> quant/dtype 배타적 정규화. 태그 문자열은 파싱하지
# 않는다 - 기본 태그(qwen2.5:0.5b)엔 양자화 정보가 이름에 없다.
QL=$(curl -s "$HOST/api/show" -d "$(jq -nc --arg m "$MODEL" '{model:$m}')" \
     | jq -r '.details.quantization_level // ""')
[ -z "$QL" ] && { say "ERROR: /api/show did not return .details.quantization_level"; exit 1; }
case "$QL" in
  F16) DTYPE="fp16"; QUANT="" ;;
  F32) DTYPE="fp32"; QUANT="" ;;
  Q8_0) DTYPE=""; QUANT="q8_0" ;;
  Q4_0) DTYPE=""; QUANT="q4_0" ;;
  *) DTYPE=""; QUANT=$(printf '%s' "$QL" | sed -E 's/^Q/q/') ;;
esac
if { [ -n "$DTYPE" ] && [ -n "$QUANT" ]; } || { [ -z "$DTYPE" ] && [ -z "$QUANT" ]; }; then
  say "ERROR: quant/dtype normalization produced an invalid state for quantization_level='$QL'"
  exit 1
fi

# /api/ps (바이트 정밀도): device 분류 + 실측 n_ctx + runtime_alloc_gb.
# device="mixed"는 엔진 비교에서 제외 대상이다 (spec §2-2) - proc= 태그로 원본을 남긴다.
read -r RUNTIME_ALLOC_GB DEVICE N_CTX_RAW <<EOF
$(curl -s "$HOST/api/ps" | jq -r --arg m "$MODEL" '
  ((.models // []) | map(select(.name == $m or .model == $m)) | .[0]) as $x
  | if $x == null then "0\tunknown\t-"
    else
      (($x.size // 0)) as $sb | (($x.size_vram // 0)) as $vb |
      [ ($vb/1073741824),
        (if $sb == 0 then "unknown"
         elif $vb >= $sb then "mps"
         elif $vb == 0 then "cpu"
         else "mixed" end),
        ($x.context_length // ($x.details.context_length) // "-")
      ] | @tsv
    end')
EOF
[ "$DEVICE" = "unknown" ] && { say "ERROR: model '$MODEL' not found in /api/ps after load"; exit 1; }

N_CTX_SRC_REQUESTED=0
N_CTX_MISMATCH=0
if [ "$N_CTX_RAW" = "-" ] || [ -z "$N_CTX_RAW" ]; then
  N_CTX="$NUM_CTX"
  N_CTX_SRC_REQUESTED=1
else
  N_CTX="$N_CTX_RAW"
  [ "$N_CTX" != "$NUM_CTX" ] && N_CTX_MISMATCH=1
fi

# 세션 전체에 공통인 notes 토큰 (매 행에 동일하게 붙는다).
SESSION_NOTES=""
[ "$DEVICE" = "mixed" ] && SESSION_NOTES="proc=${M_PROC// /_}"
if [ "$N_CTX_SRC_REQUESTED" -eq 1 ]; then
  SESSION_NOTES="${SESSION_NOTES:+$SESSION_NOTES;}n_ctx_src=requested"
elif [ "$N_CTX_MISMATCH" -eq 1 ]; then
  SESSION_NOTES="${SESSION_NOTES:+$SESSION_NOTES;}n_ctx_req=$NUM_CTX"
fi

say ""
say "-- UNIFIED SCHEMA SESSION FACTS ---------------------------------"
say "   quantization_level=$QL -> dtype='$DTYPE' quant='$QUANT'"
say "   device=$DEVICE  n_ctx(measured)=$N_CTX  runtime_alloc_gb=$(awk -v v="$RUNTIME_ALLOC_GB" 'BEGIN{printf "%.4f", v}')"

# ----------------------------------------------------------------------------
# 3. Warmup (버림, CSV에 안 남김) - spec §3-C
# ----------------------------------------------------------------------------
say ""
say "-- WARMUP (discarded) --------------------------------------------------"
WARMUP_RESP=$(curl -s "$HOST/api/generate" -d "$(warmup_body)")
if ! printf '%s' "$WARMUP_RESP" | jq -e '.done' >/dev/null 2>&1; then
  say "ERROR: warmup failed: $(printf '%s' "$WARMUP_RESP" | jq -r '.error // .' 2>/dev/null || printf '%s' "$WARMUP_RESP")"
  exit 1
fi
say "   warmup ok"

# ----------------------------------------------------------------------------
# 4. Prompt sweep (active prompts, 1회씩) - spec §1, §3-A, §3-B, §3-G
# ----------------------------------------------------------------------------
say ""
say "-- PROMPT SWEEP (n=$ACTIVE_COUNT) --------------------------------------"
SUM_TPS=0; OK_RUNS=0; I=0
for PID in $ACTIVE_IDS; do
  I=$((I+1))
  PTEXT=$(jq -r --arg id "$PID" '.prompts[] | select(.id==$id) | .text' "$PROMPTS_JSON")

  run_measured_generate "$PTEXT"
  if ! printf '%s' "$GEN_RESP" | jq -e '.done' >/dev/null 2>&1; then
    say "   $PID: ERROR $(printf '%s' "$GEN_RESP" | jq -r '.error // .' 2>/dev/null)"
    continue
  fi

  read -r GTOK PTOK LOAD_DUR PROMPT_EVAL_DUR EVAL_DUR TOTAL_DUR <<EOF
$(printf '%s' "$GEN_RESP" | jq -r '
  [ (.eval_count // 0), (.prompt_eval_count // 0), (.load_duration // 0),
    (.prompt_eval_duration // 0), (.eval_duration // 0), (.total_duration // 0)
  ] | @tsv')
EOF

  # -- 원본 30컬럼 CSV용 (ms 단위, floor) --------------------------------
  LMS=$(awk -v v="$LOAD_DUR" 'BEGIN{printf "%d", v/1e6}')
  PMS=$(awk -v v="$PROMPT_EVAL_DUR" 'BEGIN{printf "%d", v/1e6}')
  EMS=$(awk -v v="$EVAL_DUR" 'BEGIN{printf "%d", v/1e6}')
  TMS=$(awk -v v="$TOTAL_DUR" 'BEGIN{printf "%d", v/1e6}')
  TPS_R=$(awk -v g="$GTOK" -v d="$EVAL_DUR" 'BEGIN{ if (d>0) printf "%.2f", g/(d/1e9); else printf "0.00" }')
  say "   $PID: ${GTOK} tok in ${EMS} ms -> ${TPS_R} tok/s   (prompt ${PTOK} tok/${PMS} ms, total ${TMS} ms)"

  echo "$(ts_legacy),$HOSTNAME_S,$OLLAMA_VER,$MODEL,$I,$GTOK,$PTOK,$TPS_R,$LMS,$PMS,$EMS,$TMS,$MEM_TOTAL_MB,$SYS_USED_BASE,$SYS_USED_LOAD,$SYS_DELTA,$SYS_WIRED_BASE,$SYS_WIRED_LOAD,$WIRED_DELTA,$SYS_ANON_BASE,$SYS_ANON_LOAD,$ANON_DELTA,$RSS_BASE,$RSS_LOAD,$RSS_DELTA,$M_SIZE_MB,$M_VRAM_MB,${M_PROC// /},$NUM_CTX,$KEEP_ALIVE" >> "$CSV"

  SUM_TPS=$(awk -v s="$SUM_TPS" -v t="$TPS_R" 'BEGIN{print s+t}')
  OK_RUNS=$((OK_RUNS+1))

  # -- 통합 24컬럼 CSV용 -----------------------------------------------
  TTFT_MS=$(awk -v v="$PROMPT_EVAL_DUR" 'BEGIN{printf "%.3f", v/1e6}')
  DECODE_TOK_S=$(awk -v g="$GTOK" -v d="$EVAL_DUR" 'BEGIN{ if (d>0) printf "%.3f", g/(d/1e9); else printf "0" }')
  TOTAL_S=$(awk -v v="$TOTAL_DUR" 'BEGIN{printf "%.3f", v/1e9}')
  PEAK_RSS_GB=$(awk -v m="$PEAK_RSS_MB" 'BEGIN{printf "%.4f", m/1024}')

  ROW_NOTES="$SESSION_NOTES"
  [ "$GTOK" -ne "$GEN_TOKENS" ] && ROW_NOTES="${ROW_NOTES:+$ROW_NOTES;}early_eos=$GEN_TOKENS"
  [ "$LOAD_DUR" -ne 0 ] && ROW_NOTES="${ROW_NOTES:+$ROW_NOTES;}load_ms=$LMS"

  RUN_ID=$(uuidgen | tr 'A-Z' 'a-z')
  ROW_TS=$(ts_iso)

  printf '%s\n' "$RUN_ID,$SESSION_ID,$ROW_TS,$GIT_COMMIT,$DIRTY_FLAG,$COND,ollama,$OLLAMA_VER,$MODEL,$QUANT,$DTYPE,$DEVICE,$N_CTX,$PID,$PTOK,$GTOK,$TTFT_MS,$DECODE_TOK_S,$TOTAL_S,$PEAK_RSS_GB,$RUNTIME_ALLOC_GB,$HOSTNAME_S,$OS_NAME,$ROW_NOTES" >> "$UNIFIED_CSV"
done

if [ "$OK_RUNS" -gt 0 ]; then
  AVG_TPS=$(awk -v s="$SUM_TPS" -v n="$OK_RUNS" 'BEGIN{printf "%.2f", s/n}')
  say ""
  say "   average: ${AVG_TPS} tok/s over ${OK_RUNS}/${ACTIVE_COUNT} prompt(s)"
fi

# ----------------------------------------------------------------------------
# 5. Cleanup
# ----------------------------------------------------------------------------
if [ "$UNLOAD" -eq 1 ]; then
  say ""
  say "-- UNLOADING MODEL --"
  unload_model
  sleep "$SETTLE"
  read -r SYS_USED_END _ <<<"$(sys_mem)"
  say "   system memory used after unload: ${SYS_USED_END} MB"
fi

say ""
say "Legacy CSV row(s) appended to  : $CSV"
say "Unified CSV row(s) appended to : $UNIFIED_CSV"
say "Full log at                    : $LOG"
say "Done $(ts_legacy)"
say ""
