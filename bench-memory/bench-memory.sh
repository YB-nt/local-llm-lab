#!/usr/bin/env bash
#
# bench-memory.sh - Log memory usage before/after loading an Ollama model,
#                   plus per-run generation throughput, for benchmark records.
#
# macOS-focused (uses `top` / `ps`), talks to the Ollama HTTP API.
#
# Usage:
#   ./bench-memory.sh MODEL [options]
#
# Options:
#   -p, --prompt TEXT       Benchmark prompt (default: a fixed ~200-word task)
#   -f, --prompt-file PATH   Read benchmark prompt from a file
#   -n, --runs N             Number of benchmark generations (default: 3)
#   -c, --num-ctx N          Force context window size (options.num_ctx)
#   -k, --keep-alive DUR     Ollama keep_alive for the model (default: 5m)
#       --host URL           Ollama base URL (default: http://localhost:11434)
#       --csv PATH           CSV results file (default: ./bench-results.csv)
#       --log PATH           Human-readable log (default: ./bench-memory.log)
#       --no-fresh           Do NOT unload the model first; measure current state
#       --no-unload          Leave the model loaded when finished
#       --settle SECONDS     Wait after (un)load before measuring (default: 3)
#   -h, --help              Show this help
#
# Examples:
#   ./bench-memory.sh llama3.1:8b-instruct-q4_K_M
#   ./bench-memory.sh llama3.1:8b-instruct-q4_K_M -n 5 -c 8192 -k 10m
#   ./bench-memory.sh qwen2.5:14b -f ./prompt.txt --csv runs/qwen.csv

set -uo pipefail

# ----------------------------------------------------------------------------
# Defaults
# ----------------------------------------------------------------------------
HOST="http://localhost:11434"
RUNS=3
KEEP_ALIVE="5m"
NUM_CTX=""
CSV="./bench-results.csv"
LOG="./bench-memory.log"
SETTLE=3
FRESH=1
UNLOAD=1
PROMPT='Explain, in about 200 words, how a transformer neural network processes a
sequence of tokens: embeddings, self-attention, feed-forward layers, residual
connections and layer normalization, and how the final logits are produced.
Keep it accurate and self-contained.'

# ----------------------------------------------------------------------------
# Args
# ----------------------------------------------------------------------------
usage() { sed -n '2,40p' "$0" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }

[ $# -eq 0 ] && usage 1
MODEL=""
while [ $# -gt 0 ]; do
  case "$1" in
    -p|--prompt)      PROMPT="$2"; shift 2 ;;
    -f|--prompt-file) PROMPT="$(cat "$2")"; shift 2 ;;
    -n|--runs)        RUNS="$2"; shift 2 ;;
    -c|--num-ctx)     NUM_CTX="$2"; shift 2 ;;
    -k|--keep-alive)  KEEP_ALIVE="$2"; shift 2 ;;
    --host)           HOST="${2%/}"; shift 2 ;;
    --csv)            CSV="$2"; shift 2 ;;
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

# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
ts()  { date +"%Y-%m-%dT%H:%M:%S%z"; }
say() { printf '%s\n' "$*" | tee -a "$LOG"; }

# System-wide memory, parsed from `vm_stat` (page-granular, instant).
# Prints: "<used_mb> <free_mb> <wired_mb> <compressed_mb> <anon_mb>"
#   used  = total - (free + speculative + inactive + purgeable)   ~ "Memory Used"
#   anon  = anonymous pages, the clearest signal for a model held in RAM
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
      total = (free+spec+inact+purg+wired+comp+anon)   # not exact total, only for reference
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

# Ollama's own accounting for the loaded model via /api/ps.
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

# JSON body for /api/generate. $1 = prompt, $2 = keep_alive
gen_body() {
  jq -nc --arg m "$MODEL" --arg p "$1" --arg k "$2" --arg ctx "$NUM_CTX" '
    {model:$m, prompt:$p, stream:false, keep_alive:$k}
    + (if $ctx == "" then {} else {options:{num_ctx:($ctx|tonumber)}} end)'
}

unload_model() {
  curl -s "$HOST/api/generate" -d "$(gen_body '' '0')" >/dev/null
  for _ in $(seq 1 20); do model_loaded || break; sleep 0.5; done
}

# ----------------------------------------------------------------------------
# Preflight
# ----------------------------------------------------------------------------
curl -sf "$HOST/api/version" >/dev/null || { echo "ERROR: Ollama not reachable at $HOST" >&2; exit 1; }
OLLAMA_VER=$(curl -s "$HOST/api/version" | jq -r '.version // "?"')
HOSTNAME_S=$(scutil --get ComputerName 2>/dev/null || hostname)
HOSTNAME_S=$(printf '%s' "$HOSTNAME_S" | sed 's/[[:space:],]\{1,\}/_/g')   # keep CSV single-token
MEM_TOTAL_MB=$(( $(sysctl -n hw.memsize) / 1048576 ))

mkdir -p "$(dirname "$CSV")" "$(dirname "$LOG")"
if [ ! -f "$CSV" ]; then
  echo "timestamp,host,ollama_version,model,run,gen_tokens,prompt_tokens,tokens_per_sec,load_ms,prompt_eval_ms,eval_ms,total_ms,mem_total_mb,sys_used_baseline_mb,sys_used_loaded_mb,sys_used_delta_mb,sys_wired_baseline_mb,sys_wired_loaded_mb,sys_wired_delta_mb,sys_anon_baseline_mb,sys_anon_loaded_mb,sys_anon_delta_mb,ollama_rss_baseline_mb,ollama_rss_loaded_mb,ollama_rss_delta_mb,model_size_mb,model_vram_mb,processor,num_ctx,keep_alive" > "$CSV"
fi

say "======================================================================"
say "  Ollama memory benchmark  |  $(ts)"
say "  host=$HOSTNAME_S  ollama=$OLLAMA_VER  ram=${MEM_TOTAL_MB}MB"
say "  model=$MODEL  runs=$RUNS  num_ctx=${NUM_CTX:-default}  keep_alive=$KEEP_ALIVE"
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
LOAD_RESP=$(curl -s "$HOST/api/generate" -d "$(gen_body '' "$KEEP_ALIVE")")
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
# 3. Benchmark runs
# ----------------------------------------------------------------------------
say ""
say "-- GENERATION RUNS (n=$RUNS) ----------------------------------"
SUM_TPS=0; OK_RUNS=0
for i in $(seq 1 "$RUNS"); do
  R=$(curl -s "$HOST/api/generate" -d "$(gen_body "$PROMPT" "$KEEP_ALIVE")")
  if ! printf '%s' "$R" | jq -e '.done' >/dev/null 2>&1; then
    say "   run $i: ERROR $(printf '%s' "$R" | jq -r '.error // .' 2>/dev/null)"
    continue
  fi
  read -r GTOK PTOK LMS PMS EMS TMS TPS <<<"$(printf '%s' "$R" | jq -r '
    [ (.eval_count // 0),
      (.prompt_eval_count // 0),
      ((.load_duration // 0)/1e6 | floor),
      ((.prompt_eval_duration // 0)/1e6 | floor),
      ((.eval_duration // 0)/1e6 | floor),
      ((.total_duration // 0)/1e6 | floor),
      (if (.eval_duration // 0) > 0 then (.eval_count // 0) / ((.eval_duration)/1e9) else 0 end)
    ] | @tsv')"
  TPS_R=$(awk -v t="$TPS" 'BEGIN{printf "%.2f", t}')
  say "   run $i: ${GTOK} tok in ${EMS} ms -> ${TPS_R} tok/s   (prompt ${PTOK} tok/${PMS} ms, total ${TMS} ms)"

  echo "$(ts),$HOSTNAME_S,$OLLAMA_VER,$MODEL,$i,$GTOK,$PTOK,$TPS_R,$LMS,$PMS,$EMS,$TMS,$MEM_TOTAL_MB,$SYS_USED_BASE,$SYS_USED_LOAD,$SYS_DELTA,$SYS_WIRED_BASE,$SYS_WIRED_LOAD,$WIRED_DELTA,$SYS_ANON_BASE,$SYS_ANON_LOAD,$ANON_DELTA,$RSS_BASE,$RSS_LOAD,$RSS_DELTA,$M_SIZE_MB,$M_VRAM_MB,${M_PROC// /},${NUM_CTX:-default},$KEEP_ALIVE" >> "$CSV"

  SUM_TPS=$(awk -v s="$SUM_TPS" -v t="$TPS_R" 'BEGIN{print s+t}')
  OK_RUNS=$((OK_RUNS+1))
done

if [ "$OK_RUNS" -gt 0 ]; then
  AVG_TPS=$(awk -v s="$SUM_TPS" -v n="$OK_RUNS" 'BEGIN{printf "%.2f", s/n}')
  say ""
  say "   average: ${AVG_TPS} tok/s over ${OK_RUNS} run(s)"
fi

# ----------------------------------------------------------------------------
# 4. Cleanup
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
say "CSV row(s) appended to : $CSV"
say "Full log at            : $LOG"
say "Done $(ts)"
say ""
