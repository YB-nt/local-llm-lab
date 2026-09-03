#!/usr/bin/env bash
#
# run-experiments_baseline.sh — hf_baseline.py 디코딩 파라미터 실험 일괄 실행
#
# 목적: 검증을 위한 테스트 실행을 자동화한다.
#
# 사용법
#   ./scripts/run-experiments.sh
#   ./scripts/run-experiments.sh --model Qwen/Qwen2.5-1.5B-Instruct
#   ./scripts/run-experiments.sh --only 4        # 4번 실험만
#
set -uo pipefail   # -e 는 쓰지 않는다. 실험 3은 경고를 유도하므로 중단되면 안 된다.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

MODEL="Qwen/Qwen2.5-0.5B-Instruct"
SEED=42
ONLY=""
LOG_DIR="logs"
LOG_FILE="${LOG_DIR}/hf-experiments.log"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift 2 ;;
    --seed)  SEED="$2";  shift 2 ;;
    --only)  ONLY="$2";  shift 2 ;;
    --log)   LOG_FILE="$2"; shift 2 ;;
    -h|--help)
      sed -n '2,14p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 1 ;;
  esac
done

mkdir -p "$(dirname "$LOG_FILE")"

RUN="uv run python src/hf_baseline.py --model $MODEL"

# ---------------------------------------------------------------------------
# 실험 정의: 번호 | 제목 | 관찰 포인트 | 인자
# ---------------------------------------------------------------------------
run_case() {
  local num="$1" title="$2" watch="$3"; shift 3

  if [[ -n "$ONLY" && "$ONLY" != "$num" ]]; then
    return 0
  fi

  {
    echo ""
    echo "==============================================================="
    echo "[$num] $title"
    echo "관찰 포인트: $watch"
    echo "명령: $RUN $*"
    echo "---------------------------------------------------------------"
  }
  # shellcheck disable=SC2086
  $RUN "$@"
  echo ""
}

{
  echo "==============================================================="
  echo " hf_baseline.py decoding parameter experiments"
  echo " model : $MODEL"
  echo " seed  : $SEED"
  echo " date  : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo " commit: $(git rev-parse --short HEAD 2>/dev/null || echo nogit)"
  echo "==============================================================="

  # 1. greedy 결정성
  run_case 1 "greedy 결정성" \
    "출력 3개가 완전히 동일한가. 동일하다면 벤치마크가 greedy 를 쓰는 이유가 설명된다." \
    --repeat 3

  # 2. temperature 스윕 (seed 고정 — 온도 차이만 분리하기 위함)
  for t in 0.1 0.7 1.5; do
    run_case 2 "temperature=$t" \
      "온도가 오를수록 문장이 어디서부터 무너지는가. seed 고정이므로 차이는 온도 때문이다." \
      --do-sample --temperature "$t" --seed "$SEED"
  done

  # 3. sampling 없이 temperature — 경고 유도
  run_case 3 "do_sample=False + temperature=0.7" \
    "경고가 출력되는가. temperature 는 sampling 에서만 의미가 있다." \
    --temperature 0.7

  # 4. top_k=1 이 greedy 와 같아지는가
  run_case 4 "top_k=1" \
    "1번(greedy)과 출력이 같은가. 후보를 1개로 자르면 확률이 가장 높은 것만 남는다." \
    --do-sample --top-k 1 --repeat 3 --seed "$SEED"

  # 5. top_k vs top_p
  run_case 5 "top_k=5" \
    "후보를 개수로 자르는 방식." \
    --do-sample --top-k 5 --seed "$SEED"
  run_case 5 "top_p=0.9" \
    "후보를 누적확률로 자르는 방식. top_k 결과와 무엇이 다른가." \
    --do-sample --top-p 0.9 --seed "$SEED"

  # 6. repetition_penalty — 반복 루프를 유도하려면 길게 생성해야 한다
  for rp in 1.0 1.1; do
    run_case 6 "repetition_penalty=$rp" \
      "긴 생성에서 같은 구절이 반복되는가. 작은 모델일수록 차이가 크게 보인다." \
      --repetition-penalty "$rp" --max-new-tokens 256
  done

  # 7. KV 캐시
  run_case 7 "use_cache=True (기준)" \
    "체감 속도의 기준값." \
    --max-new-tokens 128
  run_case 7 "use_cache=False" \
    "얼마나 느려지는가. 캐시가 없으면 매 스텝마다 전체 시퀀스를 다시 계산한다." \
    --no-cache --max-new-tokens 128

  echo ""
  echo "==============================================================="
  echo " Completed."
  echo "==============================================================="
} 2>&1 | tee "$LOG_FILE"

echo ""
echo "로그: $LOG_FILE"
