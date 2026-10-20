#!/bin/bash
# pilot20(악보 판독)을 팀 baseline(Run B) 파이프라인 그대로 평가하는 단일 진입점.
# baseline/scripts/run_mmmu_eval.sh와 같은 순서(모델 서버 -> infer -> judge 서버 -> judge -> score)이고,
# vLLM 서버 옵션과 context 길이, GPU 메모리 비율, judge 모델은 baseline 스크립트에서 값을 그대로 읽어온다.
# baseline과 다른 점: ① 데이터(pilot20) ② 순수 greedy (temperature 0, presence_penalty 0; baseline은 0.7 / 1.5). 이유는 run_pilot20.py 상단 참고.
#
# 사용법:
#   bash run_pilot20.sh \
#     --model_path Qwen/Qwen3-VL-4B-Instruct \
#     --output_dir results/pilot20_baseline
#
# fine-tuning 후 재평가: --model_path만 체크포인트 경로로 바꾼다 (--model_revision은 생략). 나머지는 그대로.
# judge(Llama-3.1-8B-Instruct)는 gated 모델이라 HF_TOKEN이 필요하다 (judge 없이 돌리려면 --no_judge).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_SH="$HERE/../baseline/scripts/run_mmmu_eval.sh"
cd "$HERE"

[[ -f "$BASE_SH" ]] || { echo "baseline script not found: $BASE_SH (저장소 전체를 clone했는지 확인)" >&2; exit 1; }

# baseline 스크립트의 설정값을 그대로 읽는다 (값을 한 곳에서만 정의)
sh_val() { grep -m1 "^$1=" "$BASE_SH" | sed -E 's/^[A-Z_]+="?([^"#]*)"?.*/\1/' | xargs; }

MAX_NEW_TOKENS="$(sh_val MAX_NEW_TOKENS)"
MAX_MODEL_LEN="$(sh_val MAX_MODEL_LEN)"
GPU_MEM_UTIL="$(sh_val GPU_MEM_UTIL)"
JUDGE_REPO="$(sh_val JUDGE_REPO)"
JUDGE_REVISION="$(sh_val JUDGE_REVISION)"
JUDGE_MAX_MODEL_LEN="$(sh_val JUDGE_MAX_MODEL_LEN)"
CONCURRENCY="$(sh_val CONCURRENCY)"
for v in MAX_NEW_TOKENS MAX_MODEL_LEN GPU_MEM_UTIL JUDGE_REPO JUDGE_REVISION JUDGE_MAX_MODEL_LEN CONCURRENCY; do
  [[ -n "${!v}" ]] || { echo "baseline script에서 $v 값을 읽지 못했습니다" >&2; exit 1; }
done

PINNED_MODEL_REPO="Qwen/Qwen3-VL-4B-Instruct"
PINNED_MODEL_REVISION="ebb281ec70b05090aa6165b016eac8ec08e71b17"

MODEL_PATH=""
MODEL_REVISION=""
OUTPUT_DIR=""
TEMPERATURE="0"
PRESENCE_PENALTY="0"
LIMIT=""
RUN_JUDGE=1
PORT=8000
JUDGE_PORT=8001

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model_path) MODEL_PATH="$2"; shift 2;;
    --model_revision) MODEL_REVISION="$2"; shift 2;;
    --output_dir) OUTPUT_DIR="$2"; shift 2;;
    --temperature) TEMPERATURE="$2"; shift 2;;
    --presence_penalty) PRESENCE_PENALTY="$2"; shift 2;;
    --limit) LIMIT="$2"; shift 2;;
    --max_new_tokens) MAX_NEW_TOKENS="$2"; shift 2;;
    --max_model_len) MAX_MODEL_LEN="$2"; shift 2;;
    --gpu_mem_util) GPU_MEM_UTIL="$2"; shift 2;;
    --no_judge) RUN_JUDGE=0; shift 1;;
    *) echo "Unknown arg: $1" >&2; exit 1;;
  esac
done

: "${MODEL_PATH:?--model_path is required}"
: "${OUTPUT_DIR:?--output_dir is required}"

# 공식 repo id로 실행할 때는 revision을 항상 pin한다 (assignment_guidance.md §1.1)
if [[ "$MODEL_PATH" == "$PINNED_MODEL_REPO" && -z "$MODEL_REVISION" ]]; then
  MODEL_REVISION="$PINNED_MODEL_REVISION"
  echo "[info] --model_revision 생략: $PINNED_MODEL_REPO 는 고정 revision($MODEL_REVISION)으로 받습니다."
fi

if [[ "$RUN_JUDGE" == "1" ]]; then
  : "${HF_TOKEN:?HF_TOKEN is required for the gated judge model (or pass --no_judge)}"
fi

mkdir -p "$OUTPUT_DIR"

LIMIT_ARGS=()
if [[ -n "$LIMIT" ]]; then
  LIMIT_ARGS=(--limit "$LIMIT")
  echo "⚠️  스모크 테스트 모드: 앞 $LIMIT 문항만 처리합니다 (평가 수치로 쓸 수 없음)"
fi

REVISION_ARGS=()
if [[ -n "$MODEL_REVISION" ]]; then
  REVISION_ARGS=(--revision "$MODEL_REVISION")
fi

SERVER_PID=""
GPU_LOG_PID=""

cleanup() {
  if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  if [[ -n "$GPU_LOG_PID" ]] && kill -0 "$GPU_LOG_PID" 2>/dev/null; then
    kill "$GPU_LOG_PID" 2>/dev/null || true
  fi
  SERVER_PID=""
  GPU_LOG_PID=""
}
trap cleanup EXIT INT TERM

# $1=포트, $2=로그 파일. 서버가 /v1/models에 응답할 때까지 기다린다 (모델 다운로드 포함 최대 40분).
wait_for_server() {
  local port="$1" log="$2"
  for _ in $(seq 1 480); do
    if curl -sf "http://localhost:${port}/v1/models" >/dev/null; then
      return 0
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      echo "vLLM server exited. Last log lines:" >&2
      tail -50 "$log" >&2
      exit 1
    fi
    sleep 5
  done
  echo "vLLM server did not become ready in time. See $log" >&2
  exit 1
}

start_gpu_log() {
  if command -v nvidia-smi >/dev/null; then
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -l 5 >> "$OUTPUT_DIR/gpu_mem_$1.log" &
    GPU_LOG_PID=$!
  fi
}

echo "=== [1/3] infer: $MODEL_PATH ${MODEL_REVISION:+(revision=$MODEL_REVISION)} temperature=$TEMPERATURE presence_penalty=$PRESENCE_PENALTY ==="
VLLM_USE_V2_MODEL_RUNNER=0 vllm serve "$MODEL_PATH" \
  ${REVISION_ARGS[@]+"${REVISION_ARGS[@]}"} \
  --served-model-name qwen3-vl-4b \
  --dtype bfloat16 \
  --seed 42 \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  --port "$PORT" \
  > "$OUTPUT_DIR/vllm_infer.log" 2>&1 &
SERVER_PID=$!
wait_for_server "$PORT" "$OUTPUT_DIR/vllm_infer.log"
start_gpu_log infer

python run_pilot20.py infer \
  --output-dir "$OUTPUT_DIR" \
  --temperature "$TEMPERATURE" \
  --presence-penalty "$PRESENCE_PENALTY" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --concurrency "$CONCURRENCY" \
  --api-base "http://localhost:${PORT}/v1" \
  --model-path "$MODEL_PATH" \
  --model-revision "${MODEL_REVISION:-local}" \
  ${LIMIT_ARGS[@]+"${LIMIT_ARGS[@]}"}
  # top_p/top_k/repetition_penalty/seed는 baseline(run_mmmu_eval.py)에 고정돼 있다 — 여기서 바꾸지 말 것.

cleanup

if [[ "$RUN_JUDGE" == "1" ]]; then
  echo "=== [2/3] judge: $JUDGE_REPO (revision=$JUDGE_REVISION) ==="
  vllm serve "$JUDGE_REPO" \
    --revision "$JUDGE_REVISION" \
    --served-model-name judge \
    --dtype bfloat16 \
    --seed 42 \
    --max-model-len "$JUDGE_MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --port "$JUDGE_PORT" \
    > "$OUTPUT_DIR/vllm_judge.log" 2>&1 &
  SERVER_PID=$!
  wait_for_server "$JUDGE_PORT" "$OUTPUT_DIR/vllm_judge.log"
  start_gpu_log judge

  python run_pilot20.py judge \
    --output-dir "$OUTPUT_DIR" \
    --judge-api-base "http://localhost:${JUDGE_PORT}/v1" \
    --judge-model judge \
    --judge-revision "$JUDGE_REVISION" \
    ${LIMIT_ARGS[@]+"${LIMIT_ARGS[@]}"}

  cleanup
else
  echo "=== [2/3] judge: skipped (--no_judge) — rule 파서 실패 문항은 오답 처리 ==="
fi

echo "=== [3/3] score ==="
python run_pilot20.py score --output-dir "$OUTPUT_DIR" ${LIMIT_ARGS[@]+"${LIMIT_ARGS[@]}"}

echo "=== Done. See $OUTPUT_DIR/summary.json ==="
