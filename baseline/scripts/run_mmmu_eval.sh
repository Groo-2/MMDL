#!/bin/bash
# MMMU validation 평가 단일 진입점 (assignment_guidance.md §2, 템플릿 §1 "한 커맨드로 재현").
#   1) 평가 모델 vLLM 서버 기동 → infer → 서버 종료
#   2) judge 모델 vLLM 서버 기동 → judge (rule 파서 실패 문항만) → 서버 종료
#   3) score → summary.json / subject_scores.csv
# 중간에 끊겨도 같은 커맨드를 다시 실행하면 끝난 문항은 건너뛴다 (resume).
#
# 사용법 (baseline):
#   bash scripts/run_mmmu_eval.sh \
#     --model_path Qwen/Qwen3-VL-4B-Instruct \
#     --model_revision ebb281ec70b05090aa6165b016eac8ec08e71b17 \
#     --data_root /path/to/hf_cache/mmmu \
#     --output_dir ./results/baseline_mmmu_pro_cot
#
# fine-tuning 후 재평가: --model_path만 체크포인트 경로로 바꾸고 --model_revision은 생략한다.
# judge(Llama-3.1-8B-Instruct)는 gated 모델이라 HF_TOKEN 환경변수가 필요하다.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

MODEL_PATH=""
MODEL_REVISION=""
DATA_ROOT=""
OUTPUT_DIR=""
PROMPT="mmmu_pro_cot"
LIMIT=""
SUBJECT=""
CONCURRENCY=32
MAX_NEW_TOKENS=32768
# 32768 + 최대 프롬프트(약 5.7K) 여유. Qwen 공식은 128000이지만 38.5K 이상이면 결과가 같고,
# 128000은 KV cache 약 17.6GiB가 필요해 RTX 4090에서 서버가 뜨지 않는다 (README 참고).
MAX_MODEL_LEN=40960
GPU_MEM_UTIL=0.90
RUN_JUDGE=1
JUDGE_REPO="meta-llama/Llama-3.1-8B-Instruct"
JUDGE_REVISION="0e9e39f249a16976918f6564b8830bc894c89659"
JUDGE_MAX_MODEL_LEN=20480
PORT=8000
JUDGE_PORT=8001

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model_path) MODEL_PATH="$2"; shift 2;;
    --model_revision) MODEL_REVISION="$2"; shift 2;;
    --data_root) DATA_ROOT="$2"; shift 2;;
    --output_dir) OUTPUT_DIR="$2"; shift 2;;
    --prompt) PROMPT="$2"; shift 2;;
    --limit) LIMIT="$2"; shift 2;;
    --subject) SUBJECT="$2"; shift 2;;
    --concurrency) CONCURRENCY="$2"; shift 2;;
    --max_new_tokens) MAX_NEW_TOKENS="$2"; shift 2;;
    --max_model_len) MAX_MODEL_LEN="$2"; shift 2;;
    --gpu_mem_util) GPU_MEM_UTIL="$2"; shift 2;;
    --no_judge) RUN_JUDGE=0; shift 1;;
    *) echo "Unknown arg: $1" >&2; exit 1;;
  esac
done

: "${MODEL_PATH:?--model_path is required}"
: "${DATA_ROOT:?--data_root is required}"
: "${OUTPUT_DIR:?--output_dir is required}"

mkdir -p "$OUTPUT_DIR" "$DATA_ROOT"

SCOPE_ARGS=(--all)
if [[ -n "$SUBJECT" ]]; then
  SCOPE_ARGS=(--subject "$SUBJECT")
fi
if [[ -n "$LIMIT" ]]; then
  SCOPE_ARGS+=(--limit "$LIMIT")
  echo "⚠️  스모크 테스트 모드: 과목당 앞 $LIMIT 문항만 처리합니다 (baseline 수치로 쓸 수 없음)"
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

# $1=stage 이름. peak VRAM 기록용 (보고서 §1)
start_gpu_log() {
  if command -v nvidia-smi >/dev/null; then
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -l 5 >> "$OUTPUT_DIR/gpu_mem_$1.log" &
    GPU_LOG_PID=$!
  fi
}

echo "=== [1/3] infer: $MODEL_PATH ${MODEL_REVISION:+(revision=$MODEL_REVISION)} prompt=$PROMPT ==="
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

python mmmu_eval/run_mmmu_eval.py infer "${SCOPE_ARGS[@]}" \
  --data-root "$DATA_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --prompt "$PROMPT" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --concurrency "$CONCURRENCY" \
  --api-base "http://localhost:${PORT}/v1" \
  --model-path "$MODEL_PATH" \
  --model-revision "${MODEL_REVISION:-local}"
  # sampling recipe(temperature/top_p/top_k/penalty/seed)는 run_mmmu_eval.py에 고정돼 있다 — 여기서 바꾸지 말 것.

cleanup

if [[ "$RUN_JUDGE" == "1" ]]; then
  echo "=== [2/3] judge: $JUDGE_REPO (revision=$JUDGE_REVISION) ==="
  : "${HF_TOKEN:?HF_TOKEN is required for the gated judge model (or pass --no_judge)}"
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

  python mmmu_eval/run_mmmu_eval.py judge "${SCOPE_ARGS[@]}" \
    --output-dir "$OUTPUT_DIR" \
    --judge-api-base "http://localhost:${JUDGE_PORT}/v1" \
    --judge-model judge \
    --judge-revision "$JUDGE_REVISION"

  cleanup
else
  echo "=== [2/3] judge: skipped (--no_judge) — rule 파서 실패 문항은 오답 처리 ==="
fi

echo "=== [3/3] score ==="
python mmmu_eval/run_mmmu_eval.py score "${SCOPE_ARGS[@]}" --output-dir "$OUTPUT_DIR"

echo "=== Done. See $OUTPUT_DIR/summary.json ==="
