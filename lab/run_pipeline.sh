#!/usr/bin/env bash
# 실습실 전체 파이프라인. 저장소 루트에서 실행하고, 오래 걸리므로 tmux 안에서 돌린다.
#
#   tmux new -s mmdl
#   bash lab/run_pipeline.sh smoke      # 환경/속도 확인 (Music 1과목, 1회)
#   bash lab/run_pipeline.sh data pilot # 데이터 변환 + GRPO 2스텝 파일럿
#   bash lab/run_pipeline.sh baseline   # Run B 설정으로 8회 (seed 42~49)
#   bash lab/run_pipeline.sh train merge trained report
#   bash lab/run_pipeline.sh all        # 위 전부를 순서대로
#
# 모든 단계는 멈췄다가 같은 커맨드로 다시 실행하면 이어서 한다 (추론 resume, 학습 --resume).
# 환경변수로 조절: RUNS=8 STEPS=100 PROMPTS_PER_STEP=8 MAX_COMPLETION=2048 TAG_BASE=baseline TAG_NEW=grpo_run1
set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "${VENV:-.venv_lab}/bin/activate"

RUNS="${RUNS:-8}"
STEPS="${STEPS:-100}"
PROMPTS_PER_STEP="${PROMPTS_PER_STEP:-8}"
MAX_COMPLETION="${MAX_COMPLETION:-2048}"
TAG_BASE="${TAG_BASE:-baseline}"
TAG_NEW="${TAG_NEW:-grpo_run1}"
DATA_DIR="${DATA_DIR:-data/ssmr}"
MODEL="Qwen/Qwen3-VL-4B-Instruct"
REVISION="ebb281ec70b05090aa6165b016eac8ec08e71b17"

mkdir -p logs results data
export PYTHONUNBUFFERED=1

run_stage() {
  local stage="$1"
  echo; echo "################ [$(date '+%F %T')] stage: $stage ################"
  case "$stage" in
    smoke)
      python lab/check_env.py
      python lab/eval_repeat.py --tag smoke --model "$MODEL" --revision "$REVISION" --runs 1 --subject Music
      ;;
    data)
      python music_ft/build_ssmr.py --out-dir "$DATA_DIR" --cache-dir data/hf_cache
      python music_ft/dedupe.py --data-dir "$DATA_DIR" --cache-dir data/hf_cache
      (cd music_ft && python -m unittest test_reward)
      ;;
    pilot)
      python music_ft/train_grpo.py --data-dir "$DATA_DIR" --output-dir results/grpo_pilot \
        --max-steps 2 --save-steps 2 --prompts-per-step "$PROMPTS_PER_STEP" --max-completion-length "$MAX_COMPLETION" --resume
      ;;
    baseline)
      python lab/eval_repeat.py --tag "$TAG_BASE" --model "$MODEL" --revision "$REVISION" --runs "$RUNS"
      ;;
    train)
      python music_ft/train_grpo.py --data-dir "$DATA_DIR" --output-dir "results/$TAG_NEW" \
        --max-steps "$STEPS" --save-steps 5 --prompts-per-step "$PROMPTS_PER_STEP" \
        --max-completion-length "$MAX_COMPLETION" --resume
      ;;
    merge)
      python music_ft/merge_lora.py --lora "results/$TAG_NEW/final_lora" --out "results/$TAG_NEW/merged"
      ;;
    trained)
      python lab/eval_repeat.py --tag "$TAG_NEW" --model "results/$TAG_NEW/merged" --runs "$RUNS"
      ;;
    report)
      python lab/summarize_runs.py summary "results/$TAG_BASE"
      python lab/summarize_runs.py summary "results/$TAG_NEW"
      python lab/summarize_runs.py compare "results/$TAG_BASE" "results/$TAG_NEW" | tee "results/compare_${TAG_BASE}_vs_${TAG_NEW}.txt"
      ;;
    *) echo "알 수 없는 단계: $stage" >&2; exit 1;;
  esac
}

if [[ $# -eq 0 ]]; then
  sed -n '2,13p' "$0"; exit 1
fi

STAGES=("$@")
if [[ "${STAGES[0]}" == "all" ]]; then
  STAGES=(data pilot baseline train merge trained report)
fi

for s in "${STAGES[@]}"; do
  run_stage "$s" 2>&1 | tee -a "logs/${s}.log"
  [[ "${PIPESTATUS[0]}" -eq 0 ]] || { echo "단계 '$s' 실패. logs/${s}.log 확인"; exit 1; }
done
echo; echo "모든 단계 완료"
