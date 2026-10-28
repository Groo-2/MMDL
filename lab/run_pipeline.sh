#!/usr/bin/env bash
# 실습실(RTX 4090 24GB) E0 baseline + E1 SSMR GRPO 파이프라인. 저장소 루트에서 실행하고, 오래 걸리므로 tmux 안에서 돌린다.
#
#   tmux new -s mmdl
#   bash lab/run_pipeline.sh env smoke             # 환경 점검 + Music 1회 (시간/동작 확인)
#   bash lab/run_pipeline.sh data screen stress    # SSMR 변환, 통과율 측정+선별, GRPO 최악 조건 1스텝
#   bash lab/run_pipeline.sh pilot                 # GRPO 2스텝 (스텝 시간 실측)
#   bash lab/run_pipeline.sh baseline_music4 holdout_base baseline_full2
#   bash lab/run_pipeline.sh grpo merge holdout_new music4 full2 report
#   bash lab/run_pipeline.sh all                   # 위 전부를 순서대로 (music_extend 제외)
#
# 단계: env preflight smoke data screen stress pilot baseline_music4 baseline_full2 holdout_base
#       grpo merge holdout_new music4 full2 report  /  music_extend_base music_extend_new (아래 주의)
#
# 주의 1) 단계가 실패하면 코드를 고치기 전에 CUDA/패키지 점검(lab/env_triage.py)을 자동으로 먼저 실행한다.
# 주의 2) music_extend_*(Music을 8번으로 확장)는 사용자 확인 없이는 실행되지 않는다. all에도 포함되지 않는다.
#         확인했다면 직접:  CONFIRM_EXTEND_TO_8=yes bash lab/run_pipeline.sh music_extend_base music_extend_new
# 모든 단계는 같은 커맨드를 다시 실행하면 이어서 한다 (추론 resume, 학습 --resume).
# 환경변수: STEPS=140 PROMPTS_PER_STEP=8 PER_CAT=280 POOL=3000 MAX_COMPLETION=2048 TAG_BASE=baseline TAG_NEW=grpo_run1
#           CONDA_ENV=mmdl_lab (venv 대신 conda 환경을 쓸 때. tmux 안에서 conda activate는 하지 말고 이 변수만 export)
#           DRY_RUN=1 (명령만 출력하고 실행하지 않음)
set -uo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

DRY_RUN="${DRY_RUN:-0}"
if [[ "$DRY_RUN" != "1" ]]; then
  if [[ -n "${CONDA_ENV:-}" ]]; then
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
  else
    # shellcheck disable=SC1091
    source "${VENV:-.venv_lab}/bin/activate"
  fi
fi

STEPS="${STEPS:-140}"
PROMPTS_PER_STEP="${PROMPTS_PER_STEP:-8}"
PER_CAT="${PER_CAT:-280}"
POOL="${POOL:-3000}"
MAX_COMPLETION="${MAX_COMPLETION:-2048}"
TAG_BASE="${TAG_BASE:-baseline}"
TAG_NEW="${TAG_NEW:-grpo_run1}"
DATA_DIR="${DATA_DIR:-data/ssmr}"
MODEL="Qwen/Qwen3-VL-4B-Instruct"
REVISION="ebb281ec70b05090aa6165b016eac8ec08e71b17"

mkdir -p logs results data
export PYTHONUNBUFFERED=1

run() {
  echo "+ $*"
  if [[ "$DRY_RUN" == "1" ]]; then return 0; fi
  "$@"
}

confirm_extend() {
  if [[ "${CONFIRM_EXTEND_TO_8:-}" != "yes" ]]; then
    echo "music_extend(Music을 8번으로 확장)는 사용자 확인 없이 실행하지 않는다."
    echo "  music4 결과를 확인한 뒤 확장하기로 했다면 직접 설정하고 다시 실행하세요:"
    echo "    CONFIRM_EXTEND_TO_8=yes bash lab/run_pipeline.sh $1"
    exit 3
  fi
}

TRAIN_ARGS=(--data-dir "$DATA_DIR" --train-file train_subset.jsonl --prompts-per-step "$PROMPTS_PER_STEP"
            --max-completion-length "$MAX_COMPLETION")

run_stage() {
  local stage="$1"
  echo; echo "################ [$(date '+%F %T')] stage: $stage ################"
  case "$stage" in
    env)       run python lab/env_triage.py ;;
    preflight) run python lab/preflight.py --stage eval ;;
    smoke)
      run python lab/preflight.py --stage eval || return $?
      run python lab/eval_repeat.py --tag smoke --model "$MODEL" --revision "$REVISION" --runs 1 --subject Music
      ;;
    data)
      run python music_ft/build_ssmr.py --out-dir "$DATA_DIR" --cache-dir data/hf_cache || return $?
      run python music_ft/dedupe.py --data-dir "$DATA_DIR" --cache-dir data/hf_cache || return $?
      (cd music_ft && run python -m unittest test_reward)
      ;;
    screen)
      run python lab/preflight.py --stage screen --data-dir "$DATA_DIR" || return $?
      run python music_ft/select_ssmr.py --data-dir "$DATA_DIR" --model "$MODEL" --revision "$REVISION" \
        --pool-size "$POOL" --per-category "$PER_CAT" --max-tokens "$MAX_COMPLETION" --start-server
      ;;
    stress)
      run python lab/preflight.py --stage train --data-dir "$DATA_DIR" || return $?
      run rm -rf results/grpo_stress
      run python music_ft/train_grpo.py "${TRAIN_ARGS[@]}" --output-dir results/grpo_stress --max-steps 1 --stress
      ;;
    pilot)
      run python lab/preflight.py --stage train --data-dir "$DATA_DIR" || return $?
      run python music_ft/train_grpo.py "${TRAIN_ARGS[@]}" --output-dir results/grpo_pilot \
        --max-steps 2 --save-steps 2 --resume
      ;;
    baseline_music4)
      run python lab/preflight.py --stage eval || return $?
      run python lab/eval_repeat.py --tag "$TAG_BASE" --preset music4 --model "$MODEL" --revision "$REVISION"
      ;;
    baseline_full2)
      run python lab/preflight.py --stage eval || return $?
      run python lab/eval_repeat.py --tag "$TAG_BASE" --preset full2 --model "$MODEL" --revision "$REVISION"
      ;;
    holdout_base)
      run python lab/preflight.py --stage holdout --data-dir "$DATA_DIR" || return $?
      run python lab/eval_holdout.py --tag "$TAG_BASE" --model "$MODEL" --revision "$REVISION" --data-dir "$DATA_DIR"
      ;;
    grpo)
      run python lab/preflight.py --stage train --data-dir "$DATA_DIR" || return $?
      run python music_ft/train_grpo.py "${TRAIN_ARGS[@]}" --output-dir "results/$TAG_NEW" \
        --max-steps "$STEPS" --save-steps 25 --resume
      ;;
    merge)
      run python music_ft/merge_lora.py --lora "results/$TAG_NEW/final_lora" --out "results/$TAG_NEW/merged"
      ;;
    holdout_new)
      run python lab/preflight.py --stage holdout --data-dir "$DATA_DIR" || return $?
      run python lab/eval_holdout.py --tag "$TAG_NEW" --model "results/$TAG_NEW/merged" --data-dir "$DATA_DIR"
      ;;
    music4)
      run python lab/preflight.py --stage eval || return $?
      run python lab/eval_repeat.py --tag "$TAG_NEW" --preset music4 --model "results/$TAG_NEW/merged"
      ;;
    full2)
      run python lab/preflight.py --stage eval || return $?
      run python lab/eval_repeat.py --tag "$TAG_NEW" --preset full2 --model "results/$TAG_NEW/merged"
      ;;
    music_extend_base)
      confirm_extend music_extend_base
      run python lab/eval_repeat.py --tag "$TAG_BASE" --preset music_extend --model "$MODEL" --revision "$REVISION"
      ;;
    music_extend_new)
      confirm_extend music_extend_new
      run python lab/eval_repeat.py --tag "$TAG_NEW" --preset music_extend --model "results/$TAG_NEW/merged"
      ;;
    report)
      for t in "$TAG_BASE" "$TAG_NEW"; do
        run python lab/summarize_runs.py summary "results/$t/music" --subject Music
        run python lab/summarize_runs.py summary "results/$t/full"
      done
      run python lab/summarize_runs.py compare "results/$TAG_BASE/music" "results/$TAG_NEW/music" --subject Music
      run python lab/summarize_runs.py compare "results/$TAG_BASE/full" "results/$TAG_NEW/full"
      ;;
    *) echo "알 수 없는 단계: $stage" >&2; return 1 ;;
  esac
}

if [[ $# -eq 0 ]]; then
  sed -n '2,19p' "$0"; exit 1
fi

STAGES=("$@")
if [[ "${STAGES[0]}" == "all" ]]; then
  # music_extend_*는 사용자 확인이 필요하므로 포함하지 않는다
  STAGES=(data screen stress pilot baseline_music4 holdout_base baseline_full2 grpo merge holdout_new music4 full2 report)
fi

for s in "${STAGES[@]}"; do
  run_stage "$s" 2>&1 | tee -a "logs/${s}.log"
  rc=${PIPESTATUS[0]}
  if [[ "$rc" -ne 0 ]]; then
    echo; echo "단계 '$s' 실패 (종료 코드 $rc). logs/${s}.log"
    if [[ "$rc" -eq 3 ]]; then exit 3; fi   # 사용자 확인 대기(안전장치)는 환경 문제가 아니다
    echo "코드를 고치기 전에 CUDA/패키지 버전부터 점검합니다:"
    if [[ "$DRY_RUN" == "1" ]]; then echo "+ python lab/env_triage.py --after-failure"; else python lab/env_triage.py --after-failure || true; fi
    exit "$rc"
  fi
done
echo; echo "요청한 단계 완료"
