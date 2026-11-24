#!/usr/bin/env bash
# MMMU test와 MMMU-Pro 평가 실행 (제출 모델 dpo_dvmx1, 선택적으로 base).
#
#   bash lab/eval_extra.sh check     # 1) 데이터 점검 (내려받기 포함, GPU 불필요, 한 번만)
#   bash lab/eval_extra.sh smoke     # 2) 벤치마크마다 과목당 1문항(30문항)을 끝까지 돌려 본다 (서버 기동 확인, 약 10분 x 벤치마크 수)
#   bash lab/eval_extra.sh plan      # 3) 무엇이 남았는지만 출력
#   bash lab/eval_extra.sh run       # 4) 본 평가. 끊겨도 같은 명령을 다시 실행하면 이어서 한다
#
# 이미 code/ 폴더, 평가용 conda 환경(CUDA 13, mmdl130), tmux 안이라고 가정한다. HF_TOKEN은 환경변수나 .env에 둔다(커밋 금지).
#
# 환경변수로 바꾼다 (기본값)
#   MODEL_TAG=dpo_dvmx1                       결과 폴더 이름: results/<태그>/<벤치마크>/run_NN
#   MODEL_PATH=results/dpo_dvmx1/merged       평가할 합친 모델
#   WITH_BASE=0                               1이면 base도 같은 조건으로 평가한다 (base는 루프로 오래 걸린다)
#   BASE_TAG=base_130
#   BASE_MODEL=Qwen/Qwen3-VL-4B-Instruct@ebb281ec70b05090aa6165b016eac8ec08e71b17
#   BENCHMARKS="mmmu_pro_standard10 mmmu_pro_vision mmmu_test"      (mmmu_pro_standard4도 가능)
#   RUNS_PRO=2                                MMMU-Pro는 seed 42, 43
#   RUNS_TEST=1                               MMMU test는 seed 42 한 번 (10,500문항이라 오래 걸린다)
#
# 시간 가늠 (최종 모델이 MMMU validation 900문항을 seed 한 번에 약 1.6시간에 끝낸 기록을 문항 수로 환산한 추정이다. 측정값이 아니다)
#   MMMU-Pro 한 설정(1,730문항): seed 한 번에 약 3시간 이상(vision은 큰 스크린샷이라 더 걸릴 수 있다)
#   MMMU test(10,500문항): seed 한 번에 약 19시간 이상
#   base는 32,768토큰까지 도는 루프가 많아 훨씬 오래 걸린다. 시간이 모자라면 최종 모델만 먼저 돌린다(WITH_BASE=0).

set -euo pipefail
cd "$(dirname "$0")/.."

MODE="${1:-plan}"
MODEL_TAG="${MODEL_TAG:-dpo_dvmx1}"
MODEL_PATH="${MODEL_PATH:-results/dpo_dvmx1/merged}"
WITH_BASE="${WITH_BASE:-0}"
BASE_TAG="${BASE_TAG:-base_130}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-VL-4B-Instruct@ebb281ec70b05090aa6165b016eac8ec08e71b17}"
BENCHMARKS="${BENCHMARKS:-mmmu_pro_standard10 mmmu_pro_vision mmmu_test}"
RUNS_PRO="${RUNS_PRO:-2}"
RUNS_TEST="${RUNS_TEST:-1}"

runs_for() {
  if [[ "$1" == "mmmu_test" ]]; then echo "$RUNS_TEST"; else echo "$RUNS_PRO"; fi
}

check_model() {
  if [[ ! -d "$MODEL_PATH" ]]; then
    echo "모델 폴더를 찾지 못했다: $MODEL_PATH" >&2
    echo "  먼저 학습과 병합을 끝내거나 MODEL_PATH를 지정하세요 (REPRODUCE.md 5, 6단계)." >&2
    exit 2
  fi
  if [[ ! -f "$MODEL_PATH/.merge_ok" ]]; then
    echo "경고: $MODEL_PATH 에 .merge_ok 표시가 없다 (합치다 끊겼거나 예전 방식으로 합쳤을 수 있다)." >&2
  fi
}

check_token() {
  if [[ -z "${HF_TOKEN:-}" && ! -f .env ]]; then
    echo "HF_TOKEN이 없다. Llama-3.1-8B judge는 gated 모델이라 필요하다 (export HF_TOKEN=... 또는 code/.env)." >&2
    exit 2
  fi
}

case "$MODE" in
  check)
    python lab/check_benchmarks.py --benchmarks $(for b in $BENCHMARKS; do echo "$b"; done | tr '\n' ' ')
    ;;

  smoke)
    check_model; check_token
    for b in $BENCHMARKS; do
      echo "=== smoke: $b (과목당 1문항)"
      python lab/eval_repeat.py --tag "${MODEL_TAG}_smoke" --model "$MODEL_PATH" --runs 1 --benchmark "$b" --limit 1
      python lab/summarize_runs.py summary "results/${MODEL_TAG}_smoke/$b"
    done
    echo "smoke 완료. 결과는 results/${MODEL_TAG}_smoke/ 에 있고 본 평가와 폴더가 달라서 섞이지 않는다."
    ;;

  plan|run)
    [[ "$MODE" == "run" ]] && { check_model; check_token; }
    for b in $BENCHMARKS; do
      n="$(runs_for "$b")"
      specs=("${MODEL_TAG}=${MODEL_PATH}:${n}")
      if [[ "$WITH_BASE" == "1" ]]; then specs+=("${BASE_TAG}=${BASE_MODEL}:${n}"); fi
      echo "=== $b (run ${n}개씩)"
      if [[ "$MODE" == "plan" ]]; then
        python lab/eval_queue.py --benchmark "$b" --dry-run "${specs[@]}"
      else
        python lab/eval_queue.py --benchmark "$b" "${specs[@]}"
      fi
    done
    if [[ "$MODE" == "run" ]]; then
      echo "모두 끝났다. 요약:"
      for b in $BENCHMARKS; do
        for t in "$MODEL_TAG" $([[ "$WITH_BASE" == "1" ]] && echo "$BASE_TAG"); do
          [[ -d "results/$t/$b" ]] && python lab/summarize_runs.py summary "results/$t/$b" || true
          # MMMU-Pro는 577문항이 MMMU validation에서 온 것이라(모델 선택에 쓴 데이터) 출처별로도 본다
          if [[ "$b" == mmmu_pro_* && -d "results/$t/$b" ]]; then python lab/pro_origin_split.py "results/$t/$b" || true; fi
        done
        if [[ "$WITH_BASE" == "1" && -d "results/$BASE_TAG/$b" && -d "results/$MODEL_TAG/$b" ]]; then
          python lab/summarize_runs.py compare "results/$BASE_TAG/$b" "results/$MODEL_TAG/$b" || true
        fi
      done
    fi
    ;;

  *)
    echo "사용법: bash lab/eval_extra.sh check|smoke|plan|run" >&2
    exit 2
    ;;
esac
