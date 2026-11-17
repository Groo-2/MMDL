#!/usr/bin/env bash
# conda/venv 활성화 도우미. `source lab/activate_env.sh`로 불러서 쓴다.
#
# 왜 필요한가: conda 환경의 activate.d 스크립트(예: ~cuda-nvcc_activate.sh)는 설정되지 않은 변수(NVCC_PREPEND_FLAGS)를
# 그대로 참조한다. 우리 스크립트는 set -u(미설정 변수 오류) 모드라서 `conda activate`에서 "unbound variable"로 중단됐다
# (2026-10-28, 이 실습실 mmdl 환경). 활성화하는 동안만 set -u를 끄고 원래 상태로 돌려놓는다.

lab_activate_conda() {
  local name="$1" had_u=0 rc=0
  [[ $- == *u* ]] && had_u=1
  set +u
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate "$name" || rc=$?
  [[ "$had_u" == 1 ]] && set -u
  return "$rc"
}

lab_activate_venv() {
  local dir="$1" had_u=0 rc=0
  [[ $- == *u* ]] && had_u=1
  set +u
  # shellcheck disable=SC1091
  source "$dir/bin/activate" || rc=$?
  [[ "$had_u" == 1 ]] && set -u
  return "$rc"
}
