#!/usr/bin/env bash
# 제출 후보 모델을 보관한다 (실습실에서 저장소 루트로 실행). 가중치는 저장소에 올리지 않고 보관 폴더에 체크섬과 함께 묶는다.
#
#   bash lab/archive_model.sh dpo_dv1
#   ARCHIVE_DIR=/mnt/d/archive NO_MERGED=1 bash lab/archive_model.sh dpo_dv1      # 합친 모델(약 9GB)은 빼고 어댑터만
#
# 보관 폴더: $ARCHIVE_DIR(기본 ~/mmdl/archive)/<TAG>_<날짜>/
#   adapter/         학습된 LoRA 어댑터 (adapter_model.safetensors, adapter_config.json)   <- 가장 중요하고 작다(약 0.5GB)
#   merged/          어댑터를 bf16 base에 합친 평가용 모델 (NO_MERGED=1이면 생략. 어댑터로 다시 만들 수 있다)
#   results/         평가 결과(예측, 점수, 요약, 서버 로그, 환경 스냅샷). 체크포인트·옵티마이저·이미지 캐시는 제외
#   data/            학습 데이터에서 파생된 파일(쌍, 선별 결과, 응답 샘플, 중복 제거 보고). 원본 이미지는 제외(변환기로 다시 만든다)
#   logs/            이 모델과 관련된 로그
#   env/             패키지 버전 고정 파일, git 커밋과 작업 트리 상태
#   MANIFEST.txt, SHA256SUMS   파일 목록과 체크섬. 마지막에 읽기 전용으로 바꾼다.
set -euo pipefail

TAG="${1:?사용법: bash lab/archive_model.sh <TAG>  (예: dpo_dv1)}"
ARCHIVE_DIR="${ARCHIVE_DIR:-$HOME/mmdl/archive}"
DEST="$ARCHIVE_DIR/${TAG}_$(date +%Y%m%d)"
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -e "$DEST" ]]; then
  echo "이미 있다: $DEST (같은 날 다시 만들려면 먼저 이름을 바꾸거나 지우세요. 읽기 전용이면 chmod -R u+w 후)" >&2
  exit 2
fi
mkdir -p "$DEST"/{adapter,results,data,logs,env}

copy() {   # copy <원본> <보관 폴더 안 하위 경로>: 있으면 복사, 없으면 알림
  if [[ -e "$1" ]]; then
    mkdir -p "$DEST/$(dirname "$2")"
    cp -a "$1" "$DEST/$2"
    echo "보관: $1"
  else
    echo "  (없음, 건너뜀) $1"
  fi
}

# 1) 어댑터
for f in adapter_model.safetensors adapter_config.json README.md; do copy "results/$TAG/final_lora/$f" "adapter/$f"; done

# 2) 합친 모델
if [[ "${NO_MERGED:-0}" != "1" ]]; then
  copy "results/$TAG/merged" "merged"
else
  echo "NO_MERGED=1: 합친 모델은 보관하지 않는다 (어댑터로 python music_ft/merge_lora.py --low-memory 로 다시 만든다)"
fi

# 3) 평가 결과 (체크포인트, 이미지 캐시, 합친 모델은 제외)
if [[ -d "results/$TAG" ]]; then
  while IFS= read -r -d '' f; do
    rel="${f#results/$TAG/}"
    case "$rel" in checkpoint-*|jpeg/*|merged/*|final_lora/*) continue ;; esac
    mkdir -p "$DEST/results/$TAG/$(dirname "$rel")"
    cp -a "$f" "$DEST/results/$TAG/$rel"
  done < <(find "results/$TAG" -type f -print0)
  echo "보관: results/$TAG (체크포인트, jpeg, merged 제외)"
fi
for extra in "results/${TAG}_loop4" "results/${TAG}_c5"; do [[ -d "$extra" ]] && { mkdir -p "$DEST/results"; cp -a "$extra" "$DEST/results/"; echo "보관: $extra"; }; done

# 4) 학습 데이터에서 파생된 파일 (DATA_FILES로 추가 지정 가능, 공백으로 구분)
DATA_DEFAULT="data/dpo/${TAG/dpo_dv1/dpo_dv}.jsonl data/dpo/${TAG/dpo_dv1/dpo_dv}.report.json \
 data/general/dv_pool/train.jsonl data/general/dv_pool/train.dedup.jsonl data/general/dv_pool/screen_results.jsonl \
 data/general/dv_pool/screen_samples.jsonl data/general/dv_pool/select_report.json data/general/dv_pool/dedupe_report.json \
 data/general/dv_pool/convert_report.json data/rft/rft_dv.jsonl data/rft/rft_dv.report.json"
for f in $DATA_DEFAULT ${DATA_FILES:-}; do copy "$f" "$f"; done

# 5) 로그
for f in logs/*"$TAG"* logs/build_dpo.log logs/build_dv.log logs/dedupe_dv.log logs/screen_dv.log logs/general_data_probe.log; do
  [[ -e "$f" ]] && copy "$f" "$f"
done

# 6) 환경과 코드 버전
copy requirements_lab.lock.txt env/requirements_lab.lock.txt
copy "results/$TAG/env_snapshot.json" env/train_env_snapshot.json
{
  echo "보관 시각: $(date '+%F %T')"; echo "호스트: $(hostname)"
  echo "git 커밋: $(git rev-parse HEAD)"; echo "git 브랜치: $(git rev-parse --abbrev-ref HEAD)"
  echo "git 태그: $(git tag --points-at HEAD | tr '\n' ' ')"
  echo "--- git status --short"; git status --short
  echo "--- 최근 커밋"; git log --oneline -12
  echo "--- GPU"; nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null || true
} > "$DEST/env/git_and_host.txt"
git diff > "$DEST/env/git_diff_uncommitted.patch" || true
pip freeze > "$DEST/env/pip_freeze_at_archive.txt" 2>/dev/null || true

# 7) 목록과 체크섬, 읽기 전용
SHA="$(command -v sha256sum || true)"; [[ -n "$SHA" ]] || SHA="shasum -a 256"      # Linux는 sha256sum, macOS는 shasum
( cd "$DEST" && find . -type f ! -name SHA256SUMS ! -name MANIFEST.txt -print0 | sort -z | xargs -0 $SHA > SHA256SUMS
  : > MANIFEST.txt
  while IFS= read -r -d '' f; do printf '%12s  %s\n' "$(wc -c < "$f" | tr -d ' ')" "$f" >> MANIFEST.txt
  done < <(find . -type f ! -name SHA256SUMS ! -name MANIFEST.txt -print0 | sort -z) )
chmod -R a-w "$DEST"
echo
echo "완료: $DEST"
du -sh "$DEST" 2>/dev/null | cut -f1 | sed 's/^/총 크기: /'
echo "무결성 확인: (cd $DEST && $SHA -c SHA256SUMS | grep -v ': OK' || echo 모두 OK)"
