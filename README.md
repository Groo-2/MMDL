# MMDL

Qwen3-VL-4B-Instruct MMMU validation baseline 평가 (과제 1).

- [`reports/mmmu_baseline.md`](reports/mmmu_baseline.md): 제출 보고서
- [`baseline/`](baseline/): 평가 파이프라인. 실행 방법과 설계 선택의 근거는 [`baseline/README_mmmu_eval.md`](baseline/README_mmmu_eval.md), baseline 결정 과정은 [`baseline/reports/baseline_decision.md`](baseline/reports/baseline_decision.md)

한 커맨드 재현 (모든 명령은 `baseline/` 폴더에서 실행):

```bash
cd baseline
pip install -r requirements.txt
export HF_TOKEN=...   # judge(Llama-3.1-8B-Instruct)가 gated 모델이라 필요

bash scripts/run_mmmu_eval.sh \
  --model_path Qwen/Qwen3-VL-4B-Instruct \
  --model_revision ebb281ec70b05090aa6165b016eac8ec08e71b17 \
  --data_root <MMMU HF 캐시 디렉터리> \
  --output_dir ./results/baseline_mmmu_pro_cot
```

다른 브랜치: `qwen-mmmu-62-44`(처음 평가한 GitHub 원본 스크립트와 응답), `baseline_final`(`baseline/` 폴더로 옮기기 전 구조의 파이프라인 스냅샷).
