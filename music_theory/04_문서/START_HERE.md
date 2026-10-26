# 파일 확인 순서

1. 최상위 `README.md`: 데이터 구성과 baseline 적용 기준
2. `04_문서/RESEARCH.md`: 제작 방식, 논문 근거와 데이터 분리
3. `04_문서/PIPELINE.md`: 환경 준비와 실제 실행 순서

학습과 병합은 `01_학습용/`, vLLM 평가와 비교는 `02_평가용/`에서 제공합니다.
학습용 Transformers 환경과 baseline vLLM 평가 환경을 분리해 사용합니다.
저장소의 `baseline/`과 `music_theory/`는 같은 상위 폴더에 유지합니다.

학습 입력과 정답은 baseline의 `mmmu_pro_cot`에 맞춰 설명 뒤 마지막 줄에 `Answer:`가 오도록 구성했습니다.
최종 MMMU 평가는 병합 모델을 기존 `baseline/scripts/run_mmmu_eval.sh`에 연결해 실행합니다.
