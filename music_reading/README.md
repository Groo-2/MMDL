# 악보 판독 평가 (pilot20)

Qwen3-VL-4B-Instruct가 악보 이미지를 읽는지 20문항으로 평가한다.
**pilot20은 평가 전용 데이터다. 학습, 파인튜닝, few-shot 예시, 프롬프트 튜닝에 사용하지 않는다.**

```
music_reading/
├── music_reading_eval.ipynb   # 실행 노트북 (서버 실행 → 추론 → 채점)
├── scoring.py                 # 규칙 기반 채점
└── data/pilot20/              # pilot_test.jsonl + images/ (20문항)
```

## 실습실에서 실행
```bash
git clone -b music_reading https://github.com/Groo-2/MMDL.git
cd MMDL/music_reading

conda create -n mmdl python=3.12 -y
conda activate mmdl
pip install vllm==0.27.1 requests pandas Pillow jupyter
pip uninstall -y torchaudio
python -c "import torch, vllm; print(torch.__version__, torch.cuda.is_available(), vllm.__version__)"

tmux new -s music
conda activate mmdl
cd ~/MMDL/music_reading
jupyter nbconvert --to notebook --execute music_reading_eval.ipynb \
  --output music_reading_eval_run.ipynb --ExecutePreprocessor.timeout=-1
```
- 결과: `results/pilot20/` (`predictions.jsonl`, `scores.csv`, `vllm_infer.log`), 요약은 `music_reading_eval_run.ipynb`의 마지막 셀
- Qwen3-VL은 공개 모델이라 HF 토큰이 필요 없다.
- CoT로 다시 풀려면 노트북 1번 셀에서 `USE_COT = True`, `MAX_NEW_TOKENS = 4096`.
