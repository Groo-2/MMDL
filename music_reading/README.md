# 악보 판독 평가 (pilot20)

Qwen3-VL-4B-Instruct가 악보 이미지를 읽는지 20문항으로 평가한다. **팀 baseline(Run B) 파이프라인을 그대로** 쓴다.
**pilot20은 평가 전용 데이터다. 학습, 파인튜닝, few-shot 예시, 프롬프트 튜닝에 사용하지 않는다.**

```
music_reading/
├── run_pilot20.sh             # 단일 진입점: 모델 서버 → infer → judge 서버 → judge → score
├── run_pilot20.py             # baseline 코드를 import해서 pilot20에 적용 (infer / judge / score)
├── scoring.py                 # pilot20 전용 규칙 기반 채점 (참고용 열, baseline 채점 아님)
├── music_reading_eval.ipynb   # 옛 노트북. baseline과 다른 설정이라 baseline 평가에는 쓰지 않는다
└── data/pilot20/              # pilot_test.jsonl + images/ (20문항)
```

## 실습실에서 실행
```bash
git clone -b music_reading https://github.com/Groo-2/MMDL.git
cd MMDL

conda create -n mmdl python=3.12 -y
conda activate mmdl
pip install vllm==0.27.1 requests Pillow
pip uninstall -y torchaudio
python -c "import torch, vllm; print(torch.__version__, torch.cuda.is_available(), vllm.__version__)"

export HF_TOKEN=...   # judge(Llama-3.1-8B-Instruct)가 gated 모델이라 필요 (HF에서 라이선스 동의 먼저)

tmux new -s music
conda activate mmdl
cd ~/MMDL/music_reading
bash run_pilot20.sh \
  --model_path Qwen/Qwen3-VL-4B-Instruct \
  --output_dir results/pilot20_baseline
```
- 스모크 테스트: `--limit 3` (앞 3문항). **평가 수치로 쓸 수 없음**
- 중간에 끊기면 **같은 커맨드를 다시 실행**하면 된다. 끝난 문항은 건너뛴다(resume)
- judge 없이: `--no_judge` (rule 파서가 실패한 문항은 오답 처리)
- fine-tuning 후 재평가: `--model_path /path/to/ckpt`로 바꾸고 `--model_revision`은 생략. 나머지 설정은 그대로다
- 같은 `--output_dir`에 생성 설정(temperature, presence_penalty)을 바꿔 이어서 돌리면 오류로 막는다(결과가 섞이지 않도록)

## 산출물 (`--output_dir`, git에 올리지 않음)
| 파일 | 내용 |
|---|---|
| `summary.json` | 전체·rule만의 정답 수, 카테고리별 점수, 파서 경로 분포, 잘림 비율, 소요 시간, peak VRAM, 모든 설정값 |
| `scores.csv` | 문항별 정답, 최종 예측, 파서 경로, 참고 채점(`reference_strict/lenient`), 응답 원문 |
| `final_predictions.jsonl`, `failed_parses.jsonl` | 문항별 전체 기록, rule 파서가 실패한 문항 |
| `predictions.jsonl`, `judge_results.jsonl`, `sessions.json` | 원시 기록 (resume용) |
| `vllm_*.log`, `gpu_mem_*.log` | 서버 로그, VRAM 기록 |

## baseline과 같은 것 / 다른 것
`baseline/` 폴더의 파일은 **수정하지 않고** import해서 쓴다 (프롬프트, 요청 payload, rule 파서, judge 코드, vLLM 서버 옵션 값).

**같은 것**: 모델과 revision(`ebb281ec…`), bf16, 프롬프트 `mmmu_pro_cot`(주관식 문구), top_p 0.8 / top_k 20 / repetition_penalty 1.0 / seed 42,
`max_new_tokens` 32768, `max_model_len` 40960, 이미지 해상도와 JPEG q95, 파서(`Answer:` 줄 → Llama judge → 오답), judge 모델과 설정.

**다른 것 (2가지)**
1. **데이터**: baseline 로더는 MMMU 30과목 전용이라 pilot20을 읽을 수 없다. 또 baseline은 질문에 `<image N>` 표시가 없는 이미지를 입력에서 버리는데, pilot20 질문에는 표시가 없어서 그대로 두면 **모델이 이미지 없이 답한다**. 그래서 질문 앞에 `<image 1> `을 붙여 baseline 방식(`[image]`로 치환, 이미지는 텍스트 뒤)이 그대로 적용되게 했다.
2. **순수 greedy** (재현성용): `temperature` 0 (baseline 0.7), `presence_penalty` 0 (baseline 1.5). temperature 0만으로는 penalty가 그대로 적용돼 순수 greedy가 아니라서 두 값을 함께 바꿨다. `baseline/`의 상수를 고치면 MMMU baseline 재현이 깨지므로 실행 시점에만 덮어쓴다. `repetition_penalty`는 baseline 값 1.0이 이미 penalty 없음이고, temperature 0에서는 top_p/top_k가 결과에 영향이 없어서 baseline 값을 그대로 둔다. `--temperature`, `--presence_penalty`로 바꿀 수 있다.

## 채점 참고
baseline 채점(`correct`)은 최종 답이 정답과 정규화 후 **완전 일치**해야 정답이다. 그래서 문장형 답("The treble clef")이나 동의어("G clef")는 오답이다.
박자표는 분수값으로 비교해서 **3/4와 6/8을 같게 본다**(S03 1문항). `scores.csv`의 `reference_strict`, `reference_lenient`(scoring.py)와 함께 보면 이런 차이를 확인할 수 있다. 주 지표는 baseline 채점이다.
20문항이라 1문항이 5%p다. 카테고리별 수치는 참고용으로만 본다.
