# Music 해상도 실험 — 실습실 RTX 4090 실행 방법

노트북: `experiments/music_resolution_4090.ipynb`
MMMU validation Music 30문항을 이미지 토큰 441 / 784 / 1,568 / 3,920개로 바꿔 다시 풀고, baseline(Run B)과 비교한다.

## 0. 요구 사항
| 항목 | 조건 |
|---|---|
| OS | Linux 또는 Windows **WSL2(Ubuntu)**. vLLM은 Windows에서 바로 실행되지 않음 |
| GPU | RTX 4090 24GB, 다른 프로세스가 GPU를 쓰지 않는 상태 |
| 드라이버 | `nvidia-smi`의 **CUDA Version 13.0 이상** (vLLM 0.27.1이 CUDA 13.0용 torch를 설치함) |
| 디스크 | 여유 약 40GB (Qwen 약 9GB + Llama judge 약 16GB + vLLM/torch 패키지 + MMMU Music). 기본 설정(`USE_JUDGE = False`)에서는 judge를 받지 않으므로 약 25GB |
| HF 계정 | judge를 켤 때만(`USE_JUDGE = True`) `meta-llama/Llama-3.1-8B-Instruct` 라이선스 동의 + Read 토큰 |

## 1. 코드 받기
```bash
git clone -b music_resolution https://github.com/Groo-2/MMDL.git
cd MMDL
```

## 2. 파이썬 환경 만들기
conda 예시 (venv나 uv도 상관없음):
```bash
conda create -n mmdl python=3.12 -y
conda activate mmdl
pip install -r baseline/requirements.txt jupyter
pip uninstall -y torchaudio     # 설치돼 있으면 vLLM의 torch와 CUDA 버전이 달라 서버가 뜨지 않음
python -c "import torch, vllm; print(torch.__version__, torch.cuda.is_available(), vllm.__version__)"
```
마지막 줄에 `True`와 `0.27.1`이 나오면 정상.

## 3. HF 토큰 설정
.env 파일에서 관리

## 4. 실행
실행 시간이 길어서(약 2~4시간) 연결이 끊겨도 계속 돌도록 **tmux 안에서 실행하는 것을 권장**한다.
```bash
tmux new -s music
conda activate mmdl
cd MMDL
jupyter nbconvert --to notebook --execute experiments/music_resolution_4090.ipynb \
  --output music_resolution_4090_run.ipynb --ExecutePreprocessor.timeout=-1
# 화면에서 빠져나오기: Ctrl+b 다음 d / 다시 들어가기: tmux attach -t music
```
- 결과가 담긴 노트북: `experiments/music_resolution_4090_run.ipynb` (5번 셀에 비교표)
- 화면을 보며 셀 단위로 돌리고 싶으면 `jupyter lab`으로 열거나 VS Code에서 열어 1번 셀부터 실행
- 중간에 멈춰도 같은 명령을 다시 실행하면 끝난 문항은 건너뜀

## 5. 결과 위치
```
results/music_resolution/
├── music_tok_441/   music_tok_784/   music_tok_1568/   music_tok_3920/
│   ├── final_predictions.jsonl   # 문항별 응답, 정오
│   ├── summary.json              # 점수, 잘림, 사용한 해상도
│   ├── resolution.json           # 단계 설정, GPU, 소요 시간
│   └── vllm_infer.log
└── vllm_judge.log
```
결과 폴더는 git에 커밋하지 말 것(대용량, 모델 출력 원문 포함).

## 6. 문제가 생기면
| 증상 | 대처 |
|---|---|
| `Engine core initialization failed` | 노트북이 로그의 "오류로 보이는 줄"을 출력함. `nvidia-smi`로 다른 프로세스가 GPU를 쓰는지 확인 |
| 3,920 단계에서 `max seq len ... larger than ... KV cache` | 노트북 1번 셀의 `GPU_MEM_UTIL`을 0.93~0.95로 올려 다시 실행 |
| `CUDA driver version is insufficient` | 드라이버가 CUDA 13.0 미만. 실습실 관리자에게 드라이버 업데이트 요청 |
| 비교표의 "오류" 수가 0이 아님 | 해당 단계 `max_model_len`이 부족. 1번 셀 `LEVELS` 값을 4096 단위로 올려 그 단계만 다시 실행(`RUN = [단계]`) |
