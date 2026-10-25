# 실습실(RTX 4090 24GB, CUDA 12.x) 실행 가이드

baseline(Run B 설정) 8회 → SSMR GRPO 학습 → 학습 모델 8회 평가 → 비교까지 전부 실습실에서 돌립니다.
평가 코드(`mmmu_eval/`)는 수정하지 않았고, 프롬프트·sampling·파서·judge·`max_new_tokens`·해상도는 baseline과 같습니다.

## 0. 준비

| 항목 | 내용 |
|---|---|
| 드라이버 | `nvidia-smi` 맨 위 **CUDA Version이 12.9 이상**이어야 합니다 (vLLM 0.27.1의 CUDA 12 빌드가 `cu129`뿐). 12.9 미만이면 `setup_env.sh`가 이유를 알려 주고 멈춥니다 |
| Python | 3.11 이상 (TRL 1.16). `PYTHON=python3.12 bash lab/setup_env.sh`처럼 지정 가능 |
| 디스크 | 약 60GB (Qwen 9GB + Llama 16GB + 데이터 + 체크포인트와 결과) |
| HF 토큰 | `meta-llama/Llama-3.1-8B-Instruct` 라이선스 동의 + 저장소 루트 `.env`에 `HF_TOKEN=hf_...` (커밋하지 말 것) |

```bash
git clone -b ssmr https://github.com/Groo-2/MMDL.git && cd MMDL
bash lab/setup_env.sh        # .venv_lab 생성 + 환경 점검 (conv3d/cuDNN 포함)
```

## 1. 실행 순서

```bash
tmux new -s mmdl
bash lab/run_pipeline.sh smoke        # ① 환경 + 속도 확인: Music 30문항 1회 (추론 + Llama judge)
bash lab/run_pipeline.sh data pilot   # ② SSMR 변환/중복 제거 + GRPO 2스텝 (스텝 시간 측정)
bash lab/run_pipeline.sh baseline     # ③ Run B 설정 × 8회 (seed 42~49)
bash lab/run_pipeline.sh train merge trained report   # ④ 학습 → merge → 학습 모델 × 8회 → 비교
```
`all`을 주면 data → pilot → baseline → train → merge → trained → report를 한 번에 돌립니다.
GPU가 하나라 단계는 순차로 실행됩니다. 끊기면 같은 커맨드를 다시 실행하면 이어서 합니다.

**①과 ②를 먼저 돌려서 시간을 확인한 뒤** ③④를 거세요. 아래 시간은 모두 **측정 전 추정**입니다.

| 단계 | 추정 | 근거 |
|---|---|---|
| baseline 1회 (900문항) | 6~12시간 | A100(40GB)에서 232분. 4090은 KV cache가 약 1/2~1/3이라 동시 처리량이 줄어듦 → ①의 Music 시간 × 30으로 환산 |
| baseline 8회 | 2~4일 | 위 × 8 |
| GRPO 1스텝 (8문항 × 8) | 20분 안팎 | Colab A100에서 학습 구간만 760초 (64 micro-batch). 4090은 더 느리고 rollout도 HF generate |
| GRPO 100스텝 | 약 1.5일 | 위 × 100. 스텝 수는 ②의 실측으로 정함 |

## 2. 무엇이 Colab과 다른가

- **vLLM**: CUDA 12.9 빌드(`+cu129`) + torch 2.13.0(cu129). 새 가상환경이라 Colab에서 문제였던 cuDNN cu12/cu13 혼합은 없어야 하지만, `check_env.py`가 conv3d로 직접 확인합니다.
- **평가**: Qwen 서버와 Llama judge 서버를 **순차로** 띄웁니다 (bf16 9GB + 16GB는 24GB에 동시에 안 들어감). Qwen은 8회를 한 서버로 연속 추론하고, judge는 8회분을 한 서버로 일괄 판정합니다.
- **학습 rollout**: vLLM colocate 대신 **HF generate**를 씁니다. 24GB에서 학습 모델(9GB) + 옵티마이저(2GB) + vLLM 사본(9GB)을 동시에 올리면 KV cache가 1GB 안팎만 남기 때문입니다. 대신 느립니다. A100 이상이면 `--rollout vllm`으로 바꿀 수 있습니다.
- **학습 rollout sampling**: temperature 0.7, top_p 0.8, top_k 20. **presence_penalty 1.5는 rollout에서 뺐습니다** (HF generate가 지원하지 않음). 평가는 동결 recipe를 그대로 씁니다 (presence_penalty 포함).
- **micro-batch 1**: 24GB에서 이미지 5장짜리 프롬프트(약 5.5k토큰)를 안전하게 처리합니다. 배치가 1이라 패딩이 없어서 Colab에서 겪은 이미지 위치 오류도 생기지 않습니다.

## 3. 반복 평가 방식

- run `k`의 seed = `42 + k`. **run 0은 동결 recipe(seed 42) 그대로**입니다. 요청에 실리는 seed만 바뀝니다.
- 결과: `results/<tag>/run_00 ~ run_07/` (각각 `final_predictions.jsonl`, `summary.json` 등)과 `results/<tag>/summary_runs.json`.
- 비교(`report`): 문항별 평균 정답률(Avg@8)의 차이를 **문항 단위 paired bootstrap 95% 신뢰구간**으로 보여 줍니다. 과목당 30문항이라 과목별 차이는 노이즈가 큽니다.
- **비교 대상은 실습실 baseline(`results/baseline`)입니다.** Colab A100의 Run B(563/900)와는 GPU와 CUDA 빌드가 달라서 수치가 똑같이 재현되지는 않습니다.

## 4. 확인되지 않은 것 (실습실에서 처음 해 보는 것)

- `+cu129` wheel 설치와 4090에서의 vLLM 서버 기동 (`smoke`가 확인)
- HF rollout + 이미지 5장 프롬프트가 24GB에서 메모리에 들어가는지 (`pilot`이 확인). 부족하면 `--max-completion-length`를 줄이거나 알려 주세요
- 학습이 실제로 Music을 올리는지. 예산(스텝 수)이 논문의 1~3% 수준이라 효과는 장담할 수 없습니다
