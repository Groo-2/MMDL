# 실습실(RTX 4090 24GB, CUDA 12.x): E0 baseline + E1 SSMR GRPO

- **E0:** 실습실 환경에서 baseline 재측정
- **E1:** SSMR-Bench Train_visual로 GRPO(rollout 8, KL 없음, 정답 보상, LoRA) → merge → 평가
- E2(music_theory SFT)와 E3는 팀원이 버전을 맞춘 자료를 다시 받은 뒤 진행합니다 (이 폴더에는 아직 없음).

**처음이면 `lab/RUNBOOK_실습실_명령어.md`부터 보세요** (실습실에서 칠 명령어를 순서대로, 단계마다 확인할 것과 알려 줄 것을 적어 두었습니다).

평가 코드(`mmmu_eval/`)는 수정하지 않았고, 프롬프트·sampling·파서·Llama judge·`max_new_tokens`(32,768)·해상도는 baseline(Run B)과 같습니다. 바뀌는 것은 모델 경로와 요청의 seed뿐입니다.

## 0. 준비

| 항목 | 내용 |
|---|---|
| 드라이버 | `nvidia-smi` 맨 위 **CUDA Version이 12 이상**이면 됩니다. 13.x면 vLLM 기본 wheel, 12.x면 `cu129` wheel을 씁니다. **12.9보다 낮아도(예: 12.6) CUDA 마이너 버전 호환으로 실행**되고, 이 실습실 PC(WSL2, 드라이버 560.94 = CUDA 12.6)에서 `torch 2.13.0+cu129`와 `vllm 0.27.1+cu129`로 평가를 실행한 이력이 있습니다 (2026-10-14 music_resolution, 10-20 pilot20). 12 미만이면 `setup_env.sh`가 멈춥니다 |
| WSL2 | 이 PC는 WSL2입니다. GPU 드라이버는 **Windows 쪽**이라 Linux 안에서 드라이버를 설치하지 않습니다 (올려야 하면 Windows의 NVIDIA 드라이버 업데이트). Windows 앱이 VRAM을 쓰면 vLLM이 안 뜰 수 있어서(`preflight`가 확인) 긴 작업 중에는 GPU를 쓰는 Windows 앱을 닫고, 절전과 자동 재시작을 끄고, tmux를 붙여 둔 터미널 창을 닫지 마세요 |
| Python | 3.11 이상 (`PYTHON=python3.12 bash lab/setup_env.sh`) |
| 디스크 | 약 60GB (Qwen 9GB + Llama 16GB + 데이터 + 체크포인트·결과) |
| HF 토큰 | `meta-llama/Llama-3.1-8B-Instruct` 라이선스 동의 + 저장소 루트 `.env`에 `HF_TOKEN=hf_...` (커밋 금지) |

```bash
git clone -b ssmr https://github.com/Groo-2/MMDL.git && cd MMDL
bash lab/setup_env.sh          # .venv_lab 생성, lock(requirements_lab.lock.txt) 저장, 환경 점검
echo 'HF_TOKEN=hf_...' > .env
```

## 1. 실행 순서

```bash
tmux new -s mmdl
bash lab/run_pipeline.sh env vllm_check smoke  # ① 환경 점검 + vLLM 서버 기동 확인 + Music 30문항 1회 (시간/동작 확인)
bash lab/run_pipeline.sh data screen stress     # ② SSMR 변환, 통과율 측정+1,120문항 선별, GRPO 최악 조건 1스텝
bash lab/run_pipeline.sh pilot                  # ③ GRPO 2스텝 (스텝 시간 실측)
bash lab/run_pipeline.sh baseline_music4 holdout_base baseline_full2     # ④ E0
bash lab/run_pipeline.sh grpo merge holdout_new music4 full2 report      # ⑤ E1
```
`all`은 data → screen → stress → pilot → baseline_music4 → holdout_base → baseline_full2 → grpo → merge → holdout_new → music4 → full2 → report를 한 번에 돌립니다. 끊기면 같은 커맨드를 다시 실행하면 이어서 합니다. `DRY_RUN=1`을 붙이면 실행하지 않고 명령만 출력합니다.

**①~③을 먼저 하세요.** 시간은 전부 **측정 전 추정**이고, ①~③이 실측치를 줍니다.

| 단계 | 추정 | 비고 |
|---|---|---|
| `smoke` (Music 1회) | 0.5~1시간 | 이 값 × 4 = `music4`, ÷30 × 900 = `full` 1회의 대략치 |
| `screen` | 2~3시간 | 후보 3,000문항 × 8번 풀이 |
| `stress` | 수 분 | 이미지 5장 프롬프트로 24GB에 들어가는지 확인 |
| GRPO 140스텝 | 약 2일 | 1스텝 약 20분 가정. `pilot`의 스텝 시간으로 `STEPS`를 조정 (`STEPS=100 bash lab/run_pipeline.sh grpo`) |
| `music4` | 2~4시간 | Music 30문항 × seed 42~45 |
| `full2` | 12~24시간 | 900문항 × seed 42, 43 |

## 2. 평가 설계

| 평가 | 반복 | 목적 |
|---|---|---|
| `music4` | Music 30문항 × seed 42~45 | Music 개선 확인 |
| `full2` | 900문항 × seed 42, 43 | **전체 점수** (이 과제는 과목별로 떨어져도 전체가 오르면 됨). seed 42가 공식 수치, 43은 노이즈 가늠 |
| `holdout` | SSMR hold-out 400문항 | GRPO가 SSMR 자체를 배웠는지, 모델 선택의 1차 신호 (문항이 많아 Music 30문항보다 믿을 만함) |

- run `k`의 seed = 42 + k. 같은 seed로 반복하면 같은 응답이 나오므로 반복에는 서로 다른 seed를 씁니다. seed 목록은 모든 모델에 똑같이 씁니다.
- 결과: `results/<tag>/music/run_00~03`, `results/<tag>/full/run_00~01`, `results/<tag>/holdout/seed_42`. 각 폴더에 `env_snapshot.json`(GPU·드라이버·CUDA·`pip freeze`)이 저장됩니다.
- 비교는 문항 단위 paired bootstrap 95% 신뢰구간입니다. Music 30문항은 어떤 반복 수로도 작은 효과를 확정하기 어렵습니다 (논문의 +2문항은 8번으로도 겨우 검출되는 크기).
- 비교 대상은 **실습실 baseline(`results/baseline`)** 입니다. Colab A100의 Run B(563/900)와는 GPU·CUDA 빌드가 달라 수치가 똑같이 재현되지 않습니다.

### Music 8번 확장 (사용자 확인 필수)
`music4` 결과가 애매하거나 향상이 보일 때만 seed 46~49를 같은 폴더에 추가합니다. **확인 없이는 실행되지 않습니다** (환경변수가 없으면 즉시 중단, `all`에도 포함되지 않음).
```bash
CONFIRM_EXTEND_TO_8=yes bash lab/run_pipeline.sh music_extend_base music_extend_new
```

## 3. 에러가 났을 때 (순서 중요)

**코드를 고치기 전에 CUDA·패키지 버전부터 확인합니다.** 파이프라인은 단계가 실패하면 자동으로 아래 점검을 먼저 실행합니다.

```bash
python lab/env_triage.py     # logs/env_triage_<시각>.txt 저장
```
점검 항목: ① 드라이버 CUDA가 torch CUDA 이상인지 (같은 12.x에서 낮으면 마이너 버전 호환이라 경고만) ② `nvidia-*-cu12`와 `-cu13` wheel 혼재 (**버전이 다르면 FAIL**: Colab에서 `CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH`의 원인. **같은 버전이면 경고**: 이 실습실 mmdl 환경이 이 상태이고 conv3d 스모크가 통과함) ③ torch/vllm/trl/transformers 버전이 기대값(torch 2.13.0, vllm 0.27.1, trl 1.16.0, transformers ≥ 5.5.3)과 lock에 맞는지 ④ `pip check` ⑤ bf16 matmul, conv3d 스모크.

FAIL이 없는데도 실패하면 그때 에러 로그 자체를 봅니다. 긴 작업 전에는 `python lab/preflight.py --stage eval|screen|train|holdout`이 디스크, GPU 점유, 포트, HF 토큰, 파일을 점검합니다 (파이프라인이 단계마다 먼저 실행).

| 증상 | 확인/대응 |
|---|---|
| `stress`에서 CUDA out of memory | GRPO prefill(이미지 5장 × completion 8개 KV) 메모리. 대응 순서: `MAX_COMPLETION=1536` → 5이미지 문항 비중 축소. 해상도는 평가와 같게 유지 |
| vLLM 서버가 안 뜸 (`kv cache`, `max seq len`) | 로그의 해당 줄을 자동 출력합니다. 24GB에서 `max_model_len` 40,960 + util 0.90은 한 요청이 들어가는 크기(추정) |
| 포트 8000/8001 사용 중 | 이전 서버가 남음: `pkill -f "vllm serve"` |
| Llama 접근 거부 | 라이선스 동의, `HF_TOKEN` 확인. judge 없이 돌리려면 `--no-judge` |
| 학습 중 `grad_norm 0` | 로그의 `reward_std`, `frac_reward_zero_std` 확인 (통과율 필터로 학습 신호가 있는 문항만 쓰도록 선별했음) |

## 4. 알려진 한계 (실습실에서 처음 해 보는 것)
- Mac에는 GPU/vLLM이 없어서 **드라이버·CUDA 빌드, vLLM 서버 기동, GPU 메모리**는 실습실에서만 확인됩니다. Mac에서 검증한 것: 설정·선별 로직, 요청 형식과 채점(가짜 서버), merge 정확성, 안전장치, 파이프라인 단계 순서 (`python -m unittest lab.test_lab`).
- 학습 rollout은 HF generate(24GB에서 vLLM colocate 불가)라 느리고, presence_penalty 1.5를 쓰지 못합니다 (평가는 그대로 사용).
- GRPO 예산은 논문의 약 3%입니다. 효과는 장담할 수 없고, 학습 seed가 하나(42)라 학습 난수의 영향은 분리하지 못합니다.
