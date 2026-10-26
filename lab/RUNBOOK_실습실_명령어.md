# 실습실 명령어 단계별 안내 (E0 baseline + E1 SSMR GRPO)

위에서 아래로 순서대로 치면 됩니다. 각 단계 끝에 **"확인할 것"**과 **"저에게 알려 줄 것"**이 있습니다. 시간은 모두 **측정 전 추정**입니다.
원리와 설계 이유는 `lab/README_lab.md`에 있습니다.

> 규칙 2가지
> 1. **GPU는 한 번에 한 작업만** 씁니다. 한 단계가 끝난 뒤에 다음 단계를 시작하세요 (`nvidia-smi`로 비었는지 확인).
> 2. **에러가 나면 코드를 고치기 전에** CUDA/패키지 점검부터 합니다 (파이프라인이 자동으로 실행해 줍니다. 7번 참고).

---

## 0. 시작 전 확인 (1분)

```bash
nvidia-smi                 # 맨 위 오른쪽 "CUDA Version: 12.9" 이상이어야 합니다
python3 --version          # 3.11 이상
df -h .                    # 여유 공간 60GB 이상
tmux -V                    # 없으면 설치를 요청하세요
```
- `CUDA Version`이 **12.9 미만**이면 3번(설치)에서 멈춥니다. 이 경우 바로 알려 주세요 (드라이버를 올려야 합니다).
- Llama 모델은 승인이 필요합니다: 브라우저에서 https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct 에 로그인해 **라이선스에 동의**하고, https://huggingface.co/settings/tokens 에서 **Read 토큰**을 만들어 두세요.

## 1. 코드 받기

```bash
git clone -b ssmr https://github.com/Groo-2/MMDL.git
cd MMDL
```
이후 모든 명령은 `MMDL` 폴더(저장소 루트)에서 칩니다. 나중에 코드가 업데이트되면 `git pull`만 하면 됩니다.

## 2. HF 토큰 넣기

```bash
echo 'HF_TOKEN=hf_여기에_토큰' > .env
```
`.env`는 `.gitignore`에 들어 있어 커밋되지 않습니다. 파일 내용을 다른 곳에 붙여 넣지 마세요.

## 3. 환경 설치 (10~30분)

```bash
bash lab/setup_env.sh
```
- `.venv_lab` 가상환경을 만들고 torch 2.13.0(cu129) + vLLM 0.27.1 + trl 1.16.0 등을 설치한 뒤, 마지막에 **환경 점검**을 자동 실행합니다.
- 마지막 줄이 `결과: FAIL 0개 -> 환경은 정상으로 보입니다`이면 성공입니다.
- `requirements_lab.lock.txt`가 생깁니다 (보고서의 패키지 버전 기록).

**알려 줄 것:** 실패하면 마지막 40줄 (`bash lab/setup_env.sh 2>&1 | tail -40`).

## 4. tmux 시작 (연결이 끊겨도 계속 돌게)

```bash
tmux new -s mmdl
```
- 나가기(작업은 계속 실행): `Ctrl+b` 누른 뒤 `d`
- 다시 들어가기: `tmux attach -t mmdl`

앞으로의 모든 `bash lab/run_pipeline.sh ...`는 **tmux 안에서** 실행하세요.

---

## 5. 단계별 실행

### 5-1. 환경 점검 + 동작·속도 확인 (약 1시간)

```bash
bash lab/run_pipeline.sh env smoke
```
- `env`: CUDA·패키지 점검. `smoke`: Music 30문항을 1번 풀고 Llama judge로 채점합니다 (서버가 뜨는지, judge가 도는지까지 확인).
- 확인할 것: 마지막에 `Music Avg@1: ...` 같은 요약이 나오고 에러가 없을 것. 기준 baseline(Run B)은 5/30(0.167)이며, 샘플링 때문에 달라질 수 있습니다.
- 소요 시간을 적어 두세요 (이 값 × 4 ≈ `music4`).

**알려 줄 것:** `logs/smoke.log` 마지막 30줄과 걸린 시간.
```bash
tail -30 logs/smoke.log
```

### 5-2. SSMR 데이터 준비 + 학습 문항 선별 + 메모리 시험 (약 3~4시간)

```bash
bash lab/run_pipeline.sh data screen stress
```
- `data`: SSMR Train_visual 8,000문항을 변환하고 MMMU와 겹치는지 검사합니다 (겹침 0건이어야 함).
- `screen`: base 모델로 후보 3,000문항을 8번씩 풀어 **통과율 1/8~7/8인 1,120문항**을 고릅니다 (학습 신호가 있는 문항만). 2~3시간.
- `stress`: 가장 긴 프롬프트(이미지 5장)로 GRPO를 1스텝 돌려 **24GB에 들어가는지** 미리 확인합니다.
- 확인할 것:
  ```bash
  cat data/ssmr/select_report.json       # chosen_per_category가 280씩이면 정상, notes에 부족 안내가 있으면 알려 주세요
  grep "heartbeat" logs/stress.log       # "step 1 완료, 스텝 시간 ...s, GPU 최대 메모리 ..GB"가 나오면 성공
  ```
- `stress`에서 `CUDA out of memory`가 나오면 **멈추고 알려 주세요** (대응 방법을 정해 드립니다).

**알려 줄 것:** `select_report.json` 내용, `stress`의 `heartbeat` 줄 (스텝 시간과 GPU 최대 메모리).

### 5-3. GRPO 파일럿 2스텝 (약 40분~1시간)

```bash
bash lab/run_pipeline.sh pilot
```
- 스텝당 시간을 재서 본 학습의 스텝 수를 정합니다.
- 확인할 것:
  ```bash
  grep "heartbeat\|grad_norm\|reward" logs/pilot.log | tail -20
  ```
  - `step 2 완료, 스텝 시간 NNNs`가 나오면 성공.
  - 학습 로그의 `grad_norm`이 **0이 아니어야** 합니다 (0이면 학습이 안 되는 것).

**알려 줄 것:** 스텝 시간(NNNs)과 `grad_norm`, `reward` 값.
→ 스텝 수는 이렇게 정합니다: 스텝 수 = (쓸 수 있는 시간 × 3600) ÷ 스텝 시간. 약 2일(48시간)이고 스텝 시간이 1,200초(20분)면 144스텝 → 기본값 140입니다.

### 5-4. E0 baseline 평가 (약 15~30시간)

```bash
bash lab/run_pipeline.sh baseline_music4 holdout_base baseline_full2
```
- `baseline_music4`: Music 30문항 × seed 42~45 (2~4시간)
- `holdout_base`: SSMR hold-out 400문항 (1시간 안팎)
- `baseline_full2`: 전체 900문항 × seed 42, 43 (12~24시간)
- 끊기면 **같은 명령을 다시** 치면 이어서 합니다.
- 확인할 것 (각 단계가 끝난 뒤):
  ```bash
  python lab/summarize_runs.py summary results/baseline/music --subject Music
  python lab/summarize_runs.py summary results/baseline/full
  cat results/baseline/holdout/seed_42/summary.json
  ```
  요청 오류(`오류` 열)가 0이어야 합니다.

**알려 줄 것:** 위 세 명령의 출력.

### 5-5. E1 GRPO 학습 + 평가 (약 2일 + 평가 15~30시간)

```bash
STEPS=140 bash lab/run_pipeline.sh grpo
```
- `STEPS`는 5-3에서 정한 스텝 수로 바꾸세요. 체크포인트가 25스텝마다 `results/grpo_run1/`에 저장되고, 끊겨도 **같은 명령을 다시** 치면 이어서 합니다.
- 진행 확인 (다른 tmux 창이나 새 접속에서):
  ```bash
  grep "heartbeat" logs/grpo.log | tail -5
  ```

학습이 끝나면:
```bash
bash lab/run_pipeline.sh merge holdout_new music4 full2 report
```
- `merge`: LoRA를 bf16 base에 합칩니다 (`results/grpo_run1/merged`). "LoRA가 적용되지 않았다"가 나오면 알려 주세요.
- `holdout_new` → `music4` → `full2`: baseline과 같은 방식으로 평가합니다.
- `report`: baseline과 학습 모델의 비교표(문항 단위 95% 신뢰구간)를 출력하고 `logs/report.log`에 남깁니다.

**알려 줄 것:** `report`의 출력 전체 (`cat logs/report.log`).

---

## 6. 결과를 저에게 보내기

큰 파일(모델 가중치, 체크포인트)을 뺀 요약만 압축합니다:
```bash
tar czf upload.tar.gz --exclude='results/*/merged' --exclude='results/*/checkpoint-*' \
  --exclude='results/*/jpeg' --exclude='*.safetensors' --exclude='*.pt' \
  results logs data/ssmr/select_report.json data/ssmr/screen_results.jsonl
ls -lh upload.tar.gz
```
Mac에서 받기 (Mac 터미널에서 실행, 계정·주소는 실습실에 맞게):
```bash
scp 계정@실습실주소:~/MMDL/upload.tar.gz ~/Downloads/
```

## 7. 에러가 났을 때

1. 파이프라인은 단계가 실패하면 **자동으로 CUDA/패키지 점검**을 먼저 실행하고 결과를 보여 줍니다. 그 출력을 먼저 읽으세요.
   - `❌ [FAIL]` 항목이 있으면 그게 원인일 가능성이 높습니다 (예: `NVIDIA wheel 혼재`, `드라이버 CUDA vs torch CUDA`).
   - FAIL이 없다면 환경은 정상이니 에러 로그 자체를 봅니다.
2. 직접 점검하려면:
   ```bash
   python lab/env_triage.py
   ```
3. 포트가 사용 중이라는 에러가 나면 이전 서버가 남아 있는 것입니다:
   ```bash
   pkill -f "vllm serve"; sleep 10; nvidia-smi
   ```
4. GPU 메모리 부족(`CUDA out of memory`)이나 vLLM 서버가 안 뜨면, 그 단계의 로그 마지막 60줄을 보내 주세요:
   ```bash
   tail -60 logs/<단계이름>.log          # 예: logs/stress.log, logs/grpo.log
   ls results/*/vllm_*.log               # 서버 로그 (kv cache, max seq len 줄 확인)
   ```
5. **알려 줄 것:** (1)의 점검 출력 전체 + 해당 단계 로그 마지막 60줄. 두 가지를 같이 보내 주시면 한 번에 원인을 좁힐 수 있습니다.

## 8. Music 평가를 8번으로 늘리기 (저에게 먼저 확인)

`music4` 결과가 애매하거나 향상이 보일 때만 seed 46~49를 추가합니다. 이 명령은 `CONFIRM_EXTEND_TO_8=yes`를 직접 붙이지 않으면 **실행되지 않습니다**. 결과표를 저에게 보여 주고 확장하기로 정한 뒤에만 사용하세요.
```bash
CONFIRM_EXTEND_TO_8=yes bash lab/run_pipeline.sh music_extend_base music_extend_new
```

## 9. 전체 한 번에 돌리기 (선택, 5-1~5-3로 시간을 확인한 뒤에)

```bash
bash lab/run_pipeline.sh all     # data screen stress pilot baseline_music4 holdout_base baseline_full2 grpo merge holdout_new music4 full2 report
```
`music_extend_*`는 포함되지 않습니다. 총 4~6일이 걸릴 수 있으니 반드시 tmux 안에서 실행하세요.

## 10. 폴더 위치 요약

| 무엇 | 어디 |
|---|---|
| 단계별 로그 | `logs/<단계이름>.log` |
| baseline 평가 | `results/baseline/music/run_00~03`, `results/baseline/full/run_00~01`, `results/baseline/holdout/seed_42` |
| 학습 모델 평가 | `results/grpo_run1/...` (같은 구조) |
| 학습 체크포인트, LoRA, merge 모델 | `results/grpo_run1/` (`final_lora`, `merged`) |
| 선별된 학습 문항 | `data/ssmr/train_subset.jsonl`, `data/ssmr/select_report.json` |
| 환경 기록 (GPU·드라이버·CUDA·pip freeze) | 각 결과 폴더의 `env_snapshot.json`, 루트의 `requirements_lab.lock.txt` |
