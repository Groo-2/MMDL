# 수업 시간 중단과 재개

수업 시간에는 GPU 원격이 꺼진다. 프로세스가 중간에 죽어도 **같은 명령을 다시 실행하면 이어지도록** 모든 단계를 만들었다.
tmux는 접속이 끊기는 경우에는 살아 있지만, 머신이 재시작되거나 프로세스가 종료되면 함께 죽는다.

## 재개 절차 (항상 이 순서)
```bash
source $HOME/miniconda3/etc/profile.d/conda.sh && conda activate mmdl && cd ~/MMDL_ssmr     # 평가는 mmdl130
pgrep -af "vllm serve" || echo "남은 서버 없음"      # 있으면 pkill -f "vllm serve" 로 정리 (GPU 메모리를 잡고 있을 수 있다)
nvidia-smi --query-gpu=memory.used --format=csv,noheader
python lab/status.py                                  # 어디까지 저장됐는지
# 그다음 중단됐던 명령을 그대로 다시 실행
```

## 단계별 저장 단위와 잃는 양
| 단계 | 저장 단위 | 중단되면 잃는 것 | 재개 |
|---|---|---|---|
| 악보 선별(`select_ssmr.py`) | 문항 1개마다 | 진행 중이던 문항(동시 16개) | 같은 명령 |
| 학습(`train_dpo.py`) | **체크포인트 N스텝마다** | 마지막 체크포인트 이후 최대 N스텝 | 같은 명령에 `--resume` |
| 합치기(`merge_lora.py`) | 끝나야 저장 | 합치기 전체(10~20분) | 다시 실행. `merged/.merge_ok`가 없으면 반쪽이다 |
| 평가 추론(`eval_repeat.py`, `eval_holdout.py`) | 요청 1개마다 | 진행 중이던 요청(동시 32개, 32,768토큰까지 가는 루프는 최대 약 30분) | 같은 명령 |
| 평가 judge | 요청 1개마다 | 진행 중이던 요청 | 같은 명령 |

## 평가 대기열 (여러 모델을 순서대로)
평가가 모델마다 10시간 안팎이라, 한 번 시작해 두고 끊기면 같은 명령 한 줄로 이어 가도록 `lab/eval_queue.py`를 쓴다.
```bash
conda activate mmdl130 && cd ~/MMDL_ssmr
python lab/eval_queue.py \
  base_130=Qwen/Qwen3-VL-4B-Instruct@ebb281ec70b05090aa6165b016eac8ec08e71b17:2 \
  mx_grpo_team=results/mx_grpo_team/merged:2 \
  dpo_dv1_re=results/dpo_dv1_re/merged:2 2>&1 | tee -a logs/eval_queue.log
```
- 항목은 `태그=모델[@리비전]:run수`이고, 결과는 `results/<태그>/run_00, run_01`에 쌓인다(seed는 42 + 번호).
- run 폴더에 `summary.json`이 있으면 끝난 run으로 보고, 항목의 run이 모두 끝났으면 서버를 띄우지 않고 건너뛴다. 하나라도 남았으면 `eval_repeat.py`가 응답 단위로 이어받는다.
- 어느 항목이 실패하면 그 뒤 항목은 실행하지 않고 멈춘다. 서버가 남아 있으면 `pkill -f "vllm serve"`로 정리하고 원인을 해결한 뒤 **같은 명령을 다시** 실행한다.
- `--dry-run`을 붙이면 무엇이 남았는지만 출력한다.

## 체크포인트 간격 (학습)
권장: `--save-steps 20 --save-total-limit 3`

- 스텝당 약 39초(DPO 실측)라서 20스텝은 약 13분이다. 중단되면 최대 약 13분어치를 잃는다.
- 체크포인트 1개는 약 1.6GB(어댑터 0.5GB + 옵티마이저 1.06GB)이고, 저장은 몇 초(추정)라서 간격이 13분이면 오버헤드는 1~2% 이하다.
  간격을 5분 이하로 줄이면 저장 횟수와 디스크 쓰기만 늘고 잃는 양은 크게 줄지 않는다. 반대로 50스텝 이상이면 한 번 끊길 때 30분 이상을 잃는다.
- `--save-total-limit 3`은 체크포인트 3개(약 5GB)만 남긴다.
- **체크포인트 저장은 원자적이지 않다.** transformers 5.19.0의 Trainer는 `checkpoint-N`에 바로 쓰고(가중치 → 옵티마이저 → RNG → `trainer_state.json` 순), 저장 도중에 죽으면 가장 높은 번호의 체크포인트가 반쪽으로 남는다.
  `--resume`은 이어받기 전에 `trainer_state.json`이 없는 체크포인트를 `incomplete-checkpoint-N`으로 이름만 바꿔 건너뛰고(`music_ft/ckpt_utils.py`, 지우지는 않는다) 직전의 온전한 체크포인트에서 이어간다.
  단, 운영체제가 통째로 꺼져서 디스크 캐시가 비워지지 않은 경우까지는 이 검사로 알 수 없다(수업 시간 원격 종료는 대개 프로세스 종료라 해당하지 않을 것으로 본다).
- 학습 명령에는 항상 `--resume`을 붙인다. 체크포인트가 없으면 처음부터 시작하고, 있으면 마지막 것에서 이어간다.

```bash
python music_ft/train_dpo.py --data data/dpo/<쌍 파일>.jsonl --output-dir results/<이름> --epochs 1 --save-steps 20 --save-total-limit 3 --resume
```

## 죽는 순간의 줄 깨짐
이어받는 파일(`predictions.jsonl`, `judge_results.jsonl`, `screen_results.jsonl`, `screen_samples.jsonl`)에 줄을 쓰다 죽으면 마지막 줄이 반쪽이 된다.
그대로 두면 다시 읽을 때 `JSONDecodeError`로 멈추거나 다음 줄이 반쪽 뒤에 붙어 두 레코드가 함께 깨진다.
`eval_repeat.py`, `eval_holdout.py`, `select_ssmr.py`는 이어받기 전에 `lab/jsonl_repair.py`로 반쪽 마지막 줄을 잘라 낸다(그 문항은 아직 안 끝난 것이라 다시 풀린다).
직접 확인하려면 `python lab/jsonl_repair.py results/<이름>`.

선별은 응답 샘플을 먼저 쓰고 통과율을 나중에 쓴다. 그 사이에 죽어도 "끝난 것으로 보이는데 샘플이 없는" 문항이 생기지 않는다(대신 다시 풀면 샘플이 중복될 수 있고, 읽는 쪽은 마지막 줄을 쓴다).

## 수업 직전에 안전하게 멈추려면
- 학습: 체크포인트가 저장된 직후(20스텝 배수 근처)에 멈추면 잃는 것이 없다. `python lab/status.py`의 "최근 체크포인트 step"과 로그의 현재 스텝을 비교한다.
- **선별(`select_ssmr.py`)과 홀드아웃 평가(`eval_holdout.py`)**: `Ctrl+c`로 멈춘다. 시작하지 않은 문항은 취소하고 실행 중이던 문항(동시 요청 수만큼)이 끝나면 종료한다. 몇 분 걸릴 수 있다.
- **전체 평가(`eval_repeat.py`)는 `Ctrl+c`가 바로 멈추지 않는다.** 평가 본체(`mmmu_eval/`, 동결 코드)가 남은 문항을 큐에 모두 넣고 끝까지 처리하며,
  `Ctrl+c` 뒤에 끝나는 응답은 **저장되지 않는다**(2026-11-15 선별 코드에서 확인한 것과 같은 구조라서 같은 문제가 있을 것으로 본다. 평가에서는 아직 확인하지 않았다). 급하면 프로세스를 끈다.
```bash
pkill -f eval_repeat.py; pkill -f "vllm serve"; sleep 15; pgrep -af "eval_repeat|vllm serve" || echo "모두 종료"
```
  저장은 응답마다 하므로 끈 시점까지의 결과는 남고, 진행 중이던 요청만 다시 풀린다.
- 위 `pkill` 뒤에 `vllm serve`가 남아 있으면 GPU 메모리를 계속 잡는다. 꼭 같이 끄고 `nvidia-smi`로 메모리가 비었는지 확인한다.
