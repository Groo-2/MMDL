**표 9. 실험별 데이터와 주요 파라미터**

| 실험 | 학습 데이터 | 방법 | 주요 파라미터 | 학습 시간 |
|---|---|---|---|---|
| 공통 | Qwen3-VL-4B-Instruct @ ebb281e, bf16 | LoRA r=64, α=128, LLM의 q,k,v,o,gate,up,down(비전 타워 제외), 학습 가능 파라미터 132,120,576개(fp32 어댑터 약 504MiB) | - | - |
| GRPO run1 (E1) | SSMR-Bench Train_visual 1장 문항 중 통과율 1/8~7/8, 1,120문항 | GRPO, rollout 8, KL 없음(β=0), 정답 보상만, 스텝당 8문항 × 140스텝 | lr 2e-5 상수, max_completion 2048, T 0.7 / top_p 0.8 / top_k 20, grad accum 64, HF generate rollout | 9.9시간 |
| GRPO run2 (E1'') | 같음 | GRPO, 잘림 응답을 손실에서 제외(--mask-truncated), β=0.04, 60스텝 | max_completion 2048, 나머지 run1과 같음 | 15.4시간 |
| GRPO run3 (E1''') | 같음 | run2와 같고 스텝당 32문항 × 15스텝(총 480문항 동일) | 나머지 같음 | 16.3시간 |
| RFT dv1 | DeepVision 6,000문항 중 통과율 1~7/8인 1,897문항에서 base가 맞고 끝낸 응답 3,326개 | 지도학습(교차엔트로피, 응답 위치만) | lr 1e-5, grad accum 16, 1에폭 | - |
| RFT dv2 | 같음 | 같음 | lr 3e-6 | - |
| DPO dv1 | DeepVision 쌍 1,897개 (잘린 응답 1,271 / 틀린 응답 626) | DPO + 좋은 응답 NLL 보조항 | lr 5e-6 cosine, warmup 3스텝, β=0.1, NLL 가중치 0.1, 기준=어댑터를 끈 base, grad accum 16쌍, 1에폭, seed 42 | 4,998초 / 재학습 4,684초(119스텝) |
| DPO dv1+MusiXQA (제출) | DeepVision 쌍 1,897 + MusiXQA 쌍 1,052 (PitchReading 466, KeySignature 470, MeasureCount 116) = 2,949쌍 | DPO + NLL 보조항 (설정은 DPO dv1과 같음) | 185스텝 | 8,321초 |
| MusiXQA GRPO(팀원) | MusiXQA 객관식(팀원 변환) | GRPO 40스텝, 정답 보상 | lr 2e-5 상수, 장치당 batch 1, LoRA r64/α128(같은 구성), rollout 수·보상 세부는 기록에서 확인 못함 | - |

*GRPO run1의 설정은 reports/ssmr_grpo_run1/README.md, DPO는 reports/dpo_dv1/README.md, 팀원 설정은 trainer_state.json에서 읽은 값.*
