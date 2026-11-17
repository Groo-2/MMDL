**표 11. 둘 다 끝낸 문항의 유형별 정확도 % (base → 모델)**

| 모델 | 유형 | 문항 수 | 정확도 % (base → 모델) |
|---|---|---|---|
| MusiXQA GRPO(팀원) | Clef | 100 | 98.5 → 96.5 |
| MusiXQA GRPO(팀원) | KeySignature | 70 | 60.7 → 51.4 |
| MusiXQA GRPO(팀원) | MeasureCount | 64 | 68.8 → 69.5 |
| MusiXQA GRPO(팀원) | PitchReading | 8 | 81.2 → 75.0 |
| DPO(DeepVision) | Clef | 100 | 98.5 → 96.0 |
| DPO(DeepVision) | KeySignature | 70 | 60.7 → 57.9 |
| DPO(DeepVision) | MeasureCount | 63 | 69.8 → 72.2 |
| DPO(DeepVision) | PitchReading | 5 | 80.0 → 80.0 |
| DPO(DeepVision+MusiXQA) | Clef | 100 | 98.5 → 95.5 |
| DPO(DeepVision+MusiXQA) | KeySignature | 70 | 60.7 → 61.4 |
| DPO(DeepVision+MusiXQA) | MeasureCount | 64 | 68.8 → 69.5 |
| DPO(DeepVision+MusiXQA) | PitchReading | 8 | 81.2 → 75.0 |

*PitchReading은 base가 거의 끝내지 못해(약 76%가 잘림) 공통 문항이 5~8개뿐이라 비교할 수 없다(문항 수 5 미만이면 '-').*
