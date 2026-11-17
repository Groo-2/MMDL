# 제출 모델의 학습 쌍 (DPO)

| 파일 | 쌍 수 | 내용 | SHA-256 |
|---|---|---|---|
| `dpo_dv.jsonl` | 1,897 | DeepVision 수학 쌍 | `b2419a08c1228bfc626e5d5daa6ad22d5a98ea85c812f679a42d68327ac6ac25` |
| `dpo_mx.jsonl` | 1,052 | MusiXQA 악보 쌍 | `d0937312c0c3898cf35fa8789706ec2989e8612601e26ce6f5d7c2779563e5da` |
| `dpo_dv_mx.jsonl` | 2,949 | **학습에 쓴 파일**(위 둘을 이어 붙여 seed 42로 셔플) | `a1b386dd6ff753383b26b5c83a1edba03b9b3e54594da26ae298e2d2c233b147` |
| `*.report.json` | - | 쌍 구성 통계(잘린 응답 / 틀린 응답) | |

- 한 줄이 쌍 하나다: `question`, `options`, `answer`, `chosen`(맞고 끝난 응답), `rejected`(잘린 응답 또는 틀린 응답), `rejected_kind`(`truncated` / `wrong`), `source`(`dv` / `mx`), `data_dir`, `image_paths`.
- `chosen`과 `rejected`는 **base 모델(Qwen3-VL-4B-Instruct)이 생성한 응답**이다(문항당 8번 풀이, 최대 2,048토큰, T 0.7 / top_p 0.8 / top_k 20). 샘플링 결과라서 쌍을 새로 만들면 이 파일과 같지 않다.
- `dpo_dv_mx.jsonl`은 `code/music_ft/mix_dpo.py`로 `dpo_dv.jsonl`과 `dpo_mx.jsonl`에서 바이트 단위로 똑같이 만들어진다(실습실에서 `cmp`로 확인).
- **이미지는 포함하지 않았다.** `image_paths`가 가리키는 `images/*.jpg|png`는 원본에서 다시 만든다.
  - DeepVision: `skylenage/DeepVision-103K`(MIT)의 `math-77k.parquet`를 `code/music_ft/build_general.py --pool-size 12000`(seed 42)로 변환. 변환한 `train.jsonl`의 SHA-256이 `677e8f3dfb68e83f4e97db035fb4e811e2c032147ed1525ef33b95f2c629c1c2`이면 같은 풀이다.
  - MusiXQA: 릴리스 `musixqa-mcq-pilot20-v1`(MIT, 원본 puar-playground/MusiXQA)의 `images/`.
- 문항 원문과 이미지의 저작권은 각 원본 데이터셋(MIT)에 따른다. MMMU validation·test와는 중복을 제거했다.
