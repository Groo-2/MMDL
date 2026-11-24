# MusiXQA 새 문항 중복 검사와 이미지 검수

검사 대상: `data/musixqa_mcq_pilot20_v1`. 2026-11-09 기준. 원본 MusiXQA 리비전 `d4cae4a5404b9c5d6dc8b85cce4e4c5d3585aae6`을 사용한 생성 데이터다. 검수자는 Codex이며, 사람의 검수를 수행했다고 표시하지 않는다.

## 중복 검사

| 검사 | 범위 | 결과 |
| --- | --- | --- |
| MMMU Music과 대조 | `dev` 5, `validation` 30, `test` 334문항의 이미지 519장. 정확한 SHA-1 이미지, 256비트 pHash 거리 20 이하, 선택지 집합이 같고 질문 유사도 0.9 이상인 문항 | 학습 후보 5,000개 중 제거 0개, holdout 400개 중 제거 0개 |
| 학습·holdout 이미지 중복 | 생성 데이터의 고유 이미지 5,270장. SHA-256 원본 바이트와 256비트 pHash 거리 20 이하 | 동일 이미지 0쌍, 유사 이미지 0쌍. 가장 가까운 쌍의 거리 30 |
| 학습·holdout 문항 중복 | 같은 이미지와 정규화한 질문·선택지 집합의 정확 일치 | 교차 분할 0건, 분할 내부 0건 |

가중 학습 부분집합은 중복 제거 후에도 `train_recipe.dedup.jsonl` 3,200문항이며, 범주별 수는 PitchReading 1,920, KeySignature 800, MeasureCount 320, Clef 160이다. 원래 `train_recipe.jsonl`과 내용 해시가 같다. 고정 holdout도 400문항을 유지한다.

MMMU 검사는 Music 범주에 한정된다. pHash 기준을 넘는 변형 이미지나 의미만 같은 바꿔 쓴 문항을 모두 잡는 검사는 아니다. 상세 조건과 재현 정보는 데이터 디렉터리의 `dedupe_report.json`, `internal_overlap_extended_report.json`에 있다.

## 새 문항 이미지 검수

시드 42로 각 범주에서 학습 부분집합 40개, holdout 10개를 뽑아 범주별 50개, 총 200개 악보를 직접 열어 대조했다. 조표는 첫 보표의 임시표 수, MeasureCount는 실제 그려진 마디 수, PitchReading은 지정 마디의 음자리표·조표·음높이·옥타브·붙임줄의 기보된 음표 수, Clef는 보표의 음자리표를 확인했다. 관찰한 답과 저장된 정답은 별도 기록한 뒤 대조했다.

| 범주 | 통과 | 수정 필요 |
| --- | ---: | ---: |
| KeySignature | 50 | 0 |
| MeasureCount | 50 | 0 |
| PitchReading | 49 | 1 |
| Clef | 50 | 0 |
| 합계 | 199 | 1 |

수정 필요: `musixqa_train_03736`(원본 문서 `0027194`). 질문은 bass clef의 **bar 9**를 가리키지만 악보에 인쇄된 9마디의 음은 `Ab2 C3 C3 C3 Ab3 C3`이다. 저장된 정답 `Eb3 Eb3 Eb3 Eb3 Gb3`은 인쇄된 **10마디**의 음이다. 이 악보는 인쇄된 첫 마디 번호가 2이므로 원본 QA의 9번째로 그려진 마디와 인쇄 번호 9가 어긋난다. 정답 선택지 네 개 중 인쇄된 9마디에 맞는 것은 없다. 학습에 사용하기 전에 질문을 “ninth written bar”처럼 순서가 분명한 표현으로 고치고 해당 문서의 마디 대응을 재검수하거나 이 문항을 제외해야 한다.

추가로 인쇄 번호가 2에서 시작하는 문서 `0067127`, `0089056`의 조표 문항과, 인쇄 번호가 실제 마디 수보다 1 큰 문서 `0039344`, `0077602`의 MeasureCount 문항을 확인했다. 이 네 문항의 정답은 맞지만, 해당 문서에 PitchReading 문항을 새로 만들 때 마디 번호 확인이 필요하다. 현재 후보·holdout에서 이 네 문서에 대응하는 PitchReading 문항은 없다.

Clef 문항은 이번 표본에서 treble only 17개, bass only 17개, both 16개였다. `alto clef only`는 생성 데이터에서 정답이 되지 않는 선택지라 난이도 평가는 보류했다. 200문항 검수 결과를 전체 5,400문항의 정답 검증으로 일반화할 수 없다.

문항별 판정은 `data/musixqa_mcq_pilot20_v1/manual_review.csv`와 `visual_observations.json`에, 이미지와 판정을 함께 보는 화면은 `review.html`에 있다. `manual_review.csv`라는 기존 파일명은 유지했지만 이번 200건은 **Codex 이미지 검수**다. `manifest.json`에는 사람 검수 0건, 공개 상태 초안, 수정 필요 문항 1개로 기록했다.

## 수정 완료 (2026-11-09)

`musixqa_train_03736`의 질문을 **“the 9th written bar … counting from the beginning of this sheet and ignoring printed measure numbers”**로 바꿨다. 검수한 아홉 번째 기보 마디는 인쇄된 10마디이고, 기존 정답 `Eb3 Eb3 Eb3 Eb3 Gb3`와 일치한다. 이미지·선택지·정답·holdout은 그대로다. 같은 문서의 확인하지 않은 다른 마디 질문은 생성기가 자동으로 통과시키지 않도록 등록했다.

기존 검사 결과를 보존하기 위해 수정본은 `train.reviewed.jsonl`, `train_recipe.reviewed.jsonl`, `generation_audit.reviewed.jsonl`, `manual_review.reviewed.csv`, `review.reviewed.html`로 따로 만들었다. 현재 학습 입력은 **`train_recipe.reviewed.jsonl`**, 평가 입력은 **`holdout.dedup.jsonl`**이다. 수정본의 표본 검수 판정은 **200/200 통과**이며, 사람 검수로 표시하지 않는다. `correction_report.json`에 원본·수정본 해시와 변경 내역이 있다.

수정된 3,200문항과 holdout 400문항을 형식 검증했으며 ERROR 0건, WARN 1건이다. WARN은 의도한 학습 비중 때문에 Clef가 160개로 검증기의 권장 200개보다 적다는 내용이다. 수정 질문의 선택지 집합은 MMMU Music 369문항 어느 것과도 일치하지 않았다.
