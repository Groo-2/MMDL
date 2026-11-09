# MusiXQA 데이터 제작: 샘플부터 시작

실행 위치: 저장소 루트. Windows PowerShell, `.venv_lab`의 Python 3.12 사용.
요청 기준: [MusiXQA 객관식 요청서](MusiXQA_객관식_요청서.md).

## 1. 환경

```powershell
.\.venv_lab\Scripts\python.exe -m pip install -r music_ft/requirements_musixqa.txt
```

현재 PC의 `py` 런처는 존재하지 않는 Python 3.14를 가리킨다. 이미 생성한
`.venv_lab\Scripts\python.exe`를 사용한다. 데이터는 Git에서 제외된 `data/`에 저장된다.

## 2. 작은 test QA로 원본 구조 확인

```powershell
.\.venv_lab\Scripts\python.exe -X utf8 music_ft/build_musixqa.py download --test-only
.\.venv_lab\Scripts\python.exe -X utf8 music_ft/build_musixqa.py inspect
```

test QA 두 파일은 합계 약 28 MB다. 고정 revision은
`d4cae4a5404b9c5d6dc8b85cce4e4c5d3585aae6`이다.

`data/musixqa_inspection/`에 원본 분석 보고서와 텍스트 미리보기 80개가 생긴다.
**이 미리보기는 모두 원본 test에서 왔으며 학습에 사용하면 안 된다.** 이미지 존재,
악보와 정답 일치 여부는 확인하지 않은 상태다. 미리보기는 팀 제출용 샘플이 아니다.

## 3. 실제 샘플에 필요한 원본 다운로드

```powershell
.\.venv_lab\Scripts\python.exe -X utf8 music_ft/build_musixqa.py download --images
```

QA 네 파일 합계 약 1.26 GB와 이미지 원본 약 8.68 GB를 받는다. 총 약 9.94 GB이며
이미 받은 파일은 같은 local_dir의 Hugging Face 캐시를 재사용한다.
전체 이미지를 압축 해제하지 않고 선정된 문서의 PNG만 추출한다.
metadata.json과 midi.tar는 이 제작 과정에서 받지 않는다.

## 4. 팀 검토용 샘플 80문항 생성

```powershell
.\.venv_lab\Scripts\python.exe -X utf8 music_ft/build_musixqa.py build
```

출력: `data/musixqa_mcq_sample/`.

- 네 필수 카테고리마다 train 16개 + holdout 4개, 합계 80개.
- train은 원본 train, holdout은 원본 test. 원본 문서 집합의 겹침도 검사.
- 문서당 최대 2문항. 각 분할/카테고리에서 A~D 정확히 25%씩.
- PitchReading은 8음 이하이며 높은음자리표와 낮은음자리표를 절반씩 선정.
- 같은 문서/질문의 중복은 제외하고, 상충하는 라벨은 후보에서 제거.
- PNG 원본 바이트 보존. 선택된 이미지의 SHA-256을 manifest에 기록.
- 기존 형식 검증기를 마지막에 실행. 샘플 기준 분량 16/4를 전달.
- 출력 폴더가 이미 있으면 덮어쓰지 않는다. 재실행은 새 `--out-dir`로 한다.
- 실패한 출력에는 `BUILD_INCOMPLETE.json`이 남는다. 이런 폴더는 제출하지 않는다.

`generation_audit.jsonl`에는 원본 파일/행 위치, 원본 질문/정답, 오답 생성 유형이 있다.
manifest의 manual_review, MMMU 중복 제거, 팀 검토 상태는 처음에는 모두 pending이다.
자동 형식 검증을 내용 검수로 기록하지 않는다.

## 5. 샘플 검수에서 확정할 내용

검수 화면과 기록표를 생성한다.

```powershell
.\.venv_lab\Scripts\python.exe -X utf8 music_ft/prepare_musixqa_review.py
```

`data/musixqa_mcq_sample/review.html`을 브라우저로 열면 악보와 문제를 함께 볼 수 있다.
정답은 펼치기 안에 있으므로 먼저 악보를 읽고 대조한다. 이미지를 클릭하면 원본 크기로 열린다.
`manual_review.csv`는 Excel 등에서 열어 문항 ID별로 기록한다. 화면은 읽기 전용이며 체크 결과를 자동 저장하지 않는다.
`verdict`는 pass/fix/uncertain/pending, 세부 항목은 yes/no/uncertain으로 적고 검수자와 메모를 남긴다.
이 스크립트는 기존 CSV를 덮어쓰지 않고 manifest의 검수 상태를 변경하지 않는다.
CSV 검수 결과를 취합하고 실제 수행 내역에 맞게 manifest를 갱신한 후 전달한다.

1. 80문항 모두 악보와 대조: 정답, 마디 번호, 음자리표, 선택지 모호성.
2. 요청서 5-3의 충돌: 구현은 '음자리표 착각은 모든 음 변경'을 명시된 예외로 해석했다.
   이 유형은 한 문항에 최대 1개, 나머지 유형은 1~2개 음만 변경한다. 팀 샘플 검토에서 확인한다.
3. 조표 0개는 샵/플랫을 뒤집어도 0개라 오답이 되지 않는다. 인접한 비영 조표로 대체했다.
4. Clef는 요청서의 고정 네 선택지를 쓴다. 라벨 빈도와 선택지 길이의 지름길 가능성을 검토한다.
5. 질문 난이도가 과제의 대학 수준 조건에 부합하는지 확인한다.

내용 검수 결과와 WARN 사유를 기록한 뒤 샘플 폴더를 ZIP으로 묶어 팀에 전달한다.
**2026-11-08 사용자가 팀 확인을 명시적으로 건너뛰도록 지시했다. 현재 작업에서는 그 지시를 적용하며, 새 데이터 내용 검수는 별도로 수행한다.**
전체 목표는 카테고리당 train 800 이상, holdout 100 이상이다. 최종 데이터는 MMMU 중복
제거 후에도 목표 분량/정답 균형을 만족해야 하며, 카테고리별 무작위 50문항 이상을 직접 검수한다.
현재 dedupe.py는 MMMU Music 범위만 검사하므로 검사 범위를 보고서에 명시한다.

## 6. 코드 확인

```powershell
.\.venv_lab\Scripts\python.exe -X utf8 music_ft/test_build_musixqa.py
```

스트리밍 JSON, 경계 조표, 오답의 음 높이 차이, 정답 재매핑, 분량/문서 제한,
정답 위치 균형, 재현성, 모순 라벨 제외를 확인한다. 이미지 내용의 음악적 정확성은 자동 테스트 범위가 아니다.

## 7. 2026-11-08 샘플 검수 반영

상세 결과: [샘플 검수 보고서](../reports/musixqa_sample_review.md).

- 두 PitchReading 질문은 `first visible bar (printed bar number 2)`로 수정했고, 사용자 재확인까지 완료됐다. 현재 검수 결과는 80/80 pass다.
- 기존 검수 CSV의 판정과 메모를 보존했다. 화면 재생성 시 CSV의 현재 판정과 수정 상태가 표시된다.
- 알려진 번호 불일치 문서에서 검증되지 않은 PitchReading 마디 대응은 후보에서 제외한다.
  아직 전체 이미지를 대상으로 한 마디 번호 OCR 검사는 없다.
- 새로 생성되는 `quality_report.json`에서 A~D 위치와 실제 정답 내용의 분포를 각각 확인한다.
- `inspect`/`build`에 `--balance-content`를 추가하면 각 분할의 KeySignature와 Clef를 정답 내용별로 균등 선별한다.
  선별 문항이 바뀌므로 새 출력 폴더로 생성하고 새 결과를 검수해야 한다.
- 실제로 없는 alto 정답 라벨 문제는 내용 균등 선별로 해결되지 않는다. 팀 검토에서 별도 결정한다.

## 8. pilot20 기반 조정안 (현재 적용)

사용자가 제공한 pilot20 보고서를 분석해 [데이터 구성 v1](../reports/musixqa_pilot20_recipe.md)을 작성했다.
팀 확인 단계는 사용자 지시로 면제한다. 이는 팀 승인을 받았다는 뜻은 아니다.

- 학습용 3,200개: PitchReading 60%, KeySignature 25%, MeasureCount 10%, Clef 5%.
- 후보 풀 5,000개는 네 카테고리 모두 800개 이상. 평가용 holdout은 카테고리당 100개, 총 400개로 고정.
- 조표 15종과 실제 Clef 정답 세 유형을 균등 선별하고 A~D 위치도 내용별로 균등 배정. 마디 수는 희귀 경계값을 과대표집하지 않고 원본 분포를 따른다.
- 높은/낮은음자리표 음 높이 문제는 반반. 1~3/4~6/7~8음 비중은 30/45/25.
- 오선 위치 ±2칸 혼동을 포함하며, 각 음 높이 문제에 위치 착각 오답을 하나 넣는다.

```powershell
.\.venv_lab\Scripts\python.exe -X utf8 music_ft/build_musixqa_recipe.py build
```

출력 폴더 `data/musixqa_mcq_pilot20_v1/`의 `train.jsonl`은 후보 풀이고, `train_recipe.jsonl`이 조정된 학습용 부분집합이다.
학습 전에 중복 제거·내용 검수·이미지 토큰 수/메모리 점검이 필요하다. 이후 학습 명령의 `--train-file`도 해당 부분집합을 가리켜야 한다.
