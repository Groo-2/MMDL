"""Package the reviewed sample for team review; does not approve or publish it."""

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

from build_musixqa import sha256
from prepare_musixqa_review import prepare, read_jsonl

README = """# MusiXQA 객관식 샘플 — 팀 검토용

학습 64문항 + holdout 16문항, 총 80문항입니다. 네 필수 카테고리마다 학습 16 + holdout 4개입니다.
제작자 검수는 80/80 통과했고, 마디 번호 문제 두 건을 수정한 뒤 재확인했습니다.
이는 제작자 검수 결과입니다. 팀의 샘플 승인, MMMU 중복 검사, 이미지 토큰 수 측정은 별도 단계입니다.

## 먼저 볼 파일

1. 압축을 푼 뒤 `review.html`을 브라우저로 열면 원본 악보·문제·정답·검수 기록을 볼 수 있습니다.
2. `manual_review.csv`: 80문항의 검수 결과와 메모.
3. `sample_review_report.md`: 문제 발견, 수정, 해석 규칙, 남은 한계.
4. `question_revisions.json`: 두 마디 위치 질문의 수정 및 재확인 기록.
5. `manifest.json`, `validation_report.json`, `quality_report.json`: 출처, 해시, 형식 검증, 분포.

## 팀 확인 요청

- 질문·오답 구성·난이도가 요청서와 과제에 맞는지 확인해 주세요.
- Clef는 원본 정답 유형이 세 가지뿐입니다. `alto clef only`는 항상 오답인 선택지입니다.
  이 카테고리의 유지·비중·추가 데이터 필요 여부를 결정해 주세요.
- 전체 데이터 생성 시 KeySignature/Clef의 실제 정답 내용을 균등 선별하는 `--balance-content` 적용을 제안합니다.
  샘플 A~D 위치는 균등하지만, 정답 내용은 Clef both 13/20, KeySignature 7 flats 5/20으로 치우쳐 있습니다.
- PitchReading의 '음자리표 착각' 오답은 요청서의 예외로 해석해 모든 음을 바꿉니다. 다른 오답 유형은 1~2개 음만 바꿉니다.
- 마디 수는 반복 재생 횟수를 제외한 적힌 마디 수, 붙임줄 양쪽 음표는 각각 세는 기준입니다.

요청서 9절에 따라 팀이 이 샘플을 확인한 후 전체 데이터 생성으로 진행합니다.
전체 목표는 카테고리당 학습 800 이상 / holdout 100 이상이며, 최종 중복 제거 뒤에도 목표 분량을 만족해야 합니다.

## 코드와 출처

원본: puar-playground/MusiXQA (MIT), revision d4cae4a5404b9c5d6dc8b85cce4e4c5d3585aae6.
Chen et al., MusiXQA: Advancing Visual Music Understanding in Multimodal Large Language Models, 2025.
PNG는 원본 바이트 그대로이며 SHA-256은 manifest에 있습니다.

Python 3.12에서 재생성하려면 별도 환경에 `requirements_musixqa.txt`를 설치하고,
`build_musixqa.py download --images`로 원본을 받은 뒤
`build_musixqa.py build --out-dir rebuilt_sample --validator validate_mcq_dataset.py`를 실행합니다.
원본 다운로드는 합계 약 9.94 GB이며, 이 ZIP에는 선정된 샘플 이미지만 들어 있습니다.
CSV의 사람 검수 기록은 재생성 코드가 자동으로 만들어 주지 않습니다.

이 자료는 아직 전체 학습용 데이터 완성본이 아닙니다.
"""


def package(data_dir, out_path):
    root = Path(data_dir).resolve()
    output = Path(out_path).resolve()
    if output.exists():
        raise FileExistsError(f"Keep the existing archive; choose another --out: {output}")
    if (root / "BUILD_INCOMPLETE.json").exists():
        raise ValueError("Incomplete dataset build")
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    rows = [row for split in ("train", "holdout") for row in read_jsonl(root / f"{split}.jsonl")]
    by_id = {row["id"]: row for row in rows}
    with (root / "manual_review.csv").open(encoding="utf-8-sig", newline="") as stream:
        reviews = list(csv.DictReader(stream))
    if len(rows) != 80 or len(by_id) != 80 or len(reviews) != 80 or {r["id"] for r in reviews} != set(by_id):
        raise ValueError("Expected a matching 80-item sample and review sheet")
    if Counter(r["verdict"] for r in reviews) != {"pass": 80}:
        raise ValueError("Some sample items have not passed manual review")
    for review in reviews:
        row = by_id[review["id"]]
        if review["expected_answer"] != row["answer"] or review["expected_option"] != row["options"]["ABCD".index(row["answer"])]:
            raise ValueError(f"Review answer mismatch: {row['id']}")
    if manifest["manual_review"]["status"] != "completed" or manifest["manual_review"].get("pending_recheck_ids"):
        raise ValueError("Manifest review state is not complete")
    revisions = json.loads((root / "question_revisions.json").read_text(encoding="utf-8"))
    if any(r["recheck_status"] != "pass" or by_id[r["id"]]["question"] != r["new_question"] for r in revisions):
        raise ValueError("A changed question has not been rechecked")
    for name, digest in manifest["data_sha256"].items():
        if sha256(root / name) != digest:
            raise ValueError(f"Data hash mismatch: {name}")
    for name, digest in manifest["image_sha256"].items():
        if sha256(root / "images" / name) != digest:
            raise ValueError(f"Image hash mismatch: {name}")
    repo = Path(__file__).resolve().parents[1]
    validator = repo / "lab/validate_mcq_dataset.py"
    subprocess.run([sys.executable, "-X", "utf8", str(validator), "--data-dir", str(root),
                    "--min-train-per-category", "16", "--min-holdout-per-category", "4"], check=True)
    prepare(root)
    shutil.copyfile(validator, root / "validate_mcq_dataset.py")
    shutil.copyfile(repo / "music_ft/requirements_musixqa.txt", root / "requirements_musixqa.txt")
    shutil.copyfile(repo / "reports/musixqa_sample_review.md", root / "sample_review_report.md")
    (root / "README_SAMPLE.md").write_text(README, encoding="utf-8")
    names = ["train.jsonl", "holdout.jsonl", "manifest.json", "validation_report.json",
             "generation_audit.jsonl", "manual_review.csv", "question_revisions.json", "quality_report.json",
             "source_clef_distribution.json", "build_musixqa.py", "validate_mcq_dataset.py",
             "requirements_musixqa.txt", "sample_review_report.md", "review.html", "README_SAMPLE.md"]
    names += [f"images/{name}" for name in sorted(manifest["image_sha256"])]
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in names:
            archive.write(root / name, f"musixqa_mcq_sample/{name}")
    with zipfile.ZipFile(output) as archive:
        if archive.testzip() is not None or len(archive.namelist()) != len(names):
            raise ValueError("ZIP validation failed")
        for name in names:
            if archive.read(f"musixqa_mcq_sample/{name}") != (root / name).read_bytes():
                raise ValueError(f"ZIP content mismatch: {name}")
    digest = sha256(output)
    output.with_suffix(".zip.sha256").write_text(f"{digest}  {output.name}\n", encoding="ascii")
    print("Archive:", output)
    print("Size MB:", round(output.stat().st_size / 1_000_000, 2))
    print("SHA256:", digest)
    print("Team approval:", manifest.get("team_sample_review"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/musixqa_mcq_sample")
    parser.add_argument("--out", default="data/musixqa_mcq_sample.zip")
    args = parser.parse_args()
    package(args.data_dir, args.out)
