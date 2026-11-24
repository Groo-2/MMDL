"""Apply the two verified bar-reference edits, preserving the user's review history."""

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys

import build_musixqa as builder
from prepare_musixqa_review import prepare, read_jsonl

FIXES = {"musixqa_train_00038": "0019023", "musixqa_holdout_00009": "0087904"}


def apply(data_dir):
    root = Path(data_dir).resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("sample_revision") == "bar_reference_fix_1":
        print("Already applied; existing review results were preserved.")
        return
    datasets = {split: read_jsonl(root / f"{split}.jsonl") for split in ("train", "holdout")}
    all_rows = [row for rows in datasets.values() for row in rows]
    by_id = {row["id"]: row for row in all_rows}
    audits = read_jsonl(root / "generation_audit.jsonl")
    traces = {row["id"]: row for row in audits}
    with (root / "manual_review.csv").open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields, reviews = list(reader.fieldnames), list(reader)
    if len(reviews) != len(by_id) or {row["id"] for row in reviews} != set(by_id):
        raise ValueError("Review IDs do not match the dataset")
    verdicts = Counter(row["verdict"] for row in reviews)
    if verdicts != {"pass": 78, "fix": 2} or {r["id"] for r in reviews if r["verdict"] == "fix"} != set(FIXES):
        raise ValueError("Expected the reported 78 pass / two specific fixes; review the changed CSV first")
    for split in datasets:
        if builder.sha256(root / f"{split}.jsonl") != manifest["data_sha256"][f"{split}.jsonl"]:
            raise ValueError("Data hash differs from the reviewed manifest")
    changes = []
    for item_id, doc in FIXES.items():
        row, trace = by_id[item_id], traces[item_id]
        if row["source_doc_id"] != doc or row["question"] != "<image 1> " + trace["question"] or "bar 1" not in row["question"]:
            raise ValueError(f"Unexpected original question for {item_id}")
        replacement = "<image 1> " + builder.output_question(trace)
        if replacement == row["question"]:
            raise ValueError(f"No correction generated for {item_id}")
        changes.append({"id": item_id, "source_doc_id": doc,
                        "old_question": row["question"], "new_question": replacement,
                        "reason": "First visible bar is printed as 2; cause of source indexing mismatch is unconfirmed.",
                        "recheck_status": "pending"})
    backup = root.with_name(root.name + "_before_bar_reference_fix")
    if backup.exists():
        raise FileExistsError(f"Preserve existing backup and inspect before retrying: {backup}")
    backup.mkdir()
    # Image bytes do not change; their hashes remain in the backed-up manifest.
    for name in ("train.jsonl", "holdout.jsonl", "generation_audit.jsonl", "manifest.json",
                 "manual_review.csv", "validation_report.json", "build_musixqa.py", "review.html"):
        shutil.copyfile(root / name, backup / name)
    for change in changes:
        row, trace = by_id[change["id"]], traces[change["id"]]
        row["question"] = change["new_question"]
        trace["output_question"] = change["new_question"].removeprefix("<image 1> ")
        trace["question_revision"] = change
    for split, rows in datasets.items():
        builder.save_jsonl(root / f"{split}.jsonl", rows)
    builder.save_jsonl(root / "generation_audit.jsonl", audits)
    for name in ("resolution", "revision_note"):
        if name not in fields:
            fields.append(name)
    for review in reviews:
        if review["id"] in FIXES:
            review["resolution"] = "question_updated_recheck_needed"
            review["revision_note"] = "질문을 first visible bar (printed bar number 2)로 수정. 기존 정답·선택지·이미지는 동일. 기존 fix 기록을 보존하며 재확인 후 판정 필요."
    with (root / "manual_review.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(reviews)
    builder.write_json(root / "question_revisions.json", changes)
    builder.write_json(root / "quality_report.json", builder.quality_summary(all_rows, audits))
    shutil.copyfile(Path(builder.__file__), root / "build_musixqa.py")
    manifest.update(sample_revision="bar_reference_fix_1", revised_at_utc=datetime.now(timezone.utc).isoformat())
    manifest["data_sha256"] = {f"{split}.jsonl": builder.sha256(root / f"{split}.jsonl") for split in datasets}
    manifest["manual_review"] = {
        "status": "completed_with_pending_rechecks", "reviewed": len(reviews),
        "source": "User report and manual_review.csv; not an independent agent review of all 80 images",
        "verdict_counts": dict(verdicts),
        "per_category": {cat: dict(Counter(r["verdict"] for r in reviews if r["category"] == cat)) for cat in builder.CATEGORIES},
        "pending_recheck_ids": list(FIXES), "question_revisions": "question_revisions.json",
        "conventions": {
            "MeasureCount": "Count written measures once, ignoring repeat playback; do not equate the last printed number with the count.",
            "PitchReading": "Count each written notehead, including both written notes joined by a tie.",
        },
    }
    manifest["quality_report"] = "quality_report.json"
    manifest["generator_sha256"] = builder.sha256(root / "build_musixqa.py")
    builder.write_json(root / "manifest.json", manifest)
    validator = Path(__file__).resolve().parents[1] / "lab/validate_mcq_dataset.py"
    subprocess.run([sys.executable, "-X", "utf8", str(validator), "--data-dir", str(root),
                    "--min-train-per-category", "16", "--min-holdout-per-category", "4"], check=True)
    report = json.loads((root / "validation_report.json").read_text(encoding="utf-8"))
    manifest["validation"] = {"errors": report["error_count"], "warnings": report["warnings"]}
    builder.write_json(root / "manifest.json", manifest)
    prepare(root)
    print("Updated two questions; user verdicts preserved; both revisions await recheck.")
    print("Original metadata/review backup:", backup)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/musixqa_mcq_sample")
    apply(parser.parse_args().data_dir)
