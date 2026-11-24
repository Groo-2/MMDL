"""Record Codex's visual inspection without presenting it as human review."""

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1] / "data" / "musixqa_mcq_pilot20_v1"


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    selection_path = ROOT / "review_selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    for name, digest in selection["input_sha256"].items():
        if sha256(ROOT / name) != digest:
            raise ValueError(f"Review input changed: {name}")
    by_id = {row["id"]: row for name in selection["input_sha256"] for row in read_jsonl(ROOT / name)}
    rows = [by_id[row_id] for row_id in selection["selected_ids"]]
    observations = json.loads((ROOT / "visual_observations.json").read_text(encoding="utf-8"))
    if len(rows) != 200 or [item["i"] for item in observations] != list(range(200)):
        raise ValueError("Expected 200 ordered visual observations")

    audit = {row["id"]: row for row in read_jsonl(ROOT / "generation_audit.jsonl")}
    csv_path = ROOT / "manual_review.csv"
    with csv_path.open(encoding="utf-8-sig", newline="") as stream:
        previous = list(csv.DictReader(stream))
    if [row["id"] for row in previous] != selection["selected_ids"]:
        raise ValueError("CSV selection differs from visual observation order")
    if any(row["verdict"] != "pending" for row in previous):
        raise ValueError("Existing review verdicts must be preserved")

    counts = Counter()
    finding_ids = []
    for row, obs, review in zip(rows, observations, previous):
        gold = row["options"]["ABCD".index(row["answer"])]
        matches = obs["observed"] == gold
        verdict = obs.get("verdict", "pass" if matches else "fix")
        if verdict == "pass" and not matches:
            raise ValueError(f"Visual answer differs: {row['id']}")
        options_unique = len(row["options"]) == len(set(row["options"]))
        trace = audit[row["id"]]
        distractors_trace = {d["text"] for d in trace["distractors"]}
        distractors_options = set(row["options"]) - {gold}
        distractors_ok = options_unique and distractors_trace == distractors_options
        if not distractors_ok:
            verdict = "fix"
        review.update({
            "verdict": verdict,
            "image_matches_answer": "yes" if matches else "no",
            "only_one_correct_option": "yes" if options_unique and matches else "no",
            "distractors_ok": "yes" if distractors_ok else "no",
            "difficulty_ok": "uncertain" if row["category"] == "Clef" else "yes",
            "notes": "Codex image inspection; observed: " + obs["observed"] + ". " + obs["note"],
        })
        counts[(row["category"], verdict)] += 1
        if verdict != "pass":
            finding_ids.append(row["id"])

    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=previous[0].keys())
        writer.writeheader()
        writer.writerows(previous)

    dedupe = json.loads((ROOT / "dedupe_report.json").read_text(encoding="utf-8"))
    overlap = json.loads((ROOT / "internal_overlap_report.json").read_text(encoding="utf-8"))
    report = {
        "reviewer": "Codex visual inspection",
        "human_review": False,
        "selection_sha256": sha256(selection_path),
        "observations_sha256": sha256(ROOT / "visual_observations.json"),
        "csv_sha256": sha256(csv_path),
        "items": len(rows),
        "counts": {cat: {outcome: counts[(cat, outcome)] for outcome in ("pass", "fix", "uncertain")}
                   for cat in ("KeySignature", "MeasureCount", "PitchReading", "Clef")},
        "findings": finding_ids,
        "mmmu_music_deduplication": {"reference": dedupe["reference"], "train_removed": dedupe["train"]["dropped"], "holdout_removed": dedupe["holdout"]["dropped"]},
        "internal_exact_image_overlap": overlap["shared_image_hash_count"],
        "limitations": ["Visual review is by Codex, not a human music specialist.",
                        "The 200 images are a deterministic sample, not every generated item.",
                        "Clef always includes an impossible alto-clef distractor; difficulty remains uncertain.",
                        "MMMU check covers Music dev/validation/test, not all MMMU subjects."],
    }
    (ROOT / "visual_review_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"counts": report["counts"], "findings": finding_ids}, ensure_ascii=False))


if __name__ == "__main__":
    main()
