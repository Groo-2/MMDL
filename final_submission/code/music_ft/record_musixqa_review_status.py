"""Record completed checks in the local dataset manifest."""

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1] / "data" / "musixqa_mcq_pilot20_v1"


def digest(name):
    return hashlib.sha256((ROOT / name).read_bytes()).hexdigest()


def main():
    manifest_path = ROOT / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    visual = json.loads((ROOT / "visual_review_report.json").read_text(encoding="utf-8"))
    mmmu = json.loads((ROOT / "dedupe_report.json").read_text(encoding="utf-8"))
    internal = json.loads((ROOT / "internal_overlap_extended_report.json").read_text(encoding="utf-8"))
    if visual["items"] != 200 or visual["findings"] != ["musixqa_train_03736"]:
        raise ValueError("Unexpected visual review result")
    if mmmu["train"]["dropped"] or mmmu["holdout"]["dropped"]:
        raise ValueError("Deduplication removed rows; rebuild recipe first")
    if internal["cross_split_exact_image_pairs"] or internal["cross_split_near_image_pairs"]:
        raise ValueError("Cross-split overlap must be resolved first")
    if digest("holdout.jsonl") != manifest["data_sha256"]["holdout.jsonl"]:
        raise ValueError("Holdout changed")
    manifest["file_roles"].update({
        "train.dedup.jsonl": "candidate pool after MMMU Music duplicate check; no rows removed",
        "train_recipe.dedup.jsonl": "historical weighted subset after MMMU Music duplicate check, before the reviewed wording correction",
        "holdout.dedup.jsonl": "frozen diagnostic holdout after MMMU Music duplicate check; no rows removed",
        "manual_review.csv": "Codex visual inspection of 200 deterministic sample items; reviewer is not human",
    })
    for name in ("train.dedup.jsonl", "train_recipe.dedup.jsonl", "holdout.dedup.jsonl"):
        manifest["data_sha256"][name] = digest(name)
    manifest["visual_review"] = {
        "status": "completed_with_one_correction_needed",
        "reviewer": "Codex visual inspection", "human_review": False,
        "sample_size": 200, "pass": 199, "fix": 1,
        "report_file": "visual_review_report.json", "report_sha256": digest("visual_review_report.json"),
        "review_csv_sha256": digest("manual_review.csv"),
        "finding_ids": visual["findings"],
    }
    manifest["manual_review"] = {"status": "human_review_not_performed", "reviewed": 0,
                                 "minimum_per_category": 50}
    manifest["mmmu_deduplication"] = {"status": "completed", "scope": mmmu["scope"],
                                      "report_file": "dedupe_report.json", "report_sha256": digest("dedupe_report.json"),
                                      "train_removed": 0, "holdout_removed": 0}
    manifest["internal_overlap"] = {"status": "completed", "report_file": "internal_overlap_extended_report.json",
                                    "report_sha256": digest("internal_overlap_extended_report.json"),
                                    "cross_split_exact_image_pairs": 0, "cross_split_near_image_pairs": 0,
                                    "cross_split_same_image_question_option_sets": 0}
    manifest["release_status"] = "draft_one_reviewed_pitch_question_needs_correction"
    correction_path = ROOT / "correction_report.json"
    if correction_path.exists():
        correction = json.loads(correction_path.read_text(encoding="utf-8"))
        if correction["item_id"] != "musixqa_train_03736":
            raise ValueError("Unexpected correction target")
        for name, details in correction["training_files"].items():
            if digest(name) != details["output_sha256"]:
                raise ValueError(f"Corrected data hash changed: {name}")
            manifest["data_sha256"][name] = digest(name)
        manifest["file_roles"].update({
            "train.reviewed.jsonl": "corrected deduplicated candidate pool",
            "train_recipe.reviewed.jsonl": "active 3,200-item weighted training subset after duplicate check and visual correction",
            "generation_audit.reviewed.jsonl": "generation audit with corrected output wording",
            "manual_review.reviewed.csv": "Codex visual review with the correction rechecked and recorded",
            "review.reviewed.html": "read-only review screen showing corrected question and verdict",
        })
        manifest["active_training_file"] = "train_recipe.reviewed.jsonl"
        manifest["active_holdout_file"] = "holdout.dedup.jsonl"
        manifest["content_correction"] = {"status": "applied_and_verified",
                                          "report_file": "correction_report.json",
                                          "report_sha256": digest("correction_report.json"),
                                          "corrected_item_id": correction["item_id"]}
        manifest["visual_review"]["status"] = "completed_and_finding_corrected"
        manifest["visual_review"]["corrected_pass"] = 200
        manifest["visual_review"]["remaining_fixes"] = 0
        manifest["release_status"] = "draft_after_codex_visual_review_and_correction"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("Manifest updated; release remains draft")


if __name__ == "__main__":
    main()
