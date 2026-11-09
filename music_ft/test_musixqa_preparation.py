"""Protect reference coverage and reproducible, unreviewed review samples."""

from collections import Counter
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

import dedupe
import prepare_musixqa_review as review


def image_bytes():
    stream = io.BytesIO()
    Image.new("RGB", (64, 64), "white").save(stream, format="PNG")
    return stream.getvalue()


class PreparationTests(unittest.TestCase):
    def test_review_sample_is_repeatable_subset_with_both_splits(self):
        rows = [{"id": f"{cat}/{split}/{n}", "category": cat, "split": split}
                for cat in review.CATEGORIES for split in ("train", "holdout") for n in range(100)]
        selected = review.select_review_rows(rows, 50, 10, 42)
        self.assertEqual(selected, review.select_review_rows(list(reversed(rows)), 50, 10, 42))
        self.assertEqual(len({r["id"] for r in selected}), 200)
        counts = Counter((r["category"], r["split"]) for r in selected)
        for cat in review.CATEGORIES:
            self.assertEqual(counts[cat, "train"], 40)
            self.assertEqual(counts[cat, "holdout"], 10)
        with self.assertRaises(ValueError):
            review.select_review_rows(rows[:20], 50, 10, 42)

    def test_incomplete_reference_fails_instead_of_claiming_clean(self):
        with tempfile.TemporaryDirectory() as folder:
            with patch.object(dedupe, "snapshot_download", return_value=folder):
                with self.assertRaisesRegex(ValueError, "Incomplete"):
                    dedupe.load_mmmu_music(None)

    def test_exact_reference_image_removed_and_audited(self):
        data = image_bytes()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "image.png").write_bytes(data)
            for split in ("train", "holdout"):
                row = {"id": split, "image_paths": {"image_1": "image.png"},
                       "question": "fixture", "options": ["a", "b", "c", "d"]}
                (root / f"{split}.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
            refs = [{"id": "reference", "split": "test", "question": "different",
                     "options": ["e", "f", "g", "h"], "images": [data]}]
            with patch.object(dedupe, "load_mmmu_music", return_value=refs), patch("sys.argv", ["dedupe", "--data-dir", folder]):
                dedupe.main()
                with self.assertRaises(FileExistsError):
                    dedupe.main()
            report = json.loads((root / "dedupe_report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["train"]["dropped_items"][0]["reason"], "sha1:image_1")
            self.assertEqual(report["holdout"]["kept"], 0)
            self.assertEqual((root / "train.dedup.jsonl").read_text(), "")
        self.assertEqual(dedupe.parse_options('["a singer\'s voice", "piano"]'), ["a singer's voice", "piano"])

    def test_review_does_not_mark_pass_or_overwrite_records(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "image.png").write_bytes(image_bytes())
            audits = []
            for split in ("train", "holdout"):
                row = {"id": split, "category": "Clef", "source_doc_id": split,
                       "image_paths": {"image_1": "image.png"}, "question": "Which clef?",
                       "options": ["treble", "bass", "both", "alto"], "answer": "A"}
                (root / f"{split}.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
                audits.append({"id": split, "gold": "treble", "distractors": [],
                               "source_question": "original", "source_answer": "treble",
                               "source_file": "fixture", "source_index": 0})
            (root / "generation_audit.jsonl").write_text("\n".join(map(json.dumps, audits)), encoding="utf-8")
            review.prepare(root)
            csv_path = root / "manual_review.csv"
            with csv_path.open(encoding="utf-8-sig", newline="") as stream:
                self.assertTrue(all(r["verdict"] == "pending" for r in csv.DictReader(stream)))
            saved = csv_path.read_bytes()
            review.prepare(root)
            self.assertEqual(saved, csv_path.read_bytes())
            source = root / "train.jsonl"
            source.write_text(source.read_text(encoding="utf-8").replace("Which clef?", "Changed question?"), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "inputs or selection changed"):
                review.prepare(root)
            self.assertEqual(saved, csv_path.read_bytes())


if __name__ == "__main__":
    unittest.main()
