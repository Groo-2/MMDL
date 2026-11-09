"""Data correctness checks; no network or GPU required."""

from collections import Counter
import json
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest

import build_musixqa as builder


class MusiXQATests(unittest.TestCase):
    def test_streaming_json_boundaries_and_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "qa.json"
            expected = [{"question": "a \\\" b", "doc_id": "0000000"}, {"answer": "음표"}]
            path.write_text(json.dumps(expected, ensure_ascii=False), encoding="utf-8")
            for chunk in (1, 2, 7, 50):
                self.assertEqual(list(builder.iter_array(path, chunk)), expected)
            for broken in ('[{"a": 1}', '[{},]', '[{} {}]', '[{}]x', '{}'):
                path.write_text(broken, encoding="utf-8")
                with self.assertRaises(ValueError):
                    list(builder.iter_array(path, 3))

    def test_key_boundaries_and_relative_keys(self):
        self.assertEqual(builder.key_count("B Major scale"), 5)
        self.assertEqual(builder.key_count("G# Minor scale"), 5)
        self.assertEqual(builder.key_count("Ab Major scale"), -4)
        for count in range(-7, 8):
            item = {"category": "KeySignature", "value": count, "gold": builder.key_text(count)}
            for seed in range(20):
                wrong = builder.distractors(item, random.Random(seed))
                self.assertEqual(len({item["gold"], *(v for v, _ in wrong)}), 4)
                if count:
                    self.assertIn(builder.key_text(-count), [v for v, _ in wrong])

    def test_pitch_errors_are_wrong_and_keep_sequence_length(self):
        for clef, gold in [("treble", "B4 B4 E5 E5 B4 B4 E5 G#4"),
                           ("bass", "G#3 G#2 G#2 B3 B2 B2 B2"),
                           ("treble", "C4")]:
            item = {"category": "PitchReading", "gold": gold, "clef": clef}
            combos = set()
            for seed in range(60):
                wrong = builder.distractors(item, random.Random(seed))
                kinds = [kind for _, kind in wrong]
                self.assertEqual(len(set(kinds)), 3)
                combos.add(tuple(sorted(kinds)))
                signatures = {tuple(map(builder.sounding_pitch, gold.split()))}
                for text, kind in wrong:
                    notes = text.split()
                    self.assertEqual(len(notes), len(gold.split()))
                    changed = sum(a != b for a, b in zip(notes, gold.split()))
                    if kind != "opposite_clef":
                        self.assertIn(changed, (1, 2))
                    signature = tuple(map(builder.sounding_pitch, notes))
                    self.assertNotIn(signature, signatures)
                    signatures.add(signature)
            self.assertGreater(len(combos), 1)
        self.assertEqual(builder.shift_note("E4", -12), "G2")
        self.assertEqual(builder.shift_note("G2", 12), "E4")

    def test_selection_balance_and_reproducibility(self):
        pools = {cat: [] for cat in builder.CATEGORIES}
        for cat in builder.CATEGORIES:
            for index in range(60):
                doc = f"{index:07d}"
                source = {"doc_id": doc, "question": "", "answer": ""}
                if cat == "KeySignature":
                    source.update(question="What scale is this music in?", answer="B Major scale")
                elif cat == "MeasureCount":
                    source.update(question="How many measures are in the music sheet?", answer="14 measures / bars")
                elif cat == "Clef":
                    source.update(question="What clef is included in the music sheet?",
                                  answer=list(builder.CLEFS)[index % 3])
                else:
                    clef = "treble" if index % 2 else "bass"
                    source.update(question=f"Please extract the pitch of all notes in the 2nd bar of the {clef} clef.",
                                  answer="C4 D4 E4")
                item = builder.candidate(source, "train", "fixture.json", index)
                item["identity"] = cat + doc
                pools[cat].append(item)
        selected = builder.choose(pools, 16)
        rows, audit = builder.make_rows(selected, "train", 42)
        self.assertEqual((rows, audit), builder.make_rows(builder.choose(pools, 16), "train", 42))
        self.assertEqual(len(rows), 64)
        self.assertLessEqual(max(Counter(r["source_doc_id"] for r in rows).values()), 2)
        for cat in builder.CATEGORIES:
            self.assertEqual(Counter(r["answer"] for r in rows if r["category"] == cat),
                             dict.fromkeys("ABCD", 4))
        for row, trace in zip(rows, audit):
            self.assertEqual(row["options"][builder.LETTERS.index(row["answer"])], trace["gold"])

        for index, item in enumerate(pools["KeySignature"]):
            item.update(value=index % 3, gold=builder.key_text(index % 3))
        balanced = builder.choose(pools, 16, balance_content=True)
        for cat in ("KeySignature", "Clef"):
            content_counts = Counter(item["gold"] for item in balanced[cat])
            self.assertEqual(len(content_counts), 3)
            self.assertLessEqual(max(content_counts.values()) - min(content_counts.values()), 1)

    def test_known_bar_reference_is_clarified_without_changing_answer_choices(self):
        source = {"doc_id": "0087904", "question": "Please extract the pitch of all notes in the 1st bar of the treble clef.",
                  "answer": "C6 G4 G4"}
        item = builder.candidate(source, "test", "test_qa_omr.json", 11185)
        item["identity"] = "fixed-source-identity"
        self.assertIn("bar 1", item["question"])
        selected = {cat: [] for cat in builder.CATEGORIES}
        selected["PitchReading"] = [item]
        rows, audit = builder.make_rows(selected, "holdout", 42)
        self.assertIn("first visible bar (printed bar number 2)", rows[0]["question"])
        selected["PitchReading"] = [{**item, "source_doc_id": "0087000"}]
        original_rows, _ = builder.make_rows(selected, "holdout", 42)
        self.assertEqual(rows[0]["options"], original_rows[0]["options"])
        self.assertEqual(rows[0]["answer"], original_rows[0]["answer"])
        self.assertEqual(audit[0]["source_question"], source["question"])
        report = builder.quality_summary(rows, audit)
        self.assertEqual(report["bar_reference_review_queue"][0]["status"], "known_reference_clarified")
        with self.assertRaisesRegex(ValueError, "unverified_bar_mapping"):
            builder.candidate({**source, "question": source["question"].replace("1st", "2nd")}, "test", "fixture", 0)
        with self.assertRaisesRegex(ValueError, "unverified_bar_mapping"):
            builder.candidate({**source, "doc_id": "0060047"}, "test", "fixture", 0)

    def test_verified_ninth_written_bar_uses_ordinal_reference(self):
        source = {"doc_id": "0027194",
                  "question": "Please extract the pitch of all notes in the 9th bar of the bass clef.",
                  "answer": "Eb3 Eb3 Eb3 Eb3 Gb3"}
        item = builder.candidate(source, "train", "train_qa_omr.json", 2037943)
        self.assertIn("bar 9", item["question"])
        wording = builder.output_question(item)
        self.assertIn("9th written bar of the bass clef", wording)
        self.assertIn("ignoring printed measure numbers", wording)
        with self.assertRaisesRegex(ValueError, "unverified_bar_mapping"):
            builder.candidate({**source, "question": source["question"].replace("9th", "8th")},
                              "train", "train_qa_omr.json", 2037943)

    def test_conflicting_source_labels_are_not_selected(self):
        with tempfile.TemporaryDirectory() as directory:
            base = {"doc_id": "0086400", "question": "How many measures are in the music sheet?"}
            Path(directory, "test_qa_simple.json").write_text(
                json.dumps([{**base, "answer": "14 measures / bars"},
                            {**base, "answer": "15 measures / bars"}]), encoding="utf-8")
            Path(directory, "test_qa_omr.json").write_text("[]", encoding="utf-8")
            pools, _, report = builder.collect(directory, "test", 4, 42)
            self.assertEqual(report["conflicting_questions"], 1)
            self.assertEqual(pools["MeasureCount"], [])

    def test_build_runs_validator_and_preserves_image_bytes(self):
        from PIL import Image

        with tempfile.TemporaryDirectory() as directory:
            source_dir = Path(directory) / "source"
            images = source_dir / "images"
            images.mkdir(parents=True)
            for split, offset in [("train", 0), ("test", 86400)]:
                simple, omr = [], []
                for index in range(12):
                    doc = f"{offset + index:07d}"
                    for question, answer in [
                        ("What scale is this music in?", "B Major scale"),
                        ("How many measures are in the music sheet?", "14 measures / bars"),
                        ("What clef is included in the music sheet?", list(builder.CLEFS)[index % 3]),
                    ]:
                        simple.append({"doc_id": doc, "question": question, "answer": answer})
                    clef = "treble" if index % 2 else "bass"
                    omr.append({"doc_id": doc,
                                "question": f"Please extract the pitch of all notes in the 2nd bar of the {clef} clef.",
                                "answer": "C4 D4 E4"})
                    # Synthetic fixtures only; never used in real dataset outputs.
                    Image.new("RGB", (8, 8), (index, offset % 255, 200)).save(images / f"{doc}.png")
                builder.write_json(source_dir / f"{split}_qa_simple.json", simple)
                builder.write_json(source_dir / f"{split}_qa_omr.json", omr)
            outputs = [Path(directory) / "sample1", Path(directory) / "sample2"]
            for out in outputs:
                builder.build(SimpleNamespace(
                    source_dir=str(source_dir), out_dir=str(out), seed=42,
                    train_per_category=4, holdout_per_category=4,
                    validator=str(Path(__file__).resolve().parents[1] / "lab/validate_mcq_dataset.py"),
                ))
                report = json.loads((out / "validation_report.json").read_text(encoding="utf-8"))
                self.assertEqual(report["error_count"], 0)
                self.assertFalse((out / "BUILD_INCOMPLETE.json").exists())
                for image in (out / "images").iterdir():
                    self.assertEqual(image.read_bytes(), (images / image.name).read_bytes())
            for split in ("train", "holdout"):
                self.assertEqual((outputs[0] / f"{split}.jsonl").read_bytes(),
                                 (outputs[1] / f"{split}.jsonl").read_bytes())


if __name__ == "__main__":
    unittest.main()
