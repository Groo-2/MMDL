"""Quota, answer-mapping, and source-subset invariants for the pilot recipe."""

from collections import Counter
import copy
import random
import unittest

import build_musixqa as base
import build_musixqa_recipe as recipe_builder


def fixture_pools():
    result = {cat: [] for cat in base.CATEGORIES}
    for category_index, category in enumerate(base.CATEGORIES):
        for i in range(1200):
            item = {
                "source_doc_id": f"{category_index * 2000 + i:07d}", "source_split": "train",
                "source_file": "fixture.json", "source_index": i, "source_question": "fixture",
                "source_answer": "fixture", "identity": f"{category}/{i}",
                "category": category, "question": "Fixture question?",
            }
            if category == "KeySignature":
                value = i % 15 - 7
                item.update(value=value, gold=base.key_text(value))
            elif category == "MeasureCount":
                item.update(value=9 + i % 11, gold=str(9 + i % 11))
            elif category == "Clef":
                item["gold"] = base.CLEF_OPTIONS[i % 3]
            else:
                n = i % 8 + 1
                clef = "treble" if i // 8 % 2 else "bass"
                item.update(gold=" ".join((["C4", "D4", "Eb4"] * 3)[:n]), clef=clef,
                            source_question=f"Please extract the pitch of all notes in the 2nd bar of the {clef} clef.")
            result[category].append(item)
    return result


class RecipeTests(unittest.TestCase):
    def test_letter_matrix_balances_content_and_overall(self):
        rng = random.Random(42)
        for _ in range(100):
            sizes = {str(i): rng.randrange(1, 100) for i in range(rng.randrange(1, 30))}
            sizes["0"] += (-sum(sizes.values())) % 4
            matrix = recipe_builder.letter_quotas(sizes, 42)
            self.assertEqual({key: sum(values.values()) for key, values in matrix.items()}, sizes)
            self.assertEqual({letter: sum(row[letter] for row in matrix.values()) for letter in "ABCD"},
                             dict.fromkeys("ABCD", sum(sizes.values()) // 4))
            self.assertTrue(all(max(row.values()) - min(row.values()) <= 1 for row in matrix.values()))

    def test_pool_and_weighted_subset_preserve_original_rows(self):
        config = {
            "seed": 42, "pitch_train_length_weights": {"1-3": 30, "4-6": 45, "7-8": 25},
            "pitch_errors": {"staff_steps": [-2, -1, 1, 2], "require_staff_error": True},
            "training_counts": {"PitchReading": 192, "KeySignature": 80, "MeasureCount": 32, "Clef": 16},
        }
        pool_counts = {"PitchReading": 240, "KeySignature": 120, "MeasureCount": 88, "Clef": 120}
        pools = fixture_pools()
        selected = recipe_builder.select_candidates(pools, pool_counts, config["pitch_train_length_weights"])
        rows, audit = recipe_builder.render(selected, "train", config)
        recipe_builder.verify(rows, pool_counts, audit)
        by_id = {row["id"]: copy.deepcopy(row) for row in rows}
        subset = recipe_builder.select_training_subset(rows, audit, config)
        recipe_builder.verify(subset, config["training_counts"], audit)
        self.assertEqual(len(subset), 320)
        self.assertTrue(all(row == by_id[row["id"]] for row in subset))
        self.assertEqual(subset, recipe_builder.select_training_subset(rows, audit, config))
        traces = {item["id"]: item for item in audit}
        pitches = [traces[row["id"]] for row in subset if row["category"] == "PitchReading"]
        self.assertEqual(Counter(item["clef"] for item in pitches), {"treble": 96, "bass": 96})
        self.assertEqual(Counter(recipe_builder.length_bin(item["gold"]) for item in pitches),
                         recipe_builder.weighted(192, config["pitch_train_length_weights"]))
        self.assertTrue(all("staff_position" in {d["type"] for d in item["distractors"]} for item in pitches))
        with self.assertRaisesRegex(ValueError, "Insufficient retained pool"):
            recipe_builder.select_training_subset([row for row in rows if row["category"] != "Clef"], audit, config)

    def test_diagnostic_holdout_keeps_equal_categories(self):
        counts = dict.fromkeys(base.CATEGORIES, 100)
        selected = recipe_builder.select_candidates(fixture_pools(), counts)
        self.assertEqual({cat: len(items) for cat, items in selected.items()}, counts)
        self.assertEqual(Counter(item["clef"] for item in selected["PitchReading"]), {"treble": 50, "bass": 50})
        self.assertEqual(set(item["gold"] for item in selected["KeySignature"]), {base.key_text(n) for n in range(-7, 8)})

    def test_staff_errors_include_pilot_two_step_confusion(self):
        item = {"category": "PitchReading", "gold": "E4", "clef": "treble"}
        wrong_values = set()
        for seed in range(80):
            wrong = base.pitch_distractors(item, random.Random(seed), (-2, -1, 1, 2), True)
            staff = [text for text, kind in wrong if kind == "staff_position"]
            self.assertEqual(len(staff), 1)
            wrong_values.update(staff)
        self.assertIn("G4", wrong_values)
        self.assertNotIn("E4", wrong_values)

    def test_single_flat_note_distractors_do_not_collide_enharmonically(self):
        for note in ("Bb3", "Cb5", "Fb4", "B#4"):
            for seed in range(60):
                item = {"category": "PitchReading", "gold": note, "clef": "treble"}
                wrong = base.pitch_distractors(item, random.Random(seed), (-2, -1, 1, 2), True)
                signatures = {base.sounding_pitch(note), *(base.sounding_pitch(text) for text, _ in wrong)}
                self.assertEqual(len(signatures), 4)

    def test_key_policy_has_distinct_options_and_zero_as_a_distractor(self):
        zero_wrong = 0
        for value in range(-7, 8):
            item = {"category": "KeySignature", "value": value, "gold": base.key_text(value)}
            for seed in range(60):
                wrong = base.distractors(item, random.Random(seed), key_count_pairs=True)
                options = [item["gold"], *(text for text, _ in wrong)]
                self.assertEqual(len(set(options)), 4)
                if value:
                    self.assertIn(base.key_text(-value), options)
                    zero_wrong += base.key_text(0) in options
                if abs(value) >= 3:
                    magnitudes = Counter(int(option.split()[0]) for option in options)
                    self.assertEqual(sorted(magnitudes.values()), [2, 2])
        self.assertGreater(zero_wrong, 0)


if __name__ == "__main__":
    unittest.main()
