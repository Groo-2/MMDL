"""Build a balanced candidate pool and a pilot-informed training subset, without training."""

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import random
import shutil
import subprocess
import sys

import build_musixqa as base


def even(total, labels):
    return {label: total // len(labels) + (i < total % len(labels)) for i, label in enumerate(labels)}


def weighted(total, weights):
    denominator = sum(weights.values())
    result = {key: total * value // denominator for key, value in weights.items()}
    order = sorted(weights, key=lambda key: (-(total * weights[key] % denominator), key))
    for key in order[:total - sum(result.values())]:
        result[key] += 1
    return result


def length_bin(gold):
    n = len(gold.split())
    return "1-3" if n <= 3 else "4-6" if n <= 6 else "7-8"


def stratum(item):
    if item["category"] == "PitchReading":
        return item["clef"] + "/" + length_bin(item["gold"])
    return item["gold"]


def quotas(category, count, pitch_weights=None):
    if category == "KeySignature":
        return even(count, [base.key_text(n) for n in range(-7, 8)])
    if category == "Clef":
        return even(count, base.CLEF_OPTIONS[:3])
    if category == "MeasureCount":
        # Nine-measure sheets are extremely rare (14/9600 test documents).
        # There is no pilot evidence for oversampling that edge case to 1/11.
        return {"__all__": count}
    if pitch_weights is None:
        return even(count, ["treble", "bass"])
    lengths = weighted(count, pitch_weights)
    treble = {key: value // 2 for key, value in lengths.items()}
    missing = (count + 1) // 2 - sum(treble.values())
    for key in [key for key in lengths if lengths[key] % 2][:missing]:
        treble[key] += 1
    return {f"{clef}/{key}": treble[key] if clef == "treble" else value - treble[key]
            for clef in ("treble", "bass") for key, value in lengths.items()}


def letter_quotas(sizes, seed):
    """Exact overall A-D balance plus <=1 difference inside each stratum."""
    total = sum(sizes.values())
    if total % 4:
        raise ValueError("Category count must be divisible by four")
    result = {key: dict.fromkeys(base.LETTERS, size // 4) for key, size in sizes.items()}
    remaining = {letter: total // 4 - sum(row[letter] for row in result.values()) for letter in base.LETTERS}
    rng = random.Random(str(seed))
    for key in sorted(sizes):
        letters = list(base.LETTERS)
        rng.shuffle(letters)
        letters.sort(key=lambda letter: remaining[letter], reverse=True)
        for letter in letters[:sizes[key] % 4]:
            result[key][letter] += 1
            remaining[letter] -= 1
    if any(remaining.values()):
        raise ValueError("Could not allocate answer-position quotas")
    return result


def select_candidates(pools, counts, pitch_weights=None):
    selected = {category: [] for category in base.CATEGORIES}
    used, per_doc = set(), Counter()
    for category in ("PitchReading", "KeySignature", "MeasureCount", "Clef"):
        groups = quotas(category, counts[category], pitch_weights)
        for group, count in groups.items():
            picked = 0
            for item in pools[category]:
                if picked == count:
                    break
                key = ("__all__" if category == "MeasureCount" else
                       item["clef"] if category == "PitchReading" and pitch_weights is None else stratum(item))
                if key != group or item["identity"] in used or per_doc[item["source_doc_id"]] >= 2:
                    continue
                selected[category].append(item)
                used.add(item["identity"])
                per_doc[item["source_doc_id"]] += 1
                picked += 1
            if picked != count:
                raise ValueError(f"Insufficient candidates for {category}/{group}: {picked}/{count}")
    return selected


def balance_positions(rows, audit, seed):
    by_id = {item["id"]: item for item in audit}
    for category in base.CATEGORIES:
        groups = defaultdict(list)
        for row in rows:
            if row["category"] == category:
                groups[stratum(by_id[row["id"]])].append(row)
        matrix = letter_quotas({key: len(value) for key, value in groups.items()}, f"{seed}/{category}")
        for key, group in groups.items():
            positions = [letter for letter, count in matrix[key].items() for _ in range(count)]
            random.Random(f"{seed}/{category}/{key}").shuffle(positions)
            for row, letter in zip(group, positions):
                gold = row["options"][base.LETTERS.index(row["answer"])]
                wrong = [option for i, option in enumerate(row["options"]) if i != base.LETTERS.index(row["answer"])]
                wrong.insert(base.LETTERS.index(letter), gold)
                row.update(options=wrong, answer=letter)
                by_id[row["id"]].update(output_answer=letter, position_balance_group=key)


def render(selected, split, recipe):
    error = recipe["pitch_errors"]
    rows, audit = base.make_rows(selected, split, recipe["seed"],
                                 staff_steps=tuple(error["staff_steps"]), require_staff_error=error["require_staff_error"],
                                 key_count_pairs=recipe.get("key_count_pairs", False))
    traces = {item["id"]: item for item in audit}
    for row in rows:
        if row["category"] == "MeasureCount":
            row["question"] = "<image 1> How many measures are written on this music sheet? Count each written measure once, without repeating sections."
        elif row["category"] == "PitchReading":
            row["question"] += " Count each written note, including notes joined by ties."
        traces[row["id"]]["output_question"] = row["question"].removeprefix("<image 1> ")
    balance_positions(rows, audit, recipe["seed"])
    return rows, audit


def select_training_subset(rows, audit, recipe):
    traces = {item["id"]: item for item in audit}
    selected = []
    for category in base.CATEGORIES:
        counts = quotas(category, recipe["training_counts"][category], recipe["pitch_train_length_weights"])
        matrix = letter_quotas(counts, f"{recipe['seed']}/{category}/subset")
        candidates = [row for row in rows if row["category"] == category]
        candidates.sort(key=lambda row: hashlib.sha256(f"{recipe['seed']}/{row['id']}/subset".encode()).digest())
        for group, letters in matrix.items():
            for letter, count in letters.items():
                eligible = [row for row in candidates if (category == "MeasureCount" or stratum(traces[row["id"]]) == group) and row["answer"] == letter]
                if len(eligible) < count:
                    raise ValueError(f"Insufficient retained pool for {category}/{group}/{letter}: {len(eligible)}/{count}; replenish candidates before training")
                selected.extend(eligible[:count])
    random.Random(recipe["seed"]).shuffle(selected)
    return selected


def verify(rows, expected, audit):
    if Counter(row["category"] for row in rows) != expected:
        raise ValueError("Category quotas do not match")
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate IDs")
    if max(Counter(row["source_doc_id"] for row in rows).values()) > 2:
        raise ValueError("More than two questions per document")
    traces = {item["id"]: item for item in audit}
    for category in base.CATEGORIES:
        group = [row for row in rows if row["category"] == category]
        if Counter(row["answer"] for row in group) != dict.fromkeys(base.LETTERS, len(group) // 4):
            raise ValueError(f"Answer letters are imbalanced in {category}")
        if category in ("KeySignature", "Clef"):
            counts = Counter(traces[row["id"]]["gold"] for row in group)
            desired = quotas(category, len(group))
            if counts != desired:
                raise ValueError(f"Content quota mismatch in {category}")
        for row in group:
            if row["options"][base.LETTERS.index(row["answer"])] != traces[row["id"]]["gold"]:
                raise ValueError(f"Answer remapping failed: {row['id']}")


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def build(args, recipe):
    out = Path(args.out_dir)
    if out.exists():
        raise FileExistsError(f"Output exists, use another --out-dir: {out}")
    evidence = Path(recipe["evidence_file"])
    if not evidence.exists():
        evidence = Path(args.recipe).parent / evidence.name
    evidence_data = json.loads(evidence.read_text(encoding="utf-8"))
    datasets, audits, reports, docs = {}, {}, {}, {}
    for source_split, split, counts in [("train", "train", recipe["candidate_train_counts"]),
                                        ("test", "holdout", recipe["holdout_counts"])]:
        print("Collecting source candidates:", source_split, flush=True)
        pools, docs[split], reports[split] = base.collect(args.source_dir, source_split, max(counts.values()), recipe["seed"])
        selected = select_candidates(pools, counts, recipe["pitch_train_length_weights"] if split == "train" else None)
        datasets[split], audits[split] = render(selected, split, recipe)
        verify(datasets[split], counts, audits[split])
        print("Selected:", split, dict(Counter(row["category"] for row in datasets[split])), flush=True)
    if docs["train"] & docs["holdout"]:
        raise ValueError("Source documents overlap between splits")
    subset = select_training_subset(datasets["train"], audits["train"], recipe)
    verify(subset, recipe["training_counts"], audits["train"])
    out.mkdir(parents=True)
    base.write_json(out / "BUILD_INCOMPLETE.json", {"status": "in_progress"})
    all_rows = datasets["train"] + datasets["holdout"]
    all_audit = audits["train"] + audits["holdout"]
    print("Extracting original PNGs for", len({row['source_doc_id'] for row in all_rows}), "documents", flush=True)
    image_hashes = base.extract_images(args.source_dir, out / "images", {row["source_doc_id"] for row in all_rows})
    for split, rows in datasets.items():
        base.save_jsonl(out / f"{split}.jsonl", rows)
    base.save_jsonl(out / "train_recipe.jsonl", subset)
    base.save_jsonl(out / "generation_audit.jsonl", all_audit)
    for name, rows in [("pool", all_rows), ("recipe", subset + datasets["holdout"])]:
        base.write_json(out / f"quality_{name}.json", base.quality_summary(rows, all_audit))
    base.write_json(out / "recipe.json", recipe)
    shutil.copyfile(evidence, out / evidence.name)
    for path in [Path(__file__), Path(base.__file__), Path(args.validator)]:
        shutil.copyfile(path, out / path.name)
    manifest = {
        "version": recipe["name"], "created_at_utc": datetime.now(timezone.utc).isoformat(), "seed": recipe["seed"],
        "source": {"repo": base.REPO, "revision": base.REVISION, "license": "MIT",
                   "qa_sha256": {p.name: base.sha256(p) for split in ("train", "test") for p in base.sources(args.source_dir, split)}},
        "pilot_report_sha256": evidence_data["source_sha256"], "recipe_file": "recipe.json",
        "counts": {"candidate_train": recipe["candidate_train_counts"], "training_subset": recipe["training_counts"], "holdout": recipe["holdout_counts"]},
        "file_roles": {"train.jsonl": "candidate pool, not the weighted training subset", "train_recipe.jsonl": "provisional weighted subset before deduplication/review", "holdout.jsonl": "fixed balanced diagnostic evaluation set"},
        "data_sha256": {name: base.sha256(out / name) for name in ("train.jsonl", "train_recipe.jsonl", "holdout.jsonl")},
        "image_sha256": image_hashes, "source_inspection": reports,
        "python": sys.version, "packages": {p: importlib.metadata.version(p) for p in ("music21", "Pillow")},
        "team_sample_review": "waived_by_user", "manual_review": {"status": "pending", "reviewed": 0, "minimum_per_category": 50},
        "mmmu_deduplication": {"status": "pending"}, "image_token_measurement": {"status": "pending"},
        "training_status": "not_started", "release_status": "draft_before_deduplication_and_content_review",
        "limitations": evidence_data["interpretation_notes"] + [
            "Clef has only three observed correct classes. Its impossible alto option remains a shortcut; lowering its share does not remove that shortcut.",
            "An accidental in a pitch label does not identify printed natural-sign cancellation; no guaranteed natural-sign training coverage is claimed.",
            "Known bar-number registry is not whole-dataset OCR. Review the bar-reference queue and check printed numbering.",
        ],
    }
    base.write_json(out / "manifest.json", manifest)
    print("Running structural validator", flush=True)
    subprocess.run([sys.executable, "-X", "utf8", args.validator, "--data-dir", str(out),
                    "--min-train-per-category", "800", "--min-holdout-per-category", "100"], check=True)
    validation = json.loads((out / "validation_report.json").read_text(encoding="utf-8"))
    manifest["validation"] = {"errors": validation["error_count"], "warnings": validation["warnings"]}
    base.write_json(out / "manifest.json", manifest)
    (out / "BUILD_INCOMPLETE.json").unlink()
    print("Created recipe draft:", out, flush=True)
    print("Training subset:", len(subset), "Holdout:", len(datasets["holdout"]), flush=True)


def remix(args, recipe):
    root = Path(args.data_dir)
    out = root / args.output_file
    if out.exists():
        raise FileExistsError(out)
    rows = read_jsonl(root / args.input_file)
    audit = read_jsonl(root / "generation_audit.jsonl")
    subset = select_training_subset(rows, audit, recipe)
    verify(subset, recipe["training_counts"], audit)
    base.save_jsonl(out, subset)
    base.write_json(out.with_suffix(".report.json"), {
        "input_file": args.input_file, "input_sha256": base.sha256(root / args.input_file),
        "output_sha256": base.sha256(out), "counts": dict(Counter(row["category"] for row in subset)),
        "recipe": recipe, "note": "Subset of the supplied pool only. This command does not itself verify deduplication or manual-review completion.",
    })
    print("Wrote:", out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("build", "remix"):
        p = sub.add_parser(name)
        p.add_argument("--recipe", default="music_ft/musixqa_pilot20_recipe.json")
        if name == "build":
            p.add_argument("--source-dir", default="data/musixqa_source")
            p.add_argument("--out-dir", default="data/musixqa_mcq_pilot20_v1")
            p.add_argument("--validator", default="lab/validate_mcq_dataset.py")
        else:
            p.add_argument("--data-dir", default="data/musixqa_mcq_pilot20_v1")
            p.add_argument("--input-file", default="train.dedup.jsonl")
            p.add_argument("--output-file", default="train_recipe.dedup.jsonl")
    args = parser.parse_args()
    recipe = json.loads(Path(args.recipe).read_text(encoding="utf-8"))
    for name in ("candidate_train_counts", "training_counts", "holdout_counts"):
        if set(recipe[name]) != set(base.CATEGORIES) or any(not isinstance(n, int) or n <= 0 or n % 4 for n in recipe[name].values()):
            raise ValueError(f"Invalid category counts: {name}")
    if any(recipe["training_counts"][cat] > recipe["candidate_train_counts"][cat] for cat in base.CATEGORIES):
        raise ValueError("Training quota exceeds candidate pool")
    {"build": build, "remix": remix}[args.command](args, recipe)


if __name__ == "__main__":
    main()
