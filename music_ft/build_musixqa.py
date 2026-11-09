"""Pinned MusiXQA downloads, source inspection, and deterministic Tier-1 MCQs.

Use --help and lab/RUNBOOK_musixqa_data.md. The default build is an 80-item
sample for team review, not a full training dataset. No model or GPU is used.
"""

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import heapq
import importlib.metadata
import json
from pathlib import Path
import random
import re
import shutil
import subprocess
import sys
import tarfile

REPO = "puar-playground/MusiXQA"
REVISION = "d4cae4a5404b9c5d6dc8b85cce4e4c5d3585aae6"
CATEGORIES = ("KeySignature", "MeasureCount", "PitchReading", "Clef")
LETTERS = "ABCD"
PITCH_Q = re.compile(
    r"Please extract the pitch of all notes in the (\d+)(?:st|nd|rd|th) bar of the (treble|bass) clef\."
)
NOTE = re.compile(r"([A-G])([#b]?)([0-8])")
CLEFS = {
    "The music sheet includes both treble and bass clefs.": "both treble and bass clefs",
    "The music sheet only has a treble clef.": "treble clef only",
    "The music sheet only has a bass clef.": "bass clef only",
}
CLEF_OPTIONS = ["treble clef only", "bass clef only", "both treble and bass clefs", "alto clef only"]
# Visual findings from the 2026-11-08 sample review. This is a verified-document
# registry, not an OCR detector or a general +1 bar-index mapping.
BAR_NUMBER_REVIEW = {
    "0019023": {"first_printed_bar": 2, "first_bar_pitch_verified": True},
    "0087904": {"first_printed_bar": 2, "first_bar_pitch_verified": True},
    "0060047": {"first_printed_bar": 2, "first_bar_pitch_verified": False},
    "0027194": {"first_printed_bar": 2, "first_bar_pitch_verified": False,
                "verified_ordinal_bars": [9]},
}


def output_question(item):
    """Keep the source identity/RNG stable; change only the displayed reference."""
    if item["category"] == "PitchReading":
        match = PITCH_Q.fullmatch(item["source_question"])
        finding = BAR_NUMBER_REVIEW.get(item["source_doc_id"])
        if finding and match and int(match[1]) in finding.get("verified_ordinal_bars", []):
            return (f"Which sequence of pitches is written in the {match[1]}th written bar "
                    f"of the {match[2]} clef, counting from the beginning of this sheet "
                    "and ignoring printed measure numbers?")
        if finding and match and int(match[1]) == 1 and finding["first_bar_pitch_verified"]:
            return ("Which sequence of pitches is written in the first visible bar "
                    f"(printed bar number {finding['first_printed_bar']}) of the {match[2]} clef?")
    return item["question"]


def quality_summary(rows, audit):
    """Content-frequency and reference risks; this does not inspect pixels."""
    traces = {item["id"]: item for item in audit}
    distributions = {}
    for split in ("all", "train", "holdout"):
        selected = rows if split == "all" else [r for r in rows if r["source_split"] == ("test" if split == "holdout" else "train")]
        distributions[split] = {}
        for cat in CATEGORIES:
            group = [r for r in selected if r["category"] == cat]
            values = Counter(r["options"][LETTERS.index(r["answer"])] for r in group)
            distributions[split][cat] = {
                "n": len(group), "answer_letters": dict(Counter(r["answer"] for r in group)),
                "correct_option_counts": dict(sorted(values.items())),
                "largest_correct_option_fraction": max(values.values(), default=0) / len(group) if group else 0,
            }
    references = []
    for row in rows:
        trace = traces[row["id"]]
        match = PITCH_Q.fullmatch(trace["source_question"]) if row["category"] == "PitchReading" else None
        if match and (int(match[1]) == 1 or row["source_doc_id"] in BAR_NUMBER_REVIEW):
            known = BAR_NUMBER_REVIEW.get(row["source_doc_id"])
            references.append({"id": row["id"], "source_doc_id": row["source_doc_id"],
                               "source_bar": int(match[1]), "known_document_finding": known,
                               "status": "known_reference_clarified" if known and row["question"] == "<image 1> " + output_question(trace)
                               else "image_review_needed"})
    clef = [row for row in rows if row["category"] == "Clef"]
    clef_gold = {r["options"][LETTERS.index(r["answer"])] for r in clef}
    never_correct = sorted({option for r in clef for option in r["options"]} - clef_gold)
    return {
        "distributions": distributions, "bar_reference_review_queue": references,
        "clef_options_never_correct_in_this_dataset": never_correct,
        "scope": "Label/QA checks and known-document registry only; no image OCR. Absence in this sample alone does not prove absence in the source.",
    }


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def iter_array(path, chunk_size=1024 * 1024):
    """Read a JSON array incrementally; train_qa_omr.json is over 1 GB."""
    decoder = json.JSONDecoder()
    with Path(path).open(encoding="utf-8-sig") as stream:
        buffer, pos, eof = "", 0, False

        def available():
            nonlocal buffer, pos, eof
            while True:
                while pos < len(buffer) and buffer[pos].isspace():
                    pos += 1
                if pos < len(buffer) or eof:
                    return pos < len(buffer)
                buffer = stream.read(chunk_size)
                pos = 0
                eof = not buffer

        if not available() or buffer[pos] != "[":
            raise ValueError(f"{path}: expected a JSON array")
        pos += 1
        first = True
        while True:
            if not available():
                raise ValueError(f"{path}: incomplete JSON array")
            if buffer[pos] == "]":
                pos += 1
                if available():
                    raise ValueError(f"{path}: trailing content")
                return
            if not first:
                if buffer[pos] != ",":
                    raise ValueError(f"{path}: missing comma")
                pos += 1
                if not available() or buffer[pos] == "]":
                    raise ValueError(f"{path}: missing array item")
            while True:
                try:
                    item, end = decoder.raw_decode(buffer, pos)
                    break
                except json.JSONDecodeError:
                    extra = stream.read(chunk_size)
                    if not extra:
                        raise ValueError(f"{path}: incomplete or invalid JSON item") from None
                    buffer, pos = buffer[pos:] + extra, 0
            pos = end
            if not isinstance(item, dict):
                raise ValueError(f"{path}: QA item must be an object")
            yield item
            first = False


@lru_cache(maxsize=128)
def key_count(answer):
    from music21 import key

    match = re.fullmatch(r"([A-G][#b]?) (Major|Minor) scale", answer)
    if not match:
        raise ValueError("unrecognized_key_label")
    count = key.Key(match[1].replace("b", "-"), match[2].lower()).sharps
    if not -7 <= count <= 7:
        raise ValueError("key_outside_range")
    return count


def key_text(count):
    if count == 0:
        return "no sharps or flats"
    kind = "sharp" if count > 0 else "flat"
    return f"{abs(count)} {kind}{'s' if abs(count) > 1 else ''}"


def candidate(row, source_split, source_file, index):
    question, answer = row.get("question"), row.get("answer")
    if not isinstance(question, str) or not isinstance(answer, str):
        raise ValueError("invalid_question_or_answer")
    doc = row.get("doc_id")
    if not isinstance(doc, str) or not re.fullmatch(r"\d{7}", doc):
        raise ValueError("invalid_doc_id")
    result = {
        "source_doc_id": doc, "source_split": source_split,
        "source_file": source_file, "source_index": index,
        "source_question": question, "source_answer": answer,
    }
    if question == "What scale is this music in?":
        count = key_count(answer)
        result.update(category="KeySignature", gold=key_text(count), value=count,
                      question="How many sharps or flats are in the key signature of this piece?")
    elif question == "How many measures are in the music sheet?":
        match = re.fullmatch(r"(\d+) measures / bars", answer)
        if not match or not 9 <= int(match[1]) <= 19:
            raise ValueError("invalid_measure_count")
        count = int(match[1])
        result.update(category="MeasureCount", gold=str(count), value=count,
                      question="How many measures are in this music sheet?")
    elif question == "What clef is included in the music sheet?":
        if answer not in CLEFS:
            raise ValueError("unrecognized_clef_label")
        result.update(category="Clef", gold=CLEFS[answer],
                      question="Which clef(s) does this music sheet use?")
    elif match := PITCH_Q.fullmatch(question):
        finding = BAR_NUMBER_REVIEW.get(doc)
        verified = ((int(match[1]) == 1 and finding["first_bar_pitch_verified"])
                    or int(match[1]) in finding.get("verified_ordinal_bars", [])) if finding else True
        if not verified:
            raise ValueError("unverified_bar_mapping_in_known_document")
        notes = answer.split()
        if not notes or not all(NOTE.fullmatch(n) for n in notes):
            raise ValueError("unsupported_pitch_notation")
        if len(notes) >= 9:
            raise ValueError("pitch_nine_or_more_notes")
        result.update(category="PitchReading", gold=" ".join(notes), clef=match[2],
                      question=f"Which sequence of pitches is written in bar {int(match[1])} of the {match[2]} clef?")
    else:
        return None
    return result


def sources(source_dir, split):
    return [Path(source_dir) / f"{split}_qa_{kind}.json" for kind in ("simple", "omr")]


def collect(source_dir, split, target, seed):
    """Keep a deterministic, randomly ranked pool without loading all QA rows."""
    pools = defaultdict(list)
    seen, conflicts, docs = {}, set(), set()
    counts, rejected = Counter(), Counter()
    limit = max(1000, target * 20)
    for path in sources(source_dir, split):
        if not path.is_file():
            raise FileNotFoundError(f"Missing {path}; run the download command first")
        for index, row in enumerate(iter_array(path)):
            if isinstance(row.get("doc_id"), str):
                docs.add(row["doc_id"])
            try:
                item = candidate(row, split, path.name, index)
            except ValueError as exc:
                rejected[str(exc)] += 1
                continue
            if item is None:
                continue
            ident = hashlib.sha256((item["source_doc_id"] + "\0" + item["question"]).encode()).digest()
            answer_hash = hashlib.sha256(item["gold"].encode()).digest()
            if ident in seen:
                if seen[ident] != answer_hash:
                    conflicts.add(ident)
                else:
                    rejected["duplicate_qa"] += 1
                continue
            seen[ident] = answer_hash
            counts[item["category"]] += 1
            rank = int.from_bytes(hashlib.sha256(str(seed).encode() + ident).digest(), "big")
            item["identity"] = ident.hex()
            entry = (-rank, ident, item)
            heap = pools[item["category"]]
            if len(heap) < limit:
                heapq.heappush(heap, entry)
            elif entry[:2] > heap[0][:2]:
                heapq.heapreplace(heap, entry)
    available = {
        cat: [entry[2] for entry in sorted(pools[cat], key=lambda e: (-e[0], e[1]))
              if entry[1] not in conflicts]
        for cat in CATEGORIES
    }
    return available, docs, {
        "source_split": split, "documents": len(docs),
        "eligible_before_conflict_removal": dict(counts), "rejected": dict(rejected),
        "conflicting_questions": len(conflicts), "pool_limit_per_category": limit,
    }


def choose(pools, count, balance_content=False):
    per_doc, used = Counter(), set()
    selected = {cat: [] for cat in CATEGORIES}
    # Pick pitch questions first, keeping both clefs represented.
    for cat in ("PitchReading", "KeySignature", "MeasureCount", "Clef"):
        groups = [(None, count)]
        if cat == "PitchReading":
            groups = [("treble", (count + 1) // 2), ("bass", count // 2)]
        elif balance_content and cat in ("KeySignature", "Clef"):
            labels = sorted({item["gold"] for item in pools[cat]})
            if not labels:
                raise ValueError(f"No labels available for {cat}")
            groups = [(label, count // len(labels) + (i < count % len(labels))) for i, label in enumerate(labels)]
        for label, quota in groups:
            added = 0
            for item in pools[cat]:
                if added == quota:
                    break
                doc = item["source_doc_id"]
                if item["identity"] in used or per_doc[doc] >= 2:
                    continue
                field = "clef" if cat == "PitchReading" else "gold"
                if label is not None and item.get(field) != label:
                    continue
                selected[cat].append(item)
                per_doc[doc] += 1
                used.add(item["identity"])
                added += 1
            if added != quota:
                raise ValueError(f"Not enough eligible {cat}/{label} questions: {added}/{quota}")
    return selected


@lru_cache(maxsize=2048)
def sounding_pitch(note):
    from music21 import pitch
    return pitch.Pitch(note.replace("b", "-")).ps


def shift_note(note, steps):
    letter, accidental, octave = NOTE.fullmatch(note).groups()
    number = int(octave) * 7 + "CDEFGAB".index(letter) + steps
    new_octave, index = divmod(number, 7)
    return f"{'CDEFGAB'[index]}{accidental}{new_octave}"


def pitch_distractors(item, rng, staff_steps=(-1, 1), require_staff_error=False):
    # A one-note flat label can make two error types enharmonically identical
    # (e.g. moving Bb to Cb vs removing its flat gives B). Retry the whole
    # combination so earlier choices do not block a valid later error type.
    for _ in range(50):
        try:
            return _pitch_distractors_once(item, rng, staff_steps, require_staff_error)
        except ValueError:
            continue
    raise ValueError(f"No distinct pitch distractor combination for {item.get('source_doc_id')}: {item['gold']}")


def _pitch_distractors_once(item, rng, staff_steps=(-1, 1), require_staff_error=False):
    notes = item["gold"].split()
    kinds = (["staff_position"] + rng.sample(["octave", "accidental", "opposite_clef"], 2)
             if require_staff_error else rng.sample(["staff_position", "octave", "accidental", "opposite_clef"], 3))
    results, signatures = [], {tuple(map(sounding_pitch, notes))}
    for kind in kinds:
        for attempt in range(100):
            changed = list(notes)
            indices = rng.sample(range(len(notes)), rng.randint(1, min(2, len(notes))))
            if kind == "opposite_clef":
                # Specific exception in request 5-3: the entire sequence changes.
                steps = -12 if item["clef"] == "treble" else 12
                changed = [shift_note(n, steps) for n in notes]
            else:
                for i in indices:
                    letter, accidental, octave = NOTE.fullmatch(notes[i]).groups()
                    if kind == "staff_position":
                        changed[i] = shift_note(notes[i], rng.choice(staff_steps))
                    elif kind == "octave":
                        changed[i] = f"{letter}{accidental}{int(octave) + rng.choice([-1, 1])}"
                    else:
                        changed[i] = f"{letter}{'' if accidental else rng.choice(['#', 'b'])}{octave}"
            if not all(NOTE.fullmatch(n) for n in changed):
                continue
            signature = tuple(map(sounding_pitch, changed))
            if signature not in signatures:
                results.append((" ".join(changed), kind))
                signatures.add(signature)
                break
        else:
            raise ValueError(f"Could not create a distinct {kind} distractor")
    return results


def distractors(item, rng, staff_steps=(-1, 1), require_staff_error=False, key_count_pairs=False):
    cat, gold = item["category"], item["gold"]
    if cat == "PitchReading":
        return pitch_distractors(item, rng, staff_steps, require_staff_error)
    if cat == "Clef":
        return [(option, "clef_type") for option in CLEF_OPTIONS if option != gold]
    count = item["value"]
    if cat == "MeasureCount":
        # Uniformly choose the correct answer's numerical rank (0..3).
        lower = rng.randrange(4)
        values = rng.sample([count - n for n in (1, 2, 3)], lower)
        values += rng.sample([count + n for n in (1, 2, 3)], 3 - lower)
        return [(str(n), "nearby_measure_count") for n in values]
    if count:
        sign = 1 if count > 0 else -1
        if key_count_pairs:
            n = abs(count)
            lower = list(range(max(1, n - 2), n))
            upper = list(range(n + 1, min(7, n + 2) + 1))
            sides = [side for side in (lower, upper) if side]
            other = rng.choice(rng.choice(sides))
            # Include zero as a distractor as well as a correct label. Otherwise
            # merely seeing 'no sharps or flats' would reveal the answer.
            if n <= 2 and rng.random() < 0.5:
                values = [-count, 0, sign * other]
            else:
                # Two magnitudes each appear with both signs, so repetition of
                # a numeral alone cannot identify the correct magnitude.
                values = [-count, other, -other]
            return [(key_text(value), "sharp_flat_flip" if value == -count else "nearby_key_signature") for value in values]
        near = [sign * n for n in range(max(0, abs(count) - 2), min(7, abs(count) + 2) + 1)
                if n != abs(count)]
        values = [-count] + rng.sample(near, 2)
    else:
        # Flipping zero would duplicate the correct answer; use nearby nonzero keys.
        values = rng.sample([-2, -1, 1, 2], 3)
    return [(key_text(n), "sharp_flat_flip" if n == -count else "nearby_key_signature") for n in values]


def make_rows(selected, output_split, seed, staff_steps=(-1, 1), require_staff_error=False, key_count_pairs=False):
    rows, audit = [], []
    for cat in CATEGORIES:
        items = selected[cat]
        positions = [i % 4 for i in range(len(items))]
        random.Random(f"{seed}/{output_split}/{cat}/positions").shuffle(positions)
        for item, position in zip(items, positions):
            rng = random.Random(f"{seed}/{output_split}/{item['identity']}")
            wrong = distractors(item, rng, staff_steps, require_staff_error, key_count_pairs)
            rng.shuffle(wrong)
            options = [option for option, _ in wrong]
            options.insert(position, item["gold"])
            if len({o.casefold() for o in options}) != 4:
                raise ValueError("Duplicate options")
            row = {
                "id": f"musixqa_{output_split}_{len(rows) + 1:05d}",
                "question": "<image 1> " + output_question(item), "options": options,
                "question_type": "multiple-choice", "answer": LETTERS[position],
                "image_paths": {"image_1": f"images/{item['source_doc_id']}.png"},
                "category": cat, "source_doc_id": item["source_doc_id"],
                "source_split": item["source_split"],
            }
            assert options[LETTERS.index(row["answer"])] == item["gold"]
            rows.append(row)
            audit.append({"id": row["id"], **item,
                          "output_question": output_question(item),
                          "distractors": [{"text": text, "type": kind} for text, kind in wrong]})
    return rows, audit


def save_jsonl(path, rows):
    with Path(path).open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def extract_images(source_dir, destination, doc_ids):
    from PIL import Image

    wanted = {f"{doc}.png" for doc in doc_ids}
    destination.mkdir(parents=True, exist_ok=True)
    local = Path(source_dir) / "images"
    missing = set(wanted)
    for name in sorted(wanted):
        path = local / name
        if path.is_file():
            shutil.copyfile(path, destination / name)
            missing.remove(name)
    archive = Path(source_dir) / "images.tar"
    if missing and archive.is_file():
        with tarfile.open(archive, "r|*") as tar:
            for member in tar:
                name = Path(member.name).name
                if member.isfile() and name in missing:
                    with tar.extractfile(member) as src, (destination / name).open("wb") as dst:
                        shutil.copyfileobj(src, dst)
                    missing.remove(name)
                if not missing:
                    break
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} images; run download --images first")
    hashes = {}
    for name in sorted(wanted):
        path = destination / name
        with Image.open(path) as image:
            if image.format != "PNG":
                raise ValueError(f"Not a source PNG: {name}")
            image.verify()
        hashes[name] = sha256(path)
    return hashes


def download(args):
    from huggingface_hub import hf_hub_download

    splits = ["test"] if args.test_only else ["train", "test"]
    files = ["README.md"] + [f"{split}_qa_{kind}.json" for split in splits for kind in ("simple", "omr")]
    if args.images:
        files.append("images.tar")
    print("Pinned revision:", REVISION, flush=True)
    print("Downloading:", ", ".join(files), flush=True)
    for name in files:
        path = hf_hub_download(repo_id=REPO, repo_type="dataset", revision=REVISION,
                               filename=name, local_dir=args.source_dir, token=False)
        print("Ready:", path, flush=True)


def inspect_sources(args):
    pools, _, report = collect(args.source_dir, args.split, 20, args.seed)
    selected = choose(pools, 20, getattr(args, "balance_content", False))
    output_split = "holdout" if args.split == "test" else "train"
    rows, audit = make_rows(selected, output_split, args.seed)
    report.update(status="text_preview_only_images_not_checked", seed=args.seed,
                  revision=REVISION, examples={cat: next(r for r in rows if r["category"] == cat)
                                              for cat in CATEGORIES},
                  preview_count=len(rows))
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / f"{args.split}_inspection.json", report)
    # Named preview to prevent accidental use as a validated training dataset.
    save_jsonl(out / f"{args.split}_preview.jsonl", rows)
    save_jsonl(out / f"{args.split}_preview_audit.jsonl", audit)
    write_json(out / f"{args.split}_quality_report.json", quality_summary(rows, audit))
    print(json.dumps(report, ensure_ascii=False, indent=2))


def build(args):
    if args.train_per_category < 4 or args.holdout_per_category < 4:
        raise ValueError("Use at least four questions per split/category")
    if args.train_per_category % 4 or args.holdout_per_category % 4:
        raise ValueError("Counts must be multiples of four for exact answer balance")
    out = Path(args.out_dir)
    if out.exists():
        raise FileExistsError(f"Output already exists: {out}; choose a new output directory")
    source_dir = Path(args.source_dir)
    for split in ("train", "test"):
        for path in sources(source_dir, split):
            if not path.is_file():
                raise FileNotFoundError(f"Missing {path}; run download first")
    if not (source_dir / "images.tar").is_file() and not (source_dir / "images").is_dir():
        raise FileNotFoundError("No source images; run download --images first")
    datasets, audits, reports, all_docs = {}, [], {}, {}
    for source_split, output_split, count in [
        ("train", "train", args.train_per_category), ("test", "holdout", args.holdout_per_category)
    ]:
        print("Selecting:", source_split, flush=True)
        pools, all_docs[source_split], reports[output_split] = collect(source_dir, source_split, count, args.seed)
        datasets[output_split], audit = make_rows(choose(pools, count, getattr(args, "balance_content", False)), output_split, args.seed)
        audits.extend(audit)
    if all_docs["train"] & all_docs["test"]:
        raise ValueError("Source train/test documents overlap")
    out.mkdir(parents=True)
    write_json(out / "BUILD_INCOMPLETE.json", {"status": "in_progress"})
    doc_ids = {row["source_doc_id"] for rows in datasets.values() for row in rows}
    image_hashes = extract_images(source_dir, out / "images", doc_ids)
    for split, rows in datasets.items():
        save_jsonl(out / f"{split}.jsonl", rows)
    save_jsonl(out / "generation_audit.jsonl", audits)
    write_json(out / "quality_report.json", quality_summary([row for rows in datasets.values() for row in rows], audits))
    shutil.copyfile(__file__, out / "build_musixqa.py")
    source_hashes = {p.name: sha256(p) for split in ("train", "test") for p in sources(source_dir, split)}
    manifest = {
        "version": "musixqa_mcq_v1_sample", "seed": args.seed,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "content_balancing": bool(getattr(args, "balance_content", False)),
        "bar_number_review_registry": BAR_NUMBER_REVIEW,
        "source": {"repo": REPO, "revision": REVISION, "license": "MIT", "qa_sha256": source_hashes},
        "counts": {s: dict(Counter(r["category"] for r in rows)) for s, rows in datasets.items()},
        "source_inspection": reports, "image_sha256": image_hashes,
        "data_sha256": {f"{s}.jsonl": sha256(out / f"{s}.jsonl") for s in datasets},
        "python": sys.version,
        "packages": {p: importlib.metadata.version(p) for p in ("huggingface_hub", "music21", "Pillow")},
        "manual_review": {"status": "pending", "reviewed": 0},
        "mmmu_deduplication": {"status": "pending"},
        "team_sample_review": "pending",
        "decisions_requiring_sample_review": [
            "Opposite-clef distractors change all notes as the specific exception in request 5-3; other pitch errors change 1-2 notes.",
            "Zero key signature uses three nearby nonzero distractors because a sharp/flat flip would duplicate zero.",
            "Clef uses the four fixed options from the request; inspect answer-frequency/length shortcuts.",
        ],
    }
    write_json(out / "manifest.json", manifest)
    validator = Path(args.validator)
    subprocess.run([sys.executable, "-X", "utf8", str(validator), "--data-dir", str(out),
                    "--min-train-per-category", str(args.train_per_category),
                    "--min-holdout-per-category", str(args.holdout_per_category)], check=True)
    validation = json.loads((out / "validation_report.json").read_text(encoding="utf-8"))
    manifest["validation"] = {"errors": validation["error_count"], "warnings": validation["warnings"]}
    write_json(out / "manifest.json", manifest)
    (out / "BUILD_INCOMPLETE.json").unlink()
    print("Built:", out, "-- image/content review and team approval still pending")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("download", "inspect", "build"):
        command = sub.add_parser(name)
        command.add_argument("--source-dir", default="data/musixqa_source")
        if name == "download":
            command.add_argument("--test-only", action="store_true")
            command.add_argument("--images", action="store_true", help="Also download the 8.7 GB images.tar")
        else:
            command.add_argument("--seed", type=int, default=42)
            command.add_argument("--balance-content", action="store_true",
                                 help="Balance observed correct-answer values within KeySignature/Clef; changes selected items")
            command.add_argument("--out-dir", default="data/musixqa_inspection" if name == "inspect" else "data/musixqa_mcq_sample")
            if name == "inspect":
                command.add_argument("--split", choices=["train", "test"], default="test")
            else:
                command.add_argument("--train-per-category", type=int, default=16)
                command.add_argument("--holdout-per-category", type=int, default=4)
                command.add_argument("--validator", default="lab/validate_mcq_dataset.py")
    args = parser.parse_args()
    {"download": download, "inspect": inspect_sources, "build": build}[args.command](args)


if __name__ == "__main__":
    main()
