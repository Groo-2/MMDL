"""Check exact question duplicates and near-duplicate images across train/holdout."""

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from dedupe import HASH_SIZE, phash


ROOT = Path(__file__).resolve().parents[1] / "data" / "musixqa_mcq_pilot20_v1"


def rows(name):
    return [json.loads(line) for line in (ROOT / name).read_text(encoding="utf-8").splitlines()]


def main():
    train = rows("train.dedup.jsonl")
    holdout = rows("holdout.dedup.jsonl")
    image_owners = defaultdict(lambda: {"train": [], "holdout": []})
    content_owners = defaultdict(lambda: {"train": [], "holdout": []})
    for split, items in (("train", train), ("holdout", holdout)):
        for row in items:
            rel = row["image_paths"]["image_1"]
            image_owners[rel][split].append(row["id"])
            content_key = (rel, row["question"].strip().casefold(),
                           frozenset(x.strip().casefold() for x in row["options"]))
            content_owners[content_key][split].append(row["id"])

    cross_content = [{"train": v["train"], "holdout": v["holdout"]} for v in content_owners.values()
                     if v["train"] and v["holdout"]]
    within_content = [{"train": v["train"], "holdout": v["holdout"]} for v in content_owners.values()
                      if len(v["train"]) > 1 or len(v["holdout"]) > 1]
    paths = sorted(image_owners)
    image_sha = []
    hashes = []
    for i, rel in enumerate(paths, 1):
        data = (ROOT / rel).read_bytes()
        image_sha.append(hashlib.sha256(data).hexdigest())
        hashes.append(int.from_bytes(np.packbits(phash(data)).tobytes(), "big"))
        if i % 1000 == 0:
            print("pHash", i, "/", len(paths), flush=True)

    train_idx = [i for i, rel in enumerate(paths) if image_owners[rel]["train"]]
    holdout_idx = [i for i, rel in enumerate(paths) if image_owners[rel]["holdout"]]
    near = []
    closest = {"distance": HASH_SIZE ** 2 + 1}
    for h in holdout_idx:
        distances = [(hashes[h] ^ hashes[t]).bit_count() for t in train_idx]
        m = min(distances)
        t = train_idx[distances.index(m)]
        if m < closest["distance"]:
            closest = {"distance": m, "train_image": paths[t], "holdout_image": paths[h]}
        if m <= 20:
            near.extend({"distance": distance, "train_image": paths[t], "holdout_image": paths[h]}
                        for t, distance in zip(train_idx, distances) if distance <= 20)
    exact = [{"train_image": paths[t], "holdout_image": paths[h]}
             for t in train_idx for h in holdout_idx if image_sha[t] == image_sha[h]]
    report = {
        "scope": "train.dedup.jsonl versus holdout.dedup.jsonl",
        "method": "Exact SHA-256 image bytes, 256-bit pHash Hamming <=20, and exact normalized question/options with same image",
        "train_rows": len(train), "holdout_rows": len(holdout),
        "unique_images": len(paths),
        "cross_split_exact_image_pairs": exact,
        "cross_split_near_image_pairs": near,
        "closest_cross_split_images": closest,
        "cross_split_same_image_question_option_sets": cross_content,
        "within_split_same_image_question_option_sets": within_content,
        "limitations": "pHash may miss crops, rotations or altered layouts; this is an image and exact-text check, not a semantic paraphrase search."
    }
    (ROOT / "internal_overlap_extended_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: (len(v) if isinstance(v, list) else v) for k, v in report.items()
                      if k not in ("scope", "method", "limitations")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
