"""SSMR 학습/hold-out 문항 중 MMMU Music(dev/validation/test)과 겹치는 것을 찾아 제거한다.

교수님 규칙: 학습 데이터는 MMMU 평가 데이터셋(val/test)이 아니어야 한다.

검사
  1) 이미지: sha1 완전 일치 + pHash(256비트) 해밍 거리 <= --max-hamming
     SSMR의 악보 이미지(image_1)와 선택지 이미지 모두를 MMMU Music의 image_1..7 전체와 비교한다.
  2) 텍스트: 질문 글자 유사도(difflib) >= --text-ratio 이고 선택지 집합이 같으면 중복

입력 jsonl(train.jsonl, holdout.jsonl)과 같은 폴더에 *.dedup.jsonl과 dedupe_report.json을 쓴다.
MMMU는 run_mmmu_eval.py와 같은 revision을 pin해서 parquet로 직접 읽는다(datasets 불필요).
"""

import argparse
import difflib
import hashlib
import io
import json
import re
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import snapshot_download
from PIL import Image

MMMU_REPO = "MMMU/MMMU"
MMMU_REVISION = "98e6ac0cb9b7b2cd2c991b85a50762edc4aedc68"  # run_mmmu_eval.py와 동일

HASH_SIZE = 16     # 16x16 저주파 = 256비트 (32x32 해시는 얇은 오선 이미지에서 너무 거칠다)
RESIZE = 64


def _dct_matrix(n):
    k = np.arange(n)[:, None]
    i = np.arange(n)[None, :]
    mat = np.cos(np.pi * (2 * i + 1) * k / (2 * n))
    mat[0] *= 1 / np.sqrt(2)
    return mat * np.sqrt(2 / n)


_DCT = _dct_matrix(RESIZE)


def phash(data):
    """바이트로 받은 이미지의 256비트 pHash를 bool 배열로 돌려준다."""
    img = Image.open(io.BytesIO(data))
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(bg, img)
    img = img.convert("L").resize((RESIZE, RESIZE), Image.LANCZOS)
    pixels = np.asarray(img, dtype=np.float64)
    dct = _DCT @ pixels @ _DCT.T
    low = dct[:HASH_SIZE, :HASH_SIZE].flatten()
    return low > np.median(low[1:])  # DC 성분 제외한 중앙값


def load_mmmu_music(cache_dir):
    root = snapshot_download(
        repo_id=MMMU_REPO,
        repo_type="dataset",
        revision=MMMU_REVISION,
        allow_patterns=["Music/*"],
        cache_dir=cache_dir,
    )
    items = []
    for path in sorted(Path(root, "Music").glob("*.parquet")):
        split = path.name.split("-")[0]
        for row in pq.read_table(path).to_pylist():
            images = []
            for n in range(1, 8):
                cell = row.get(f"image_{n}")
                if cell and cell.get("bytes"):
                    images.append(cell["bytes"])
            items.append({
                "id": row["id"],
                "split": split,
                "question": row["question"],
                "options": row["options"],
                "images": images,
            })
    return items


def normalize(text):
    return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()


def parse_options(value):
    if isinstance(value, list):
        return value
    try:
        return json.loads(value.replace("'", '"'))
    except Exception:
        return [value]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, help="build_ssmr.py의 --out-dir")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--max-hamming", type=int, default=20)
    parser.add_argument("--text-ratio", type=float, default=0.9)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    mmmu = load_mmmu_music(args.cache_dir)
    print("MMMU Music:", {s: sum(m["split"] == s for m in mmmu) for s in sorted({m['split'] for m in mmmu})})

    ref_sha, ref_hash, ref_owner = set(), [], []
    for item in mmmu:
        for data in item["images"]:
            ref_sha.add(hashlib.sha1(data).hexdigest())
            ref_hash.append(phash(data))
            ref_owner.append(f"{item['split']}:{item['id']}")
    ref_hash = np.stack(ref_hash)  # (R, 256)
    print("MMMU Music 이미지:", len(ref_hash))

    mmmu_text = [
        (normalize(m["question"]), frozenset(normalize(o) for o in parse_options(m["options"])), m["id"])
        for m in mmmu
    ]

    report = {"max_hamming": args.max_hamming, "text_ratio": args.text_ratio}

    for name in ("train", "holdout"):
        src = data_dir / f"{name}.jsonl"
        rows = [json.loads(line) for line in src.open(encoding="utf-8")]
        kept, dropped, min_dists = [], [], []

        for row in rows:
            reason, nearest = None, None
            row_min = 10 ** 9

            for key, rel in row["image_paths"].items():
                data = (data_dir / rel).read_bytes()
                if hashlib.sha1(data).hexdigest() in ref_sha:
                    reason, nearest = f"sha1:{key}", None
                    row_min = 0
                    break
                dist = (ref_hash != phash(data)).sum(axis=1)
                j = int(dist.argmin())
                if dist[j] < row_min:
                    row_min, nearest = int(dist[j]), ref_owner[j]
                if dist[j] <= args.max_hamming and reason is None:
                    reason = f"phash:{key}"

            min_dists.append(row_min)

            if reason is None:
                q = normalize(row["question"].replace("<image 1>", ""))
                opts = frozenset(normalize(o) for o in row["options"])
                for mq, mo, mid in mmmu_text:
                    if mo == opts and difflib.SequenceMatcher(None, q, mq).ratio() >= args.text_ratio:
                        reason, nearest = "text", mid
                        break

            if reason:
                dropped.append({"id": row["id"], "reason": reason, "nearest": nearest, "min_hamming": row_min})
            else:
                kept.append(row)

        out = data_dir / f"{name}.dedup.jsonl"
        with out.open("w", encoding="utf-8") as f:
            for row in kept:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        arr = np.array(min_dists)
        report[name] = {
            "total": len(rows),
            "kept": len(kept),
            "dropped": len(dropped),
            "min_hamming_percentiles": {str(p): float(np.percentile(arr, p)) for p in (0, 1, 5, 50)},
            "dropped_examples": dropped[:20],
        }
        print(f"{name}: {len(rows)} → {len(kept)} (제거 {len(dropped)}), 최소 해밍 분포 "
              f"{report[name]['min_hamming_percentiles']}")

    (data_dir / "dedupe_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
