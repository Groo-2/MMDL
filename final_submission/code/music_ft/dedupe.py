"""학습/hold-out 문항 중 MMMU(dev/validation/test)와 겹치는 것을 찾아 제거한다.

과제의 평가 무결성: MMMU 평가 데이터(validation/test)를 학습에 쓰지 않는다. 일반 데이터는 어느 과목과도 겹칠 수 있어서 기본은 전 과목을 대조한다.

검사
  1) 이미지: sha1 완전 일치 + pHash(256비트) 해밍 거리 <= --max-hamming
     질문 이미지(image_1)와 선택지 이미지 모두를 MMMU의 image_1..7 전체와 비교한다.
  2) 텍스트: 정규화한 질문이 MMMU 질문과 완전히 같으면(30자 이상) 중복,
     선택지 집합이 같고 글자 유사도(difflib) >= --text-ratio 이어도 중복

입력 <data-dir>/<name>.jsonl(기본 train, holdout)과 같은 폴더에 <name>.dedup.jsonl과 dedupe_report.json을 쓴다.
MMMU는 run_mmmu_eval.py와 같은 revision을 pin해서 parquet로 직접 읽는다(datasets 불필요).

  python music_ft/dedupe.py --data-dir data/ssmr                                                  # Music만 (기존 동작)
  python music_ft/dedupe.py --data-dir data/general/dv_pool --subjects all --names train          # 전 과목 (일반 데이터)
  --ref-cache data/mmmu_ref.pkl 을 주면 MMMU 기준 해시를 저장해 다음 실행에서 재사용한다.
"""

import argparse
import difflib
import hashlib
import io
import json
import pickle
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

MMMU_REPO = "MMMU/MMMU"
MMMU_REVISION = "98e6ac0cb9b7b2cd2c991b85a50762edc4aedc68"  # run_mmmu_eval.py와 동일

HASH_SIZE = 16     # 16x16 저주파 = 256비트 (32x32 해시는 얇은 오선 이미지에서 너무 거칠다)
RESIZE = 64
MIN_EXACT_CHARS = 30     # 이보다 짧은 질문은 완전 일치만으로 중복이라 부르지 않는다 (예: "Find x.")


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


def normalize(text):
    return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()


def question_key(text):
    """'<image 1>' 같은 표시를 빼고 정규화한 질문. MMMU 쪽과 우리 쪽에 똑같이 적용한다."""
    return normalize(re.sub(r"<image\s*\d*>", " ", str(text)))


def parse_options(value):
    if isinstance(value, list):
        return value
    try:
        return json.loads(value.replace("'", '"'))
    except Exception:
        return [value]


def load_mmmu(cache_dir, subjects="Music"):
    """MMMU의 dev/validation/test 문항을 하나씩 돌려준다. subjects는 'all' 또는 'Music,Math' 같은 목록."""
    import pyarrow.parquet as pq
    from huggingface_hub import snapshot_download

    wanted = None if subjects == "all" else [s.strip() for s in subjects.split(",")]
    patterns = ["*/*.parquet"] if wanted is None else [f"{s}/*.parquet" for s in wanted]
    root = snapshot_download(repo_id=MMMU_REPO, repo_type="dataset", revision=MMMU_REVISION,
                             allow_patterns=patterns, cache_dir=cache_dir)
    for path in sorted(Path(root).glob("*/*.parquet")):
        if wanted is not None and path.parent.name not in wanted:
            continue
        split = path.name.split("-")[0]
        for row in pq.read_table(path).to_pylist():
            images = []
            for n in range(1, 8):
                cell = row.get(f"image_{n}")
                if cell and cell.get("bytes"):
                    images.append(cell["bytes"])
            yield {"id": row["id"], "split": split, "question": row["question"], "options": row["options"], "images": images}


def build_reference(items):
    """MMMU 문항들에서 중복 검사 기준(이미지 해시, 질문 색인)을 만든다."""
    sha, hashes, owner = set(), [], []
    by_options, exact = defaultdict(list), {}
    count = 0
    for m in items:
        count += 1
        for data in m["images"]:
            sha.add(hashlib.sha1(data).hexdigest())
            hashes.append(phash(data))
            owner.append(f"{m['split']}:{m['id']}")
        key = question_key(m["question"])
        by_options[frozenset(normalize(o) for o in parse_options(m["options"]))].append((key, m["id"]))
        if len(key) >= MIN_EXACT_CHARS:
            exact[key] = m["id"]
    return {"sha": sha, "hash": np.stack(hashes) if hashes else np.zeros((0, HASH_SIZE * HASH_SIZE), dtype=bool),
            "owner": owner, "by_options": dict(by_options), "exact": exact, "items": count}


def check_row(row, data_dir, ref, max_hamming=20, text_ratio=0.9):
    """(제거 사유 또는 None, 가장 가까운 MMMU 문항, 최소 해밍 거리)."""
    reason, nearest, row_min = None, None, 10 ** 9
    for key, rel in row["image_paths"].items():
        data = (Path(data_dir) / rel).read_bytes()
        if hashlib.sha1(data).hexdigest() in ref["sha"]:
            return f"sha1:{key}", None, 0
        if len(ref["hash"]):
            dist = (ref["hash"] != phash(data)).sum(axis=1)
            j = int(dist.argmin())
            if dist[j] < row_min:
                row_min, nearest = int(dist[j]), ref["owner"][j]
            if dist[j] <= max_hamming and reason is None:
                reason = f"phash:{key}"
    if reason is None:
        q = question_key(row["question"])
        if len(q) >= MIN_EXACT_CHARS and q in ref["exact"]:
            return "text_exact", ref["exact"][q], row_min
        opts = frozenset(normalize(o) for o in row["options"])
        matcher = difflib.SequenceMatcher(None)
        matcher.set_seq2(q)
        for mq, mid in ref["by_options"].get(opts, ()):
            matcher.set_seq1(mq)
            if matcher.real_quick_ratio() >= text_ratio and matcher.quick_ratio() >= text_ratio and matcher.ratio() >= text_ratio:
                return "text", mid, row_min
    return reason, nearest, row_min


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", required=True, help="train.jsonl(과 holdout.jsonl)이 있는 폴더")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--subjects", default="Music", help="대조할 MMMU 과목: 'all' 또는 'Music,Math' (기본 Music)")
    parser.add_argument("--names", default="train,holdout", help="처리할 jsonl 이름(확장자 제외), 쉼표로 구분")
    parser.add_argument("--max-hamming", type=int, default=20)
    parser.add_argument("--text-ratio", type=float, default=0.9)
    parser.add_argument("--ref-cache", default=None, help="MMMU 기준 해시를 저장/재사용할 pickle 경로")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    cache_key = {"subjects": args.subjects, "revision": MMMU_REVISION}
    ref = None
    if args.ref_cache and Path(args.ref_cache).exists():
        with open(args.ref_cache, "rb") as f:
            saved = pickle.load(f)
        if saved.get("key") == cache_key:
            ref = saved["ref"]
            print(f"MMMU 기준 재사용: {args.ref_cache}")
    if ref is None:
        ref = build_reference(load_mmmu(args.cache_dir, args.subjects))
        if args.ref_cache:
            with open(args.ref_cache, "wb") as f:
                pickle.dump({"key": cache_key, "ref": ref}, f)
    print(f"MMMU({args.subjects}) 문항 {ref['items']}개, 이미지 {len(ref['hash'])}개")

    report = {"subjects": args.subjects, "mmmu_items": ref["items"], "mmmu_images": int(len(ref["hash"])),
              "max_hamming": args.max_hamming, "text_ratio": args.text_ratio}

    for name in [n.strip() for n in args.names.split(",") if n.strip()]:
        src = data_dir / f"{name}.jsonl"
        if not src.exists():
            print(f"{name}: {src}가 없어 건너뜀")
            continue
        rows = [json.loads(line) for line in src.open(encoding="utf-8")]
        kept, dropped, min_dists = [], [], []
        for i, row in enumerate(rows, 1):
            reason, nearest, row_min = check_row(row, data_dir, ref, args.max_hamming, args.text_ratio)
            min_dists.append(row_min)
            if reason:
                dropped.append({"id": row["id"], "reason": reason, "nearest": nearest, "min_hamming": row_min})
            else:
                kept.append(row)
            if i % 2000 == 0:
                print(f"  {name}: {i}/{len(rows)} (제거 {len(dropped)})", flush=True)

        out = data_dir / f"{name}.dedup.jsonl"
        with out.open("w", encoding="utf-8") as f:
            for row in kept:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        arr = np.array(min_dists) if min_dists else np.array([0])
        by_reason = defaultdict(int)
        for d in dropped:
            by_reason[d["reason"].split(":")[0]] += 1
        report[name] = {
            "total": len(rows), "kept": len(kept), "dropped": len(dropped), "dropped_by_reason": dict(by_reason),
            "min_hamming_percentiles": {str(p): float(np.percentile(arr, p)) for p in (0, 1, 5, 50)},
            "dropped_examples": dropped[:20],
        }
        print(f"{name}: {len(rows)} → {len(kept)} (제거 {len(dropped)} {dict(by_reason)}), 최소 해밍 분포 "
              f"{report[name]['min_hamming_percentiles']}")

    (data_dir / "dedupe_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
