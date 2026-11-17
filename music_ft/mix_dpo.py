"""여러 DPO 쌍 파일(build_dpo.py의 출력)을 이어 붙여 하나로 섞는다.

제출 모델(DeepVision 쌍 + MusiXQA 쌍)은 이렇게 만들었다:
  python music_ft/mix_dpo.py --inputs data/dpo/dpo_dv.jsonl data/dpo/dpo_mx.jsonl --out data/dpo/dpo_dv_mx.jsonl

- 입력 순서대로 이어 붙인 뒤 --seed(기본 42)로 한 번 셔플한다. 학습은 이 파일의 순서대로 쌍을 읽는다.
- 쌍이 가리키는 이미지(data_dir + image_paths)가 모두 있는지 확인하고, 없는 것이 하나라도 있으면 실패한다(학습 도중에 죽는 것을 막는다).
- 같은 id가 두 입력에 모두 있으면 실패한다(서로 다른 데이터에서 온 쌍이 섞여 같은 문항이 두 번 학습되는 것을 막는다).
"""

import argparse
import json
import os
import random
from collections import Counter
from pathlib import Path


def mix(inputs, seed=42):
    """입력 파일들의 쌍을 이어 붙여 셔플한 리스트를 돌려준다. (rows, 입력별 개수)"""
    rows, counts, seen = [], {}, {}
    for path in inputs:
        n = 0
        with Path(path).open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                if r["id"] in seen:
                    raise ValueError(f"같은 id가 두 입력에 있다: {r['id']} ({seen[r['id']]}, {path})")
                seen[r["id"]] = str(path)
                rows.append(r)
                n += 1
        counts[str(path)] = n
    random.Random(seed).shuffle(rows)
    return rows, counts


def missing_images(rows, root="."):
    """쌍이 가리키는 이미지 중 없는 것 [(id, 경로)]."""
    return [(r["id"], p) for r in rows for p in r["image_paths"].values()
            if not os.path.exists(os.path.join(root, r["data_dir"], p))]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--inputs", nargs="+", required=True, help="build_dpo.py가 만든 쌍 파일들")
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-check-images", action="store_true", help="이미지 존재 확인을 건너뛴다(다른 곳에서 만든 파일을 옮길 때)")
    args = parser.parse_args(argv)

    rows, counts = mix(args.inputs, args.seed)
    if not args.no_check_images:
        miss = missing_images(rows)
        if miss:
            raise SystemExit(f"없는 이미지 {len(miss)}개 (예: {miss[:3]}). 이미지를 먼저 만들거나 --no-check-images를 쓰세요.")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    kinds = Counter(r["rejected_kind"] for r in rows)
    print(f"입력 {counts} -> 합계 {len(rows)}쌍 | 나쁜 응답 종류 {dict(kinds)} -> {out}")


if __name__ == "__main__":
    main()
