"""Generate a local HTML review report and a blank CSV; never mark items reviewed."""

import argparse
from collections import Counter
import csv
import html
import hashlib
import json
from pathlib import Path
import random
from urllib.parse import quote


CATEGORIES = ("KeySignature", "MeasureCount", "PitchReading", "Clef")
GUIDANCE = {
    "KeySignature": "음자리표 옆 조표의 샵/플랫 개수를 세세요. 마디 안의 임시표와 구분하고, 조 이름을 추측하지 않습니다.",
    "MeasureCount": "반복 연주 횟수를 제외하고 악보에 실제로 적힌 마디를 합산하세요. 피아노 위·아래 보표를 중복해서 세지 않습니다. 시작 번호가 2일 수 있으므로 마지막 인쇄 번호를 곧바로 마디 수로 쓰지 마세요.",
    "PitchReading": "지정된 마디와 음자리표의 음 이름·옥타브·조표·임시표를 대조하세요. 이 데이터는 붙임줄 양쪽의 적힌 음표를 각각 셉니다. 원본 QA의 마디 인덱스와 인쇄 번호가 다르면 따로 확인하세요.",
    "Clef": "악보 전체에서 사용한 음자리표를 확인하세요. 높은음자리표만/낮은음자리표만/둘 다를 구별합니다.",
}
FIELDS = ["id", "split", "category", "source_doc_id", "expected_answer", "expected_option",
          "verdict", "image_matches_answer", "only_one_correct_option", "distractors_ok",
          "difficulty_ok", "notes"]


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def select_review_rows(rows, per_category, holdout_per_category, seed):
    """Random sample within each category/split; never infer review verdicts."""
    if per_category is None:
        return rows
    if not 0 < holdout_per_category < per_category:
        raise ValueError("Need both train and holdout review items in each category")
    rng = random.Random(seed)
    selected = []
    for cat in CATEGORIES:
        for split, count in (("train", per_category - holdout_per_category),
                             ("holdout", holdout_per_category)):
            candidates = sorted((r for r in rows if r["category"] == cat and r["split"] == split),
                                key=lambda r: r["id"])
            if len(candidates) < count:
                raise ValueError(f"Not enough {cat}/{split} items: {len(candidates)} < {count}")
            selected.extend(rng.sample(candidates, count))
    return selected


def prepare(data_dir, train_file="train.jsonl", holdout_file="holdout.jsonl",
            per_category=None, holdout_per_category=10, seed=42):
    root = Path(data_dir).resolve()
    if (root / "BUILD_INCOMPLETE.json").exists():
        raise ValueError("Sample build is incomplete")
    rows = []
    inputs = {"train": train_file, "holdout": holdout_file}
    for split, name in inputs.items():
        rows.extend({**row, "split": split} for row in read_jsonl(root / name))
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate item IDs")
    rows = select_review_rows(rows, per_category, holdout_per_category, seed)
    ids = [row["id"] for row in rows]
    selection = {
        "seed": seed, "method": "random within category and split" if per_category else "all items",
        "per_category": per_category, "holdout_per_category": holdout_per_category if per_category else None,
        "input_sha256": {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in inputs.values()},
        "selected_ids": ids,
        "selected_rows_sha256": hashlib.sha256(json.dumps(rows, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
        "counts": dict(Counter(f"{r['category']}/{r['split']}" for r in rows)),
        "note": "Selection only. No human or visual review has been performed by this command.",
    }
    selection_path = root / "review_selection.json"
    if selection_path.exists() and json.loads(selection_path.read_text(encoding="utf-8")) != selection:
        raise ValueError("Review inputs or selection changed; preserve and reconcile previous review first")
    audit = {row["id"]: row for row in read_jsonl(root / "generation_audit.jsonl")}
    for row in rows:
        if row["id"] not in audit:
            raise ValueError(f"Missing audit for {row['id']}")
        path = (root / row["image_paths"]["image_1"]).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f"Missing or outside image path for {row['id']}")

    csv_path = root / "manual_review.csv"
    if csv_path.exists():
        with csv_path.open(encoding="utf-8-sig", newline="") as stream:
            previous = list(csv.DictReader(stream))
        if {row["id"] for row in previous} != set(ids):
            raise ValueError("Existing manual_review.csv has different IDs; preserve it and reconcile first")
        expected = {row["id"]: row for row in rows}
        for old in previous:
            current = expected[old["id"]]
            if old["expected_answer"] != current["answer"] or old["expected_option"] != current["options"]["ABCD".index(current["answer"])]:
                raise ValueError(f"Reviewed answer changed: {old['id']}")
        print("Preserved existing review CSV:", csv_path)
    else:
        with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            for row in rows:
                writer.writerow({
                    "id": row["id"], "split": row["split"], "category": row["category"],
                    "source_doc_id": row["source_doc_id"], "expected_answer": row["answer"],
                    "expected_option": row["options"]["ABCD".index(row["answer"])],
                    "verdict": "pending",
                })

    with csv_path.open(encoding="utf-8-sig", newline="") as stream:
        review_rows = {row["id"]: row for row in csv.DictReader(stream)}
    verdict_counts = Counter(row.get("verdict", "pending") for row in review_rows.values())

    esc = html.escape
    cards = []
    for cat in CATEGORIES:
        group = [row for row in rows if row["category"] == cat]
        cards.append(f'<section id="{cat}"><h2>{cat} · {len(group)}문항</h2><p>{esc(GUIDANCE[cat])}</p>')
        for row in group:
            trace = audit[row["id"]]
            review = review_rows[row["id"]]
            url = quote(row["image_paths"]["image_1"], safe="/")
            options = "".join(f"<li>{esc(option)}</li>" for option in row["options"])
            wrong = "".join(f"<li><code>{esc(d['type'])}</code>: {esc(d['text'])}</li>"
                            for d in trace["distractors"])
            notes = []
            if any(d["type"] == "opposite_clef" for d in trace["distractors"]):
                notes.append("생성 규칙: 음자리표 착각 오답은 요청서의 예외로 해석해 모든 음을 변경했습니다.")
            if cat == "KeySignature" and trace.get("value") == 0:
                notes.append("생성 규칙: 조표 0개는 샵/플랫을 뒤집어도 같으므로 인접한 비영 조표를 오답으로 썼습니다.")
            note_html = "".join(f'<p class="note">{esc(note)}</p>' for note in notes)
            review_html = f"<p>기록된 검수 결과: <strong>{esc(review.get('verdict', 'pending'))}</strong></p>"
            if review.get("resolution"):
                review_html += f"<p class='note'>수정 상태: {esc(review['resolution'])}</p>"
            if review.get("notes"):
                review_html += f"<details><summary>기존 검수 메모</summary><p>{esc(review['notes'])}</p></details>"
            if review.get("revision_note"):
                review_html += f"<p class='note'>{esc(review['revision_note'])}</p>"
            cards.append(f'''<article id="{esc(row['id'], quote=True)}">
<div class="question">
<p class="meta">{esc(row['id'])} · {esc(row['split'])} · 문서 {esc(row['source_doc_id'])}</p>
<h3>{esc(row['question'].removeprefix('<image 1> '))}</h3>
{review_html}
<ol type="A">{options}</ol>
<p>악보를 먼저 읽은 뒤 정답을 펼쳐 비교하세요.</p>
<details><summary>정답 · 원본 라벨 · 오답 생성 근거 보기</summary>
<p><strong>정답: {esc(row['answer'])} — {esc(trace['gold'])}</strong></p>
<p>원본 질문: {esc(trace['source_question'])}</p>
<p>원본 답: {esc(trace['source_answer'])}</p>
<p class="meta">{esc(trace['source_file'])}, 배열 인덱스 {trace['source_index']} (0부터)</p>
<ul>{wrong}</ul>{note_html}</details>
<p class="record">CSV에서 <code>{esc(row['id'])}</code> 행에 결과를 기록하세요.</p>
</div>
<a class="score" href="{url}" target="_blank" rel="noopener"><img src="{url}" loading="lazy" alt="문서 {esc(row['source_doc_id'])} 원본 악보"><span>이미지를 누르면 원본 크기로 열립니다.</span></a>
</article>''')
        cards.append("</section>")
    counts = Counter(row["split"] for row in rows)
    nav = " · ".join(f'<a href="#{cat}">{cat}</a>' for cat in CATEGORIES)
    document = '''<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>MusiXQA 샘플 내용 검수</title>
<style>
body{font:16px/1.6 system-ui,sans-serif;color:#202632;background:#f3f5f8;margin:0}
main{max-width:1500px;margin:auto;padding:28px}header{background:white;padding:24px;border-radius:12px}
h1{margin-top:0}nav{position:sticky;top:0;background:#edf2fa;padding:12px;z-index:1}
a{color:#1952a5}article{display:grid;grid-template-columns:minmax(300px,1fr) minmax(400px,1.5fr);gap:24px;background:white;padding:24px;margin:20px 0;border:1px solid #d5dbe5;border-radius:12px;scroll-margin-top:64px}
section{scroll-margin-top:64px}.question{min-width:0}.score{display:block}.score img{display:block;width:100%;height:auto}.score span,.meta{color:#606b7a;font-size:13px}
li{margin:10px 0;overflow-wrap:anywhere}details{padding:14px;background:#eef4fc}summary{cursor:pointer;font-weight:bold}
.note{padding:10px;background:#fff1d6}.record{border-top:1px solid #ddd;padding-top:14px}code{overflow-wrap:anywhere}
@media(max-width:900px){article{grid-template-columns:1fr}main{padding:12px}}
@media print{nav{position:static}article{break-before:page}}
</style></head><body><main><header><h1>MusiXQA 샘플 내용 검수</h1>
'''
    document += f"<p>총 {len(rows)}문항 · 학습 {counts['train']} · holdout {counts['holdout']}. CSV 기준 검수 기록: {esc(', '.join(f'{key} {value}' for key, value in sorted(verdict_counts.items())))}.</p>"
    document += '''<p>이 화면은 읽기 전용입니다. <a href="manual_review.csv">manual_review.csv</a>를 Excel 등에서 열어 문항별 결과를 기록하세요.
<code>verdict</code>: <strong>pass</strong>(통과), <strong>fix</strong>(수정 필요), <strong>uncertain</strong>(판단 보류), <strong>pending</strong>(미검수).</p>
<p><code>image_matches_answer</code>, <code>only_one_correct_option</code>, <code>distractors_ok</code>, <code>difficulty_ok</code>에는 yes/no/uncertain을 적고,
<code>notes</code>에 문제점·수정 제안을 남기세요. 확신이 없는 문항은 uncertain으로 기록합니다.</p>
</header>'''
    if per_category:
        document += f"<p>카테고리별 무작위 {per_category}문항(학습 {per_category - holdout_per_category} + holdout {holdout_per_category})을 검수합니다. 이 표본의 결과를 전체 문항의 개별 검수 완료로 기록하지 않습니다.</p>"
    manifest_path = root / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text(encoding="utf-8")).get("team_sample_review") == "waived_by_user":
        document += "<p>팀 확인 단계는 사용자 지시로 생략했습니다. 새 문항의 내용 검수는 별도로 기록합니다.</p>"
    document += f"<nav>{nav}</nav>" + "\n".join(cards) + "</main></body></html>\n"
    (root / "review.html").write_text(document, encoding="utf-8")
    selection_path.write_text(json.dumps(selection, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("Review report:", root / "review.html")
    print("Items:", len(rows), "(no review verdicts were set automatically)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/musixqa_mcq_sample")
    parser.add_argument("--train-file", default="train.jsonl")
    parser.add_argument("--holdout-file", default="holdout.jsonl")
    parser.add_argument("--per-category", type=int, default=None)
    parser.add_argument("--holdout-per-category", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    prepare(args.data_dir, args.train_file, args.holdout_file,
            args.per_category, args.holdout_per_category, args.seed)
