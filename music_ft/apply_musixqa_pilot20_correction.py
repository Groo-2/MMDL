"""Publish a reviewed training variant for the verified ninth-bar wording fix."""

import csv
import hashlib
import json
from pathlib import Path

import build_musixqa as builder


ROOT = Path(__file__).resolve().parents[1] / "data" / "musixqa_mcq_pilot20_v1"
ITEM_ID = "musixqa_train_03736"
OLD_QUESTION = ("<image 1> Which sequence of pitches is written in bar 9 of the bass clef? "
                "Count each written note, including notes joined by ties.")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def write_jsonl(path, rows):
    if path.exists():
        raise FileExistsError(path)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    audit_path = ROOT / "generation_audit.jsonl"
    audit = read_jsonl(audit_path)
    trace = next(x for x in audit if x["id"] == ITEM_ID)
    assert trace["source_doc_id"] == "0027194"
    assert trace["source_answer"] == "Eb3 Eb3 Eb3 Eb3 Gb3"
    assert trace["output_question"] == OLD_QUESTION.removeprefix("<image 1> ")
    new_question = "<image 1> " + builder.output_question(trace) + " Count each written note, including notes joined by ties."
    if "9th written bar" not in new_question or "ignoring printed measure numbers" not in new_question:
        raise ValueError("Expected ordinal-bar wording was not generated")

    inputs = {"train.dedup.jsonl": "train.reviewed.jsonl",
              "train_recipe.dedup.jsonl": "train_recipe.reviewed.jsonl"}
    before_after = {}
    for src_name, dst_name in inputs.items():
        src, dst = ROOT / src_name, ROOT / dst_name
        rows = read_jsonl(src)
        matches = [x for x in rows if x["id"] == ITEM_ID]
        if len(matches) != 1 or matches[0]["question"] != OLD_QUESTION:
            raise ValueError(f"Unexpected source item in {src_name}")
        matches[0]["question"] = new_question
        write_jsonl(dst, rows)
        before_after[dst_name] = {"input_file": src_name, "input_sha256": digest(src),
                                  "output_sha256": digest(dst), "rows": len(rows)}

    reviewed_audit = [dict(x) for x in audit]
    next(x for x in reviewed_audit if x["id"] == ITEM_ID)["output_question"] = new_question.removeprefix("<image 1> ")
    write_jsonl(ROOT / "generation_audit.reviewed.jsonl", reviewed_audit)

    csv_path = ROOT / "manual_review.csv"
    with csv_path.open(encoding="utf-8-sig", newline="") as stream:
        reviews = list(csv.DictReader(stream))
    target = next(x for x in reviews if x["id"] == ITEM_ID)
    if target["verdict"] != "fix" or target["image_matches_answer"] != "no":
        raise ValueError("The original visual finding must be present")
    target.update(verdict="pass", image_matches_answer="yes", only_one_correct_option="yes",
                  notes=target["notes"] + " RESOLVED: The question now explicitly asks for the ninth written bar, ignoring printed measure numbers; the unchanged answer matches printed bar 10.")
    reviewed_csv = ROOT / "manual_review.reviewed.csv"
    if reviewed_csv.exists():
        raise FileExistsError(reviewed_csv)
    with reviewed_csv.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=reviews[0].keys())
        writer.writeheader()
        writer.writerows(reviews)

    html_path = ROOT / "review.html"
    html = html_path.read_text(encoding="utf-8")
    start = html.index(f'<article id="{ITEM_ID}">')
    end = html.index("</article>", start) + len("</article>")
    card = html[start:end]
    old_heading = f"<h3>{OLD_QUESTION.removeprefix('<image 1> ')}</h3>"
    if card.count(old_heading) != 1 or card.count("<strong>fix</strong>") != 1:
        raise ValueError("Unexpected review HTML card")
    card = card.replace(old_heading, f"<h3>{new_question.removeprefix('<image 1> ')}</h3>")
    card = card.replace("<strong>fix</strong>", "<strong>pass</strong>", 1)
    card = card.replace("<ol type=", "<p class='note'>수정 완료: 인쇄 번호와 관계없이 처음부터 센 아홉 번째 마디를 묻습니다. 정답과 선택지는 그대로입니다.</p><ol type=", 1)
    html = html[:start] + card + html[end:]
    html = html.replace("CSV 기준 검수 기록: fix 1, pass 199", "CSV 기준 검수 기록: pass 200", 1)
    html = html.replace('href="manual_review.csv"', 'href="manual_review.reviewed.csv"', 1)
    reviewed_html = ROOT / "review.reviewed.html"
    if reviewed_html.exists():
        raise FileExistsError(reviewed_html)
    reviewed_html.write_text(html, encoding="utf-8")

    report = {
        "item_id": ITEM_ID, "source_doc_id": "0027194", "source_qa_unchanged": True,
        "old_question": OLD_QUESTION, "new_question": new_question,
        "correct_option": "D", "correct_text": "Eb3 Eb3 Eb3 Eb3 Gb3",
        "verified_mapping": "ninth written bar from the beginning is printed measure 10; printed measure 9 has Ab2 C3 C3 C3 Ab3 C3",
        "method": "Question wording only; image, options, answer, row order and holdout remain unchanged",
        "training_files": before_after,
        "audit_file": "generation_audit.reviewed.jsonl", "audit_sha256": digest(ROOT / "generation_audit.reviewed.jsonl"),
        "review_csv": "manual_review.reviewed.csv", "review_csv_sha256": digest(reviewed_csv),
        "review_html": "review.reviewed.html", "review_html_sha256": digest(reviewed_html),
        "mmmu_image_check": "Prior MMMU Music image deduplication remains applicable because no image bytes changed",
        "human_review": False,
    }
    report_path = ROOT / "correction_report.json"
    if report_path.exists():
        raise FileExistsError(report_path)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("Corrected", ITEM_ID, "in", len(inputs), "training files")


if __name__ == "__main__":
    main()
