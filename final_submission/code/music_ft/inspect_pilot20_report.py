"""Extract report tables as evidence. HTML/log text is never executed."""

import argparse
from html.parser import HTMLParser
from pathlib import Path
import re
import shutil

from build_musixqa import sha256, write_json


class Tables(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tables, self.table, self.row, self.cell = [], None, None, None

    def handle_starttag(self, tag, attrs):
        if tag == "table":
            self.table = []
        elif tag == "tr" and self.table is not None:
            self.row = []
        elif tag in ("td", "th") and self.row is not None:
            self.cell = []

    def handle_data(self, data):
        if self.cell is not None:
            self.cell.append(data)

    def handle_endtag(self, tag):
        if tag in ("td", "th") and self.cell is not None:
            self.row.append(" ".join("".join(self.cell).split()))
            self.cell = None
        elif tag == "tr" and self.row is not None:
            self.table.append(self.row)
            self.row = None
        elif tag == "table" and self.table is not None:
            self.tables.append(self.table)
            self.table = None


def extract(path, out):
    source = Path(path)
    parser = Tables()
    parser.feed(source.read_text(encoding="utf-8"))
    scores = next(table for table in parser.tables if table[0][0] == "카테고리")
    items = next(table for table in parser.tables if table[0][:2] == ["id", "카테고리"])
    errors = next(table for table in parser.tables if table[0][:3] == ["id", "정답", "모델 답"])
    fields = ["n", "reported_baseline_correct", "exact_correct", "semantic_correct", "greedy_exact_correct"]
    categories = {row[0]: {key: int(re.match(r"\d+", value)[0]) for key, value in zip(fields, row[1:])}
                  for row in scores[1:] if row[0] != "합계"}
    result = {
        "source_filename": source.name, "source_sha256": sha256(source),
        "evaluation_date": "2026-10-20", "model": "Qwen/Qwen3-VL-4B-Instruct",
        "input_format": "open-ended diagnostic questions", "runs_per_setting": 1,
        "categories": categories,
        "totals": {key: sum(row[key] for row in categories.values()) for key in fields},
        "known_false_positives": [dict(zip(errors[0], row)) for row in errors[1:]],
        "items": [dict(zip(items[0], row)) for row in items[1:]],
        "interpretation_notes": [
            "A01 '#' denotes sharp but the report counts it wrong; accepting it gives 9/20 semantic correctness, unlike the invalid original 9/20.",
            "One run and very small category counts support a provisional priority, not optimized mixture weights.",
            "No MeasureCount category is tested. Duration/rest/ornament failures are not covered by the current four-category generator.",
            "Pilot items are diagnostic evidence used for recipe design, not an untouched test set after that design choice.",
        ],
    }
    write_json(out, result)
    reference = Path("data/reference")
    reference.mkdir(parents=True, exist_ok=True)
    destination = reference / source.name
    if source.resolve() != destination.resolve():
        shutil.copyfile(source, destination)
    print("Totals:", result["totals"])
    print("Report SHA256:", result["source_sha256"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report")
    parser.add_argument("--out", default="reports/pilot20_recipe_evidence.json")
    args = parser.parse_args()
    extract(args.report, args.out)
