"""SSMR 문항(build_ssmr.py가 만든 MMMU 형식 jsonl)을 vLLM 서버에 보내는 공용 코드.

select_ssmr.py(통과율 측정)와 lab/eval_holdout.py(hold-out 평가)가 함께 쓴다.
프롬프트는 mmmu_eval/prompting.py의 build_mmmu_pro_cot_content()를 그대로 써서 평가와 같은 문구·이미지 순서·JPEG q95를 보장한다.
"""

import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _mmmu_eval_dir():
    for cand in (HERE.parent / "baseline" / "mmmu_eval", HERE.parent / "mmmu_eval"):
        if cand.is_dir():
            return str(cand)
    raise SystemExit("mmmu_eval 폴더를 찾지 못했다 (../baseline/mmmu_eval 또는 ../mmmu_eval)")


sys.path.insert(0, _mmmu_eval_dir())
sys.path.insert(0, str(HERE))

from prompting import MAX_PIXELS, MIN_PIXELS, build_mmmu_pro_cot_content  # noqa: E402
from reward import score_one  # noqa: E402

import requests  # noqa: E402

CATEGORIES = ["Beat", "Chord", "Interval", "Scale"]


def load_rows(path, limit=None):
    rows = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
                if limit and len(rows) >= limit:
                    break
    return rows


def build_content(row, data_dir):
    """OpenAI chat content (텍스트 먼저, 이미지는 등장 순서대로)를 평가와 같은 함수로 만든다."""
    from PIL import Image

    sample = {
        "question": row["question"],
        "options": row["options"],
        "question_type": row["question_type"],
    }
    for key, rel in row["image_paths"].items():
        sample[key] = Image.open(Path(data_dir) / rel)
    content, _ = build_mmmu_pro_cot_content(sample)
    return content


def chat(api_base, model, content, *, n=1, max_tokens=2048, temperature=0.7, top_p=0.8, top_k=20,
         presence_penalty=None, repetition_penalty=1.0, seed=None, timeout=3600, retries=2):
    """한 요청으로 n개 응답을 받는다. 반환: [(text, finish_reason), ...]"""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "n": n,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "repetition_penalty": repetition_penalty,
        "max_tokens": max_tokens,
        # 평가 코드(call_model)와 같은 해상도 설정
        "mm_processor_kwargs": {"min_pixels": MIN_PIXELS, "max_pixels": MAX_PIXELS},
    }
    if presence_penalty is not None:
        payload["presence_penalty"] = presence_penalty
    if seed is not None:
        payload["seed"] = seed

    last_error = None
    for attempt in range(retries + 1):
        try:
            response = requests.post(f"{api_base}/chat/completions", json=payload, timeout=timeout)
            response.raise_for_status()
            choices = response.json()["choices"]
            return [((c["message"]["content"] or ""), c.get("finish_reason")) for c in choices]
        except Exception as e:  # noqa: BLE001
            last_error = e
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"요청 실패: {last_error!r}")
