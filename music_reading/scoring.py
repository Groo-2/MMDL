"""악보 판독 pilot20 채점 (규칙 기반, judge 모델 없음).

- strict : 최종 답을 정규화한 뒤 정답과 완전히 같을 때만 정답
- lenient: 문장형 답변도 허용. 같은 category의 정답 라벨 중 응답에 들어 있는 가장 긴 라벨을 예측으로 보고 비교
"""
import re

WORD_NUM = {'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4', 'five': '5',
            'six': '6', 'seven': '7', 'eight': '8', 'nine': '9'}
SYNONYMS = {  # 정답 표기와 같은 뜻인 흔한 다른 표기 (정규화 후 기준)
    'g clef': 'treble clef', 'f clef': 'bass clef',
    'crotchet': 'quarter note', 'minim': 'half note', 'semibreve': 'whole note', 'quaver': 'eighth note',
    'dotted crotchet': 'dotted quarter note',
    'crotchet rest': 'quarter rest', 'quaver rest': 'eighth rest',
    'acciaccatura': 'acciaccatura', 'lower mordent': 'mordent',
    'sharp sign': 'sharp', 'flat sign': 'flat', 'natural sign': 'natural',
    'eighth': 'eighth note', 'quarter': 'quarter note', 'half': 'half note', 'whole': 'whole note',
}
PITCH_RE = re.compile(r'\b([A-Ga-g])\s*([#♯b♭]?)\s*(-?\d)\b')
TIME_RE = re.compile(r'\b(\d{1,2})\s*/\s*(\d{1,2})\b')


def extract_final(text: str) -> str:
    """'Answer: xxx' 형식이 있으면 마지막 것만, 없으면 마지막 비어 있지 않은 줄."""
    text = (text or '').strip()
    m = re.findall(r'answer\s*[:：]\s*(.+)', text, flags=re.I)
    if m:
        return m[-1].strip()
    lines = [l for l in text.splitlines() if l.strip()]
    return lines[-1].strip() if lines else ''


def norm(s: str) -> str:
    s = (s or '').lower().replace('♯', '#').replace('♭', 'b')
    s = re.sub(r'[*_`"\'.,;:!?()\[\]]', ' ', s)
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return SYNONYMS.get(s, s)


def norm_pitch(s: str):
    m = None
    for m in PITCH_RE.finditer(s or ''):
        pass
    if not m:
        return None
    letter, acc, octv = m.groups()
    return letter.upper() + {'#': '#', '♯': '#', 'b': 'b', '♭': 'b', '': ''}[acc] + octv


def label_sets(rows):
    """category별 정답 라벨 집합 (lenient 매칭용). 이 집합은 정답 비교 후보일 뿐 모델에 주지 않는다."""
    out = {}
    for r in rows:
        out.setdefault(r['category'], set()).add(norm(r['answer']))
    # clef / duration 처럼 라벨 간 포함관계가 있는 것을 위해 같은 문제 유형끼리 합침
    return out


def score_one(row, response, labels_by_cat):
    gold, cat = row['answer'], row['category']
    final = extract_final(response)

    if cat in ('pitch', 'pitch_bass'):
        pred, g = norm_pitch(final) or norm_pitch(response), norm_pitch(gold)
        ok = pred is not None and pred == g
        return {'pred': pred or final, 'strict': ok, 'lenient': ok}

    if cat == 'key_signature':
        def num(t):
            t = norm(t)
            for w, d in WORD_NUM.items():
                t = re.sub(rf'\b{w}\b', d, t)
            n = re.findall(r'\d+', t)
            return n[0] if n else None
        pred = num(final)
        strict = norm(final) == norm(gold)
        return {'pred': pred or final, 'strict': strict, 'lenient': pred is not None and pred == norm(gold)}

    if cat == 'time_signature':
        def ts(t):
            m = TIME_RE.search(t or '')
            return f'{m.group(1)}/{m.group(2)}' if m else None
        pred = ts(final) or ts(response)
        strict = norm(final) == norm(gold)
        return {'pred': pred or final, 'strict': strict, 'lenient': pred is not None and pred == gold}

    # 라벨형 (clef, duration, accidental, rest, ornament)
    strict = norm(final) == norm(gold)
    text = norm(final) or norm(response)
    hits = [l for l in labels_by_cat.get(cat, ()) if re.search(rf'\b{re.escape(l)}\b', text)]
    pred = max(hits, key=len) if hits else norm(final)
    return {'pred': pred, 'strict': strict, 'lenient': pred == norm(gold)}
