"""Small, explicit Western music-theory rules. No benchmark material is used."""
from fractions import Fraction
import hashlib
import json
import re

LETTERS = 'CDEFGAB'
NATURAL = [0, 2, 4, 5, 7, 9, 11]
QUALITIES = {'P': 'Perfect', 'M': 'Major', 'm': 'Minor', 'A': 'Augmented', 'd': 'Diminished'}
ORDINALS = ['unison', 'second', 'third', 'fourth', 'fifth', 'sixth', 'seventh',
            'octave', 'ninth', 'tenth', 'eleventh', 'twelfth', 'thirteenth', 'fourteenth', 'double octave']
CHORDS = {
    'major triad': ['P1', 'M3', 'P5'], 'minor triad': ['P1', 'm3', 'P5'],
    'diminished triad': ['P1', 'm3', 'd5'], 'augmented triad': ['P1', 'M3', 'A5'],
    'dominant seventh': ['P1', 'M3', 'P5', 'm7'],
    'major seventh': ['P1', 'M3', 'P5', 'M7'],
    'minor seventh': ['P1', 'm3', 'P5', 'm7'],
    'half-diminished seventh': ['P1', 'm3', 'd5', 'm7'],
    'diminished seventh': ['P1', 'm3', 'd5', 'd7'],
}
SCALES = {
    'major': [0, 2, 4, 5, 7, 9, 11],
    'natural minor': [0, 2, 3, 5, 7, 8, 10],
    'harmonic minor': [0, 2, 3, 5, 7, 8, 11],
    'ascending melodic minor': [0, 2, 3, 5, 7, 9, 11],
    'Dorian': [0, 2, 3, 5, 7, 9, 10], 'Phrygian': [0, 1, 3, 5, 7, 8, 10],
    'Lydian': [0, 2, 4, 6, 7, 9, 11], 'Mixolydian': [0, 2, 4, 5, 7, 9, 10],
    'Locrian': [0, 1, 3, 5, 6, 8, 10],
}
ROOTS = [letter + accidental for letter in LETTERS for accidental in ('', '#', 'b')]
INTERVAL_CODES = ['P1','m2','M2','A2','m3','M3','P4','A4','d5','P5','A5','m6','M6',
                  'm7','M7','d7','P8','m9','M9','m10','M10','P11','A11','P12','m13','M13','m14','M14','P15']
INV = ['root position', 'first inversion', 'second inversion', 'third inversion']


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def parse_pitch(value):
    m = re.fullmatch(r'([A-G])([#b]*)(-?\d+)', value)
    if not m:
        raise ValueError(value)
    letter, accidental, octave = m.groups()
    idx = LETTERS.index(letter)
    alter = accidental.count('#') - accidental.count('b')
    return idx, alter, int(octave)


def midi(value):
    idx, alter, octave = parse_pitch(value)
    return 12 * (octave + 1) + NATURAL[idx] + alter


def pitch_name(value):
    return re.sub(r'-?\d+$', '', value)


def accidental_text(alter):
    return '#' * alter if alter >= 0 else 'b' * -alter


def interval_semitones(code):
    quality, num = code[0], int(code[1:])
    simple = (num - 1) % 7
    base = NATURAL[simple] + 12 * ((num - 1) // 7)
    perfect = simple in (0, 3, 4)
    delta = {'P': 0, 'M': 0, 'm': -1, 'A': 1, 'd': -1 if perfect else -2}[quality]
    return base + delta


def interval_label(code):
    return f'{QUALITIES[code[0]]} {ORDINALS[int(code[1:]) - 1]}'


def transpose(start, code, direction=1):
    idx, _, octave = parse_pitch(start)
    number = int(code[1:])
    target_diatonic = octave * 7 + idx + direction * (number - 1)
    out_oct, out_idx = divmod(target_diatonic, 7)
    target_midi = midi(start) + direction * interval_semitones(code)
    alter = target_midi - (12 * (out_oct + 1) + NATURAL[out_idx])
    return LETTERS[out_idx] + accidental_text(alter) + str(out_oct)


def spelled_chord(root, quality):
    return [transpose(root, code) for code in CHORDS[quality]]


def spelled_scale(root, mode):
    idx, _, octave = parse_pitch(root)
    out = []
    for degree, semi in enumerate(SCALES[mode]):
        out_oct, out_idx = divmod(octave * 7 + idx + degree, 7)
        alter = midi(root) + semi - (12 * (out_oct + 1) + NATURAL[out_idx])
        out.append(LETTERS[out_idx] + accidental_text(alter) + str(out_oct))
    return out


def no_extreme(pitches):
    return all(abs(parse_pitch(p)[1]) <= 2 for p in pitches)


def pair_group(a, b):
    # Both directions, any common octave shift, and simple/compound versions
    # of the same spelled pitch-class pair stay together. Letter spelling is
    # retained so that augmented fourth and diminished fifth remain distinct.
    normalized = sorted(pitch_name(p) for p in (a,b))
    return 'interval:' + digest(normalized)


def frac(value):
    return Fraction(str(value))


def rational(value):
    value = Fraction(value)
    return str(value.numerator) if value.denominator == 1 else f'{value.numerator}/{value.denominator}'


def normalize(text):
    import unicodedata
    text = unicodedata.normalize('NFKC', text).lower()
    text = text.replace('♯', '#').replace('♭', 'b')
    return ' '.join(re.findall(r'[a-z0-9#]+', text))
