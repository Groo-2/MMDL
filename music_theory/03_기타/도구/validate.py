#!/usr/bin/env python3
"""Independent checks: music21 oracle + fractions + schema/split invariants."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
from fractions import Fraction
from music21 import chord, duration, interval, key, pitch, roman, scale
from _layout import PROJECT_ROOT,Layout

ROOT=PROJECT_ROOT


def m21pitch(text):
    return pitch.Pitch(text.replace('b','-'))


def display(p):
    return p.nameWithOctave.replace('-','b')


def oracle(row):
    s=row['spec'];task=row['task']
    if task in ('interval','transposition'):
        a=m21pitch(s['start']);b=m21pitch(s['end'])
        if task=='interval':
            measured=interval.Interval(a,b)
            assert measured.name==s['code'],(measured.name,s)
            return measured.niceName.replace('Double-octave','double octave')
        code=s['code'][0]+('-' if s['direction']<0 else '')+s['code'][1:]
        return display(interval.Interval(code).transposePitch(a))
    if task.startswith('chord_'):
        root=m21pitch(s['root']);quality=s['quality']
        formulas={
            'major triad':(0,4,7),'minor triad':(0,3,7),'diminished triad':(0,3,6),
            'augmented triad':(0,4,8),'dominant seventh':(0,4,7,10),
            'major seventh':(0,4,7,11),'minor seventh':(0,3,7,10),
            'half-diminished seventh':(0,3,6,10),'diminished seventh':(0,3,6,9)}
        actual=tuple(round(m21pitch(p).ps-root.ps) for p in s['pitches'])
        assert actual==formulas[quality],(actual,s)
        # Independently check each written chord member's generic interval.
        for p,num in zip(s['pitches'],[1,3,5,7]):
            assert interval.Interval(root,m21pitch(p)).generic.undirected==num
        if task=='chord_spelling':
            return ' '.join(display(m21pitch(p)) for p in s['pitches'])
        c=chord.Chord([m21pitch(p) for p in s['voicing']])
        c.root(root)
        if task=='chord_inversion':
            return ['root position','first inversion','second inversion','third inversion'][c.inversion()]
        members=sorted({(round(m21pitch(p).ps-root.ps)%12) for p in s['voicing']})
        matches=[name for name,formula in formulas.items() if sorted(formula)==members]
        assert len(matches)==1,matches
        return matches[0]
    if task.startswith('scale_'):
        classes={'major':scale.MajorScale,'natural minor':scale.MinorScale,
                 'harmonic minor':scale.HarmonicMinorScale,'ascending melodic minor':scale.MelodicMinorScale,
                 'Dorian':scale.DorianScale,'Phrygian':scale.PhrygianScale,'Lydian':scale.LydianScale,
                 'Mixolydian':scale.MixolydianScale,'Locrian':scale.LocrianScale}
        obj=classes[s['mode']](m21pitch(s['root']))
        values=[display(obj.pitchFromDegree(i)) for i in range(1,8)]
        assert values==s['tones'],(values,s)
        return values[s['degree']-1] if task=='scale_degree' else ' '.join(values)
    if task in ('duration_sum','measure_completion'):
        type_names={'whole':'whole','half':'half','quarter':'quarter','eighth':'eighth','sixteenth':'16th'}
        values=[]
        for name in s['names']:
            dots=int(name.startswith('dotted '));base=name.removeprefix('dotted ')
            d=duration.Duration(type=type_names[base],dots=dots)
            values.append(Fraction(d.quarterLength))
        assert [str(v) for v in values]==s['values']
        total=sum(values,Fraction())
        if task=='measure_completion':
            total=Fraction(s['meter'][0]*4,s['meter'][1])-total
            assert total>0
        return str(total)+' quarter-note units'
    if task=='tempo':
        # Independent formulation: seconds per whole note, times total whole notes.
        seconds_per_whole=Fraction(60*4,s['bpm'])/Fraction(s['beat_unit'])
        total_whole=Fraction(s['meter'][0]*s['measures'],s['meter'][1])
        return str(total_whole*seconds_per_whole)+' seconds'
    if task=='roman_numeral':
        k=key.Key(s['tonic'].replace('b','-'),'major')
        c=chord.Chord([m21pitch(p) for p in s['voicing']])
        figure=roman.romanNumeralFromChord(c,k).figure.replace('o','°')
        return figure[:-2]+'6/4' if figure.endswith('64') else figure
    if task=='cadence':
        rules={(5,1):'Authentic cadence',(4,1):'Plagal cadence',(2,5):'Half cadence',(5,6):'Deceptive cadence'}
        assert tuple(s['degrees'][-2:])==tuple(s['final_pair'])
        return rules[tuple(s['degrees'][-2:])]
    if task=='concept':
        card=s['card']
        if card['origin']=='interval_formula_card':
            iv=interval.Interval(card['code'])
            assert f'{iv.semitones} semitones' in card['definition']
        elif card['origin']=='chord_formula_card':
            codes={'major triad':['P1','M3','P5'],'minor triad':['P1','m3','P5'],
                   'diminished triad':['P1','m3','d5'],'augmented triad':['P1','M3','A5'],
                   'dominant seventh':['P1','M3','P5','m7'],'major seventh':['P1','M3','P5','M7'],
                   'minor seventh':['P1','m3','P5','m7'],'half-diminished seventh':['P1','m3','d5','m7'],
                   'diminished seventh':['P1','m3','d5','d7']}[card['quality']]
            expected=', '.join(interval.Interval(c).niceName.lower() for c in codes)
            assert card['definition']==f'From its root, a {card["quality"]} uses {expected}.'
        elif card['origin']=='scale_formula_card':
            classes={'major':scale.MajorScale,'natural minor':scale.MinorScale,
                     'harmonic minor':scale.HarmonicMinorScale,'ascending melodic minor':scale.MelodicMinorScale,
                     'Dorian':scale.DorianScale,'Phrygian':scale.PhrygianScale,'Lydian':scale.LydianScale,
                     'Mixolydian':scale.MixolydianScale,'Locrian':scale.LocrianScale}
            obj=classes[card['mode']]('C4')
            numbers=[int(x) for x in re.findall(r'\d+',card['definition'])]
            expected=[round(obj.pitchFromDegree(i).ps-60) for i in range(1,8)]
            assert numbers==expected,(numbers,expected)
        return card['term'] if s['view']=='identify' else card['definition']
    raise ValueError(task)


def clean(text):
    # Harmless formatter differences only, not enharmonic or interval equivalence.
    return re.sub(r'\s+',' ',str(text)).strip().lower()


def golden_checks():
    # Explicit gold standards independent of the generated JSONL.
    cases=[('C4','F#4','A4'),('C4','Gb4','d5'),('B3','C4','m2'),('C4','Db5','m9'),
           ('C#4','E4','m3'),('F4','B4','A4'),('C4','Bbb4','d7'),('C4','G5','P12')]
    for a,b,expected in cases:
        assert interval.Interval(m21pitch(a),m21pitch(b)).name==expected
    assert display(interval.Interval('M-3').transposePitch(m21pitch('C4')))=='Ab3'
    assert Fraction(duration.Duration(type='quarter',dots=2).quarterLength)==Fraction(7,4)
    assert roman.romanNumeralFromChord(chord.Chord(['F4','A4','C5']),key.Key('C')).figure=='IV'
    assert chord.Chord(['B3','D4','F4','G4']).inversion()==1
    return len(cases)+4


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--release',type=Path,default=ROOT)
    args=parser.parse_args();root=args.release;layout=Layout(root)
    records=[];errors=[];checked=Counter();families=defaultdict(set);schemas=0
    for split,n in [('train',3000),('dev',300),('test',300)]:
        rows=[json.loads(x) for x in layout.records(split).read_text().splitlines()]
        assert len(rows)==n,(split,len(rows))
        records.extend(rows)
    ids=set();semantic=set();prompts=set();bad_images=0
    for row in records:
        try:
            assert row['id'] not in ids;ids.add(row['id'])
            assert row['semantic_hash'] not in semantic;semantic.add(row['semantic_hash'])
            question=clean(row['messages'][1]['content'])
            assert question not in prompts;prompts.add(question)
            families[row['family_id']].add(row['split'])
            assert len(row['messages'])==3
            assert all(isinstance(m['content'],str) for m in row['messages'])
            assert [m['role'] for m in row['messages']]==['system','user','assistant']
            assert not any(k in row for k in ['image','images','abc_score'])
            assert row['messages'][2]['content'].startswith('Answer: ')
            if row['format']=='mcq':
                assert set(row['options'])==set('ABCD')
                assert len(set(map(clean,row['options'].values())))==4
                assert row['options'][row['answer_letter']]==row['answer_text']
            actual=oracle(row)
            if row['task']=='roman_numeral':
                assert actual==row['answer_text'],f'{actual!r} != {row["answer_text"]!r}'
            else:
                assert clean(actual)==clean(row['answer_text']),f'{actual!r} != {row["answer_text"]!r}'
            checked[row['task']]+=1;schemas+=1
        except Exception as e:
            errors.append({'id':row['id'],'task':row['task'],'error':repr(e)})
    assert all(len(s)==1 for s in families.values()),'family leakage'
    for split,n in [('train',3000),('dev',300),('test',300)]:
        rows=[r for r in records if r['split']==split]
        letters=Counter(r['answer_letter'] for r in rows if r['format']=='mcq')
        assert len(set(letters.values()))==1,letters
    coverage_report={}
    for split in ['train','dev','test']:
        subset=[r for r in records if r['split']==split]
        coverage_report[split]={}
        for task,field,required in [
            ('scale_spelling','mode',{'major','natural minor','harmonic minor','ascending melodic minor','Dorian','Phrygian','Lydian','Mixolydian','Locrian'}),
            ('scale_degree','mode',{'major','natural minor','harmonic minor','ascending melodic minor','Dorian','Phrygian','Lydian','Mixolydian','Locrian'}),
            ('chord_quality','quality',{'major triad','minor triad','diminished triad','augmented triad','dominant seventh','major seventh','minor seventh','half-diminished seventh','diminished seventh'}),
            ('chord_spelling','quality',{'major triad','minor triad','diminished triad','augmented triad','dominant seventh','major seventh','minor seventh','half-diminished seventh','diminished seventh'}),
            ('chord_inversion','inversion',{0,1,2,3}),
            ('interval','code',{'m2','M2','A2','m3','M3','P4','A4','d5','P5','A5','m6','M6','m7','M7','d7','P8','m9','M9','m10','M10','P11','A11','P12','m13','M13','m14','M14','P15'}),
            ('transposition','direction',{-1,1}),
        ]:
            actual={r['spec'][field] for r in subset if r['task']==task}
            missing=required-actual
            if missing:errors.append({'split':split,'task':task,'missing_subskills':list(missing)})
            coverage_report[split][task+'/'+field]={'required':len(required),'observed':len(actual),'missing':list(missing)}
    for split in ['dev','test']:
        rows=[json.loads(x) for x in (layout.evaluation/f'{split}_questions.jsonl').read_text().splitlines()]
        for r in rows:
            assert not any(k in r for k in ['answer_text','answer_letter','explanation','spec','family_id'])
            assert all(m['role']!='assistant' for m in r['messages'])
    for file in layout.training.glob('*.jsonl'):
        for line in file.read_text().splitlines():
            assert set(json.loads(line))=={'messages'},f'audit metadata in training export: {file}'
    report={'status':'PASS' if not errors else 'FAIL','records':len(records),'verified_records':schemas,
            'independent_oracle':'music21 and independent fractions calculations', 'checked_by_task':dict(checked),
            'golden_cases':golden_checks(),'cross_split_family_overlap':0,
            'conceptual_card_check':'metadata and formula checks; expert review of authored definitions still pending',
            'subskill_coverage':coverage_report,
            'human_expert_review':'NOT performed by this pipeline',
            'errors':errors}
    (layout.audit/'validation.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(report,indent=2,ensure_ascii=False))
    raise SystemExit(bool(errors))


if __name__=='__main__':main()
