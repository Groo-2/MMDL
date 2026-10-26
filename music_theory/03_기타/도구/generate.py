#!/usr/bin/env python3
"""Pure text music theory: synthetic facts + deterministic rule applications."""
import argparse
from collections import Counter, defaultdict
from fractions import Fraction
import itertools
import json
from pathlib import Path
import random

from concepts import cards
from theory import (CHORDS, INTERVAL_CODES, INV, ROOTS, SCALES, digest, frac,
                    interval_label, interval_semitones, midi, no_extreme,
                    normalize, pair_group, pitch_name, rational, spelled_chord,
                    spelled_scale, transpose)
from _layout import PROJECT_ROOT
from baseline_contract import sft_messages, PROTOCOL, assert_baseline

ROOT = PROJECT_ROOT
SEED = 20261026
QUOTAS = {'interval':360, 'transposition':360, 'chord_quality':180,
          'chord_spelling':120, 'chord_inversion':180, 'scale_degree':270,
          'scale_spelling':120, 'duration_sum':390, 'measure_completion':180,
          'tempo':180, 'roman_numeral':270, 'cadence':180, 'concept':210}
ROMANS = ['I','ii','iii','IV','V','vi','vii°']
DURATION_NAMES = {'whole':Fraction(4), 'half':Fraction(2), 'quarter':Fraction(1),
                  'eighth':Fraction(1,2), 'sixteenth':Fraction(1,4),
                  'dotted half':Fraction(3), 'dotted quarter':Fraction(3,2),
                  'dotted eighth':Fraction(3,4)}
CADENCES = {'Authentic cadence':(5,1), 'Plagal cadence':(4,1),
            'Half cadence':(2,5), 'Deceptive cadence':(5,6)}


def candidate(task, category, question, answer, explanation, spec, group, distractors, difficulty='medium'):
    return dict(task=task, category=category, question=question, answer_text=answer,
                explanation=explanation, spec=spec, family_id=group,
                semantic_hash=digest([task, spec]), distractors=list(dict.fromkeys(x for x in distractors if x != answer)),
                difficulty=difficulty)


def interval_pool():
    for root in ROOTS:
        for code in INTERVAL_CODES:
            for direction in (-1,1):
                start = root+'4'
                end = transpose(start,code,direction)
                if not no_extreme([end]) or midi(start)==midi(end):
                    continue
                group = pair_group(start,end)
                direct = 'ascending' if direction==1 else 'descending'
                name = interval_label(code)
                spec = dict(start=start,end=end,code=code,direction=direction)
                explanation = (f'The written letter distance is {int(code[1:])}; the absolute pitch distance is '
                               f'{interval_semitones(code)} semitones, giving a {name.lower()}.')
                nearby=sorted([c for c in INTERVAL_CODES if c!=code],
                              key=lambda c:(abs(interval_semitones(c)-interval_semitones(code)),abs(int(c[1:])-int(code[1:]))))[:6]
                yield candidate('interval','intervals',f'What is the spelled {direct} interval from {start} to {end}? Give the full interval name; retain compound interval numbers.',
                                name, explanation,spec,group,[interval_label(c) for c in nearby], 'hard' if int(code[1:])>8 or code[0] in 'Ad' else 'medium')
                choices = [transpose(start,c,direction) for c in nearby]
                choices = [p for p in choices if no_extreme([p])]
                yield candidate('transposition','intervals',f'Transpose {start} {"up" if direction==1 else "down"} by a {name.lower()}. Give the correctly spelled pitch including its octave.',
                                end,f'A {name.lower()} changes the letter position by {int(code[1:])-1} steps and the pitch by {interval_semitones(code)} semitones. The result is {end}.',
                                spec,group,choices, 'hard' if int(code[1:])>8 or code[0] in 'Ad' else 'medium')


def chord_pool():
    for root in ROOTS:
        for quality in CHORDS:
            pitches = spelled_chord(root+'3',quality)
            if not no_extreme(pitches):
                continue
            family='chord:'+digest([root,quality])
            peer_qualities=[q for q in CHORDS if len(CHORDS[q])==len(CHORDS[quality])]
            alternatives=[' '.join(spelled_chord(root+'3',q)) for q in peer_qualities if q!=quality]
            spec=dict(root=root+'3',quality=quality,pitches=pitches)
            yield candidate('chord_spelling','chords',f'Spell a {root} {quality} in close root position, beginning on {root}3. Include octave numbers, ordered from low to high.',
                            ' '.join(pitches), f'The root-relative interval formula gives {", ".join(pitches)}.',spec,family,alternatives)
            for inv in range(len(pitches)):
                voicing=pitches[:]
                for _ in range(inv):
                    moved=voicing.pop(0)
                    voicing.append(transpose(moved,'P8'))
                qspec={**spec,'inversion':inv,'voicing':voicing}
                prompt=f'The pitches in a tertian chord are {", ".join(voicing)}, ordered from lowest to highest. '
                yield candidate('chord_quality','chords',prompt+'What is its quality? Use the written spelling.',
                                quality, f'Rearranging these tones into thirds gives root {root} and the {quality} pattern.',qspec,family,peer_qualities)
                # Augmented triads and diminished sevenths can be root-ambiguous
                # enharmonically; explicitly state the intended root.
                yield candidate('chord_inversion','chords',prompt+f'The intended root is {root}. Identify the inversion.',
                                INV[inv],f'The lowest note is chord member {inv+1}, so the chord is in {INV[inv]}.',
                                qspec,family,INV, 'easy' if inv==0 else 'medium')


def scale_pool():
    for root in ROOTS:
        for mode in SCALES:
            tones=spelled_scale(root+'4',mode)
            if not no_extreme(tones):
                continue
            family='scale:'+digest([root,mode])
            spec=dict(root=root+'4',mode=mode,tones=tones)
            alternatives=[' '.join(spelled_scale(root+'4',m)) for m in SCALES if m!=mode]
            yield candidate('scale_spelling','scales',f'List degrees 1 through 7 of the ascending {root} {mode} scale, starting on {root}4. Include octave numbers and omit the repeated tonic.',
                            ' '.join(tones),f'Applying the {mode} semitone pattern while advancing one letter per degree gives {", ".join(tones)}.',spec,family,alternatives,
                            'medium' if mode in ['major','natural minor'] else 'hard')
            for degree in range(2,8):
                p=tones[degree-1]
                ds=[x for x in tones if x!=p]
                ds+=[transpose(p,'m2'),transpose(p,'M2')]
                yield candidate('scale_degree','scales',f'What is scale degree {degree} of the ascending {root} {mode} scale whose tonic is {root}4? Give pitch spelling and octave.',
                                p,f'Degree {degree} lies {SCALES[mode][degree-1]} semitones above {root}4 and is spelled {p}.',
                                {**spec,'degree':degree},family,ds,'easy' if mode=='major' else 'medium')


def rhythm_pool():
    rng=random.Random(SEED+1)
    for _ in range(16000):
        names=[rng.choice(list(DURATION_NAMES)) for _ in range(rng.randint(3,7))]
        values=[DURATION_NAMES[x] for x in names]
        # Canonicalize ordering and uniform rescaling: reordered and proportionally
        # identical rhythm examples remain in one family, even across tasks.
        shortest=min(values)
        signature=sorted(str(x/shortest) for x in values)
        family='rhythm:'+digest(signature)
        total=sum(values,Fraction())
        desc=', '.join(x+' note' for x in names)
        spec=dict(names=names,values=[str(v) for v in values])
        answer=rational(total)+' quarter-note units'
        distractors=[rational(total+d)+' quarter-note units' for d in [Fraction(-1),Fraction(-1,2),Fraction(1,4),Fraction(1,2),Fraction(1)] if total+d>0]
        yield candidate('duration_sum','rhythm',f'Add the durations of this successive sequence: {desc}. Rests of the same written durations would count equally. Give the total in quarter-note units.',
                        answer,f'The durations are {" + ".join(rational(v) for v in values)} quarter-note units; the sum is {rational(total)}.',spec,family,distractors)
        for numerator,denominator in [(3,4),(4,4),(5,4),(6,8),(7,8),(9,8),(12,8)]:
            capacity=Fraction(numerator*4,denominator)
            if total>=capacity:
                continue
            missing=capacity-total
            answer=rational(missing)+' quarter-note units'
            ds=[rational(missing+d)+' quarter-note units' for d in [Fraction(-1,2),Fraction(1,4),Fraction(1,2),Fraction(1),Fraction(2)] if missing+d>0]
            yield candidate('measure_completion','rhythm',f'One full {numerator}/{denominator} measure contains {desc}. No pickup or meter change is intended. How much duration remains to fill the measure, in quarter-note units?',
                            answer,f'The measure holds {rational(capacity)} quarter-note units. Subtracting {rational(total)} leaves {rational(missing)}.',
                            {**spec,'meter':[numerator,denominator]},family,ds,'medium')


def tempo_pool():
    for numerator,denominator in [(2,4),(3,4),(4,4),(5,4),(3,8),(6,8),(7,8),(9,8),(12,8)]:
        for bpm in range(48,145,6):
            for measures in range(2,17):
                for beat_unit,unit in [('quarter note',Fraction(1)),('dotted quarter note',Fraction(3,2)),('half note',Fraction(2))]:
                    units=Fraction(numerator*4,denominator)*measures
                    beats=units/unit
                    seconds=beats*60/bpm
                    family='tempo:'+digest([numerator,denominator,bpm,str(unit),measures])
                    answer=rational(seconds)+' seconds'
                    ds=[rational(seconds*f)+' seconds' for f in [Fraction(1,2),Fraction(2),Fraction(3,2),Fraction(2,3)]]
                    yield candidate('tempo','rhythm',f'A passage has {measures} complete bars of {numerator}/{denominator}. The metronome marking is {beat_unit} = {bpm} per minute. At constant tempo and with no repeats or pauses, how long is the passage? Give exact seconds as an integer or fraction.',
                                    answer,f'There are {rational(units)} quarter-note units, or {rational(beats)} specified metronome beats. Time is {rational(beats)} × 60/{bpm} = {rational(seconds)} seconds.',
                                    dict(meter=[numerator,denominator],bpm=bpm,measures=measures,beat_unit=str(unit)),family,ds,'hard')


def harmony_pool():
    qualities=['major triad','minor triad','minor triad','major triad','major triad','minor triad','diminished triad']
    for tonic in ROOTS:
        scale=spelled_scale(tonic+'3','major')
        if not no_extreme(scale):
            continue
        for degree in range(1,8):
            quality=qualities[degree-1]
            root=scale[degree-1]
            pitches=spelled_chord(root,quality)
            if not no_extreme(pitches):
                continue
            family='chord:'+digest([pitch_name(root),quality])
            for inv in range(3):
                voicing=pitches[:]
                for _ in range(inv):
                    voicing.append(transpose(voicing.pop(0),'P8'))
                ans=ROMANS[degree-1]+['','6','6/4'][inv]
                ds=[r+['','6','6/4'][inv] for r in ROMANS if r!=ROMANS[degree-1]]
                ds += [ROMANS[degree-1]+suffix for suffix in ['','6','6/4'] if ROMANS[degree-1]+suffix!=ans]
                yield candidate('roman_numeral','harmony',f'In {tonic} major, analyze the diatonic triad with notes {", ".join(voicing)} from bass upward. Give its Roman numeral and inversion figure; root position has no figure, first inversion uses 6, second inversion uses 6/4.',
                                ans,f'The root {pitch_name(root)} is degree {degree} of {tonic} major. The chord is in {INV[inv]}, giving {ans}.',
                                dict(tonic=tonic,degree=degree,inversion=inv,voicing=voicing,root=root,quality=quality),family,ds,'hard')
        for label,pair in CADENCES.items():
            for prefix in [(1,2),(6,4),(1,4),(3,6),(6,2),(4,2),(1,6),(2,3)]:
                sequence=prefix+pair
                names=[]
                for degree in sequence:
                    q=qualities[degree-1]
                    names.append(pitch_name(scale[degree-1])+' '+q)
                final=' → '.join(names)
                yield candidate('cadence','harmony',f'In {tonic} major, a phrase ends with these root-position chords: {final}. Classify the cadence from the final two chords. Do not distinguish perfect from imperfect authentic cadences.',
                                label,f'The final harmonic degrees are {ROMANS[pair[0]-1]} → {ROMANS[pair[1]-1]}, the pattern for a {label.lower()}.',
                                dict(tonic=tonic,degrees=sequence,final_pair=pair),
                                'cadence:'+digest([tonic,label]),list(CADENCES),'medium')


def concept_pool():
    for card in cards():
        term=card['term']; definition=card['definition']
        family='concept:'+digest(term)
        metadata={'card':card}
        other_terms=[x['term'] for x in cards() if x['term']!=term]
        yield candidate('concept','conceptual_knowledge',f'Explain the music-theory term "{term}" in one concise sentence.',
                        definition,definition,{**metadata,'view':'define'},family,
                        [x['definition'] for x in cards() if x['term']!=term],'easy')
        yield candidate('concept','conceptual_knowledge',f'Which music-theory term matches this description? {definition}',
                        term,f'This is the definition of {term}.',{**metadata,'view':'identify'},family,other_terms,'easy')
        yield candidate('concept','conceptual_knowledge',f'Correct this mistaken claim about {term}: "{card["false_statement"]}"',
                        definition,'Use the standard definition given in the correction.',{**metadata,'view':'correct'},family,
                        [x['definition'] for x in cards() if x['term']!=term],'medium')


def split_of(family):
    value=int(digest(['family_split_v2',family])[:12],16)%1000
    return 'train' if value<800 else ('dev' if value<900 else 'test')


def coverage(item):
    s=item['spec'];task=item['task'];features=set()
    keys={'interval':['code','direction'],'transposition':['code','direction'],
          'chord_quality':['quality','inversion'],'chord_spelling':['quality'],
          'chord_inversion':['quality','inversion'],'scale_degree':['mode','degree'],
          'scale_spelling':['mode'],'roman_numeral':['degree','inversion'],
          'cadence':['final_pair'],'tempo':['meter','beat_unit'],
          'measure_completion':['meter'],'concept':['view']}.get(task,[])
    for k in keys:
        features.add(k+':'+json.dumps(s[k],sort_keys=True))
    if task=='duration_sum':
        features.update('duration:'+name for name in s['names'])
    return features


def stratified_select(pool,n,rng):
    # Greedily cover every requested subskill first, then prefer new families.
    # This prevents a small test set accidentally omitting major/minor scales or
    # a chord quality just because the top-level category quota was satisfied.
    remaining=pool[:];rng.shuffle(remaining)
    needed=set().union(*(coverage(item) for item in remaining))
    chosen=[];family_counts=Counter()
    while needed and len(chosen)<n:
        best=max(range(len(remaining)),key=lambda i:len(coverage(remaining[i])&needed))
        item=remaining.pop(best)
        gain=coverage(item)&needed
        if not gain:break
        chosen.append(item);family_counts[item['family_id']]+=1;needed-=gain
    if needed:raise RuntimeError(f'Insufficient quota for subskill coverage: {needed}')
    while len(chosen)<n:
        best=min(range(len(remaining)),key=lambda i:family_counts[remaining[i]['family_id']])
        item=remaining.pop(best);chosen.append(item);family_counts[item['family_id']]+=1
    return chosen


def grouped_split_assignment(items):
    by_group=defaultdict(list)
    for item in items:by_group[item['family_id']].append(item)
    feature_map={g:set(task_item['task']+'/'+feature for task_item in vals for feature in coverage(task_item))
                 for g,vals in by_group.items()}
    required=set().union(*feature_map.values())
    task_counts={g:Counter(i['task'] for i in vals) for g,vals in by_group.items()}
    rng=random.Random(SEED+99);available=sorted(by_group);rng.shuffle(available)
    assigned={}
    # Reserve disjoint families that cover all subskills before hash-filling.
    # Hashing alone can place every example of a rare quality into one split.
    for split in ['dev','test']:
        needed=required.copy();counts=Counter()
        while needed:
            best=max(range(len(available)),key=lambda i:len(feature_map[available[i]]&needed))
            group=available.pop(best);gain=feature_map[group]&needed
            if not gain:raise RuntimeError(f'Cannot cover subskills for {split}: {needed}')
            assigned[group]=split;needed-=gain;counts.update(task_counts[group])
        for task,quota in QUOTAS.items():
            while counts[task]<quota//10:
                best=max(range(len(available)),key=lambda i:task_counts[available[i]][task])
                group=available.pop(best)
                if not task_counts[group][task]:raise RuntimeError(f'Cannot fill {split}/{task}')
                assigned[group]=split;counts.update(task_counts[group])
    for group in available:assigned[group]=split_of(group)
    return assigned


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=ROOT/'03_기타'/'생성버전'/'new_release')
    args=parser.parse_args()
    pools=defaultdict(lambda:defaultdict(list)); seen=set();items=[]
    for source in [interval_pool(),chord_pool(),scale_pool(),rhythm_pool(),tempo_pool(),harmony_pool(),concept_pool()]:
        for item in source:
            if item['semantic_hash'] in seen or len(item['distractors'])<3:
                continue
            seen.add(item['semantic_hash'])
            items.append(item)
    assignment=grouped_split_assignment(items)
    for item in items:
        pools[item['task']][assignment[item['family_id']]].append(item)
    out=args.output
    if out.exists() and any(p.is_file() for p in out.rglob('*')):
        raise SystemExit(f'Output must be empty; use a fresh version folder: {out}')
    for sub in ['training','evaluation','audit']:
        (out/sub).mkdir(parents=True,exist_ok=True)
    all_rows=[]
    stats={}
    for split,factor in [('train',1),('dev',0.1),('test',0.1)]:
        rng=random.Random(SEED+['train','dev','test'].index(split))
        selected=[]
        for task,quota in QUOTAS.items():
            n=int(quota*factor); pool=pools[task][split]
            if len(pool)<n:
                raise RuntimeError(f'{task}/{split} needs {n} but has {len(pool)} unique candidates')
            selected.extend(stratified_select(pool,n,rng))
        rng.shuffle(selected)
        rows=[]; mc_index=0
        for index,item in enumerate(selected):
            format_='mcq' if split!='train' or index%2==0 else 'short_answer'
            row={k:v for k,v in item.items() if k!='distractors'}
            row.update(id=f'mt2_{split}_{index+1:05}',split=split,format=format_,modality='text_only',language='en',
                       provenance='new rule-based synthesis; no public benchmark input to generation')
            if format_=='mcq':
                correct=mc_index%4; mc_index+=1
                opts=rng.sample(item['distractors'],3)
                opts.insert(correct,item['answer_text'])
                row.update(options=dict(zip('ABCD',opts)),answer_letter='ABCD'[correct])
            row['messages']=sft_messages(row)
            rows.append(row)
        if split=='train':
            write_jsonl(out/'training'/'train.jsonl',[{'messages':r['messages']} for r in rows])
        else:
            public=[]; answers=[]
            for r in rows:
                public.append({k:r[k] for k in ['id','task','category','format','question','options']} |
                              {'messages':r['messages'][:-1]})
                answers.append({k:r[k] for k in ['id','task','category','answer_text','answer_letter','family_id']})
            write_jsonl(out/'evaluation'/f'{split}_questions.jsonl',public)
            write_jsonl(out/'evaluation'/f'{split}_answer_key.jsonl',answers)
            if split=='dev':
                write_jsonl(out/'training'/'dev_loss.jsonl',[{'messages':r['messages']} for r in rows])
        write_jsonl(out/'audit'/f'{split}_records.jsonl',rows)
        all_rows.extend(rows)
        stats[split]={'count':len(rows),'tasks':dict(Counter(r['task'] for r in rows)),
                      'categories':dict(Counter(r['category'] for r in rows)),
                      'families':len(set(r['family_id'] for r in rows)),
                      'formats':dict(Counter(r['format'] for r in rows)),
                      'answer_letters':dict(Counter(r.get('answer_letter') for r in rows if r['format']=='mcq'))}
    manifest={'version':'music_theory_text_v2.2','created':'2026-10-27','seed':SEED,'stats':stats,
              'protocol':PROTOCOL,'baseline_contract':assert_baseline(),
              'source_inputs':['original theory rules','original conceptual cards'],
              'rule_version':'theory.py v2.0', 'human_expert_review':'pending',
              'generalization':'Held-out spelled pitch scenarios; shared theory rules, templates, and transposed structural patterns are intentional. This is interpolation, not an OOD or unseen-rule benchmark.',
              'exclusion_scope':'public evaluation text lexical screening in a separate step; no guarantee about private model data',
              'known_limits':['Western tonal theory, twelve-tone equal temperament',
                              'synthetic text; no images, audio, ABC notation, music history, or world-music coverage',
                              'music21 cross-checks do not replace music-expert review',
                              'no measured training gain yet; 300 test MCQs, chance accuracy 25%']}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
    samples=[]
    for task in QUOTAS:
        samples.extend([r for r in all_rows if r['split']=='train' and r['task']==task][:2])
    write_jsonl(out/'audit'/'training_review_sample.jsonl',samples)
    print(json.dumps(stats,indent=2))


def write_jsonl(path,rows):
    path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')


if __name__=='__main__':
    main()
