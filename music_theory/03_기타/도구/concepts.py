"""Original concise teaching cards; definitions are standard theory facts.

Not copied from a benchmark or textbook. Reviewed automatically only; the release
includes a human review sheet for music-qualified reviewers.
"""
from theory import CHORDS, INTERVAL_CODES, SCALES, interval_label, interval_semitones

# term, independently written definition, intentionally false contrast
BASE = [
 ('enharmonic equivalence', 'Different pitch spellings can have the same sounding pitch in twelve-tone equal temperament.', 'Every pair of differently spelled notes has a different sounding pitch.'),
 ('diatonic interval number', 'Count the written letter names inclusively from the first note to the second.', 'Count only the number of piano keys between the notes.'),
 ('interval quality', 'Perfect, major, minor, augmented, or diminished specifies an interval relative to its diatonic number.', 'An interval quality is determined by loudness.'),
 ('interval inversion', 'Moving one member of a simple interval by an octave makes the interval numbers add to nine.', 'Inverting a simple interval makes its numbers add to eight.'),
 ('compound interval', 'An interval larger than an octave is a compound interval.', 'An interval becomes compound whenever its notes are played together.'),
 ('melodic interval', 'Two pitches heard successively form a melodic interval.', 'Two pitches must sound simultaneously to form a melodic interval.'),
 ('harmonic interval', 'Two pitches sounding together form a harmonic interval.', 'Two pitches sounding together have no interval relationship.'),
 ('pitch class', 'Pitches separated by octaves belong to the same pitch class.', 'Each octave register has an unrelated pitch class system.'),
 ('scientific pitch notation', 'A letter name, an accidental if needed, and an octave number specify a pitch; middle C is C4 here.', 'Scientific pitch notation uses only a dynamic marking to specify pitch.'),
 ('sharp', 'A sharp raises a natural letter pitch by one semitone.', 'A sharp always raises a pitch by one whole tone.'),
 ('flat', 'A flat lowers a natural letter pitch by one semitone.', 'A flat always lowers a pitch by one whole tone.'),
 ('double sharp', 'A double sharp raises a natural letter pitch by two semitones.', 'A double sharp raises a natural letter pitch by four semitones.'),
 ('double flat', 'A double flat lowers a natural letter pitch by two semitones.', 'A double flat lowers a natural letter pitch by four semitones.'),
 ('natural sign', 'A natural sign cancels a sharp or flat for the affected written pitch according to accidental rules.', 'A natural sign moves every pitch upward by one semitone.'),
 ('tonic', 'The tonic is scale degree one and serves as the tonal center.', 'The tonic is always scale degree five.'),
 ('supertonic', 'The supertonic is scale degree two.', 'The supertonic is scale degree six.'),
 ('mediant', 'The mediant is scale degree three.', 'The mediant is scale degree seven.'),
 ('subdominant', 'The subdominant is scale degree four.', 'The subdominant is scale degree two.'),
 ('dominant', 'The dominant is scale degree five.', 'The dominant is scale degree three.'),
 ('submediant', 'The submediant is scale degree six.', 'The submediant is scale degree four.'),
 ('leading tone', 'A leading tone lies a semitone below the tonic.', 'A leading tone must lie a whole tone below the tonic.'),
 ('subtonic', 'The subtonic is a seventh scale degree a whole tone below the tonic.', 'The subtonic is a seventh scale degree a semitone below the tonic.'),
 ('relative keys', 'Relative major and minor keys share a key signature and have different tonics.', 'Relative major and minor keys must share the same tonic.'),
 ('parallel keys', 'Parallel major and minor keys share a tonic and use different scale collections.', 'Parallel keys must share a key signature and have different tonics.'),
 ('key signature', 'A key signature specifies the default altered pitch letters for a passage.', 'A key signature uniquely distinguishes a major key from its relative minor without context.'),
 ('natural minor scale', 'Natural minor uses scale degrees 1, 2, flat 3, 4, 5, flat 6, flat 7 relative to parallel major.', 'Natural minor raises scale degrees six and seven relative to parallel major.'),
 ('harmonic minor scale', 'Harmonic minor raises the seventh degree of natural minor.', 'Harmonic minor lowers the tonic of natural minor.'),
 ('classical melodic minor', 'In common classical practice, melodic minor raises degrees six and seven ascending and commonly uses natural minor descending.', 'Classical melodic minor always raises only degree two.'),
 ('triad', 'A tertian triad contains a root, a third, and a fifth.', 'A tertian triad consists of a root, a second, and a fourth.'),
 ('chord root', 'The root is the reference pitch from which a chord is built, regardless of its bass position.', 'The lowest sounding note is always the root of every chord.'),
 ('chord inversion', 'A chord inversion is identified by which chord member is in the bass.', 'A chord inversion is determined by which note is loudest.'),
 ('root position', 'A chord is in root position when its root is in the bass.', 'A chord is in root position whenever its fifth is in the bass.'),
 ('first inversion', 'A chord is in first inversion when its third is in the bass.', 'A chord is in first inversion when its root is in the bass.'),
 ('second inversion', 'A chord is in second inversion when its fifth is in the bass.', 'A chord is in second inversion when its third is in the bass.'),
 ('third inversion', 'A seventh chord is in third inversion when its seventh is in the bass.', 'A triad has a third inversion with a fourth chord member.'),
 ('Roman numeral analysis', 'Roman numerals relate chord roots and qualities to a specified tonal center.', 'Roman numerals identify the instrument playing a chord.'),
 ('major-key diatonic triads', 'The triads on degrees one through seven in major are major, minor, minor, major, major, minor, and diminished.', 'All seven diatonic triads of a major scale are major triads.'),
 ('authentic cadence', 'An authentic cadence has dominant-to-tonic harmonic motion at a phrase ending.', 'Every tonic-to-dominant motion is an authentic cadence.'),
 ('plagal cadence', 'A plagal cadence has subdominant-to-tonic harmonic motion at a phrase ending.', 'A plagal cadence is a dominant-to-submediant motion.'),
 ('half cadence', 'A half cadence ends a phrase on the dominant.', 'A half cadence must end on the tonic.'),
 ('deceptive cadence', 'A deceptive cadence substitutes an expected tonic arrival, commonly using V to vi in major.', 'A V to I ending is always a deceptive cadence.'),
 ('perfect authentic cadence', 'In a basic textbook definition, both V and I are in root position and the tonic is the highest voice of the final I chord.', 'Every V to I cadence is perfect authentic regardless of inversion or soprano.'),
 ('secondary dominant', 'A secondary dominant functions as the dominant of a chord other than the main tonic.', 'A secondary dominant is defined as any chord played for a second time.'),
 ('tonicization', 'Tonicization gives a non-tonic chord a temporary tonic-like emphasis without requiring an established new key.', 'Every temporary emphasis necessarily establishes a full modulation.'),
 ('modulation', 'Modulation establishes a new tonal center within a musical passage.', 'Changing the loudness of a passage is modulation in tonal harmony.'),
 ('passing tone', 'A passing tone connects chord tones by stepwise motion through a nonchord pitch.', 'A passing tone must be approached and left by leaps.'),
 ('neighbor tone', 'A neighbor tone moves by step away from a chord tone and returns to that same tone.', 'A neighbor tone must resolve to a different chord tone by a leap.'),
 ('suspension', 'A suspension retains a prepared tone into a new harmony and usually resolves downward by step.', 'A suspension must be an unprepared upward leap.'),
 ('retardation', 'A retardation resembles a prepared suspension but resolves upward.', 'A retardation is defined only by a reduction in tempo.'),
 ('appoggiatura', 'In a common harmonic account, an appoggiatura is an accented nonchord tone that resolves by step.', 'An appoggiatura must remain unresolved for the whole composition.'),
 ('anticipation', 'An anticipation sounds a note of the next harmony before that harmony arrives.', 'An anticipation is a note that only belongs to the previous harmony after it ends.'),
 ('pedal point', 'A pedal point sustains or repeats a pitch while the surrounding harmony changes.', 'A pedal point requires every surrounding harmony to remain unchanged.'),
 ('dotted duration', 'One dot adds half of the original undotted duration.', 'One dot doubles the original undotted duration.'),
 ('double-dotted duration', 'Two dots add one half and one quarter of the original duration.', 'Two dots add two complete original durations.'),
 ('tie', 'A tie joins successive notes of the same sounding pitch into one sustained duration.', 'A tie joins arbitrary different pitches into a single new pitch.'),
 ('slur', 'A slur indicates a connected articulation or grouping across notes and need not join equal pitches.', 'A slur always adds the duration of identical pitches exactly as a tie does.'),
 ('simple meter', 'In simple meter, each main beat normally divides into two equal parts.', 'In simple meter, each main beat must divide into three equal parts.'),
 ('compound meter', 'In compound meter, each main beat normally divides into three equal parts.', 'In compound meter, each main beat must divide into two equal parts.'),
 ('beat unit in 6/8', 'In conventional compound-duple 6/8, a dotted quarter note is one main beat.', 'In conventional compound-duple 6/8, every eighth note is one main beat.'),
 ('tuplet', 'A tuplet fits a specified number of equal notes into the time normally occupied by another specified number.', 'A tuplet changes pitch without changing duration.'),
 ('triplet', 'A standard three-in-the-time-of-two triplet scales each written duration by two thirds.', 'A standard three-in-the-time-of-two triplet scales each written duration by three halves.'),
 ('syncopation', 'Syncopation emphasizes normally weak metrical positions or displaces an expected accent.', 'Syncopation means every accent must fall on the strongest beat.'),
 ('anacrusis', 'An anacrusis is an incomplete opening measure or pickup before the first full measure.', 'An anacrusis is necessarily a complete measure at the end.'),
 ('tempo', 'Tempo describes the rate of musical beats; a metronome marking must specify a beat unit.', 'Tempo measures only the loudness of the music.'),
 ('ritardando', 'Ritardando requests a gradual slowing of tempo.', 'Ritardando requests a gradual increase in tempo.'),
 ('accelerando', 'Accelerando requests a gradual increase in tempo.', 'Accelerando requests a gradual slowing of tempo.'),
 ('crescendo', 'Crescendo requests a gradual increase in loudness.', 'Crescendo is defined as a gradual slowing of tempo.'),
 ('diminuendo', 'Diminuendo requests a gradual decrease in loudness.', 'Diminuendo is defined as a gradual increase in pitch.'),
 ('staccato', 'Staccato asks for notes to be detached in articulation.', 'Staccato means all notes must be joined without separation.'),
 ('legato', 'Legato asks for smooth, connected articulation.', 'Legato means every note must be sharply detached.'),
 ('trill', 'A trill rapidly alternates a principal note and an adjacent upper note, subject to style and notation.', 'A trill is defined as a single sustained note without alternation.'),
 ('equal temperament', 'Twelve-tone equal temperament divides the octave into twelve equal semitone ratios.', 'Twelve-tone equal temperament divides an octave into twelve equal frequency differences.'),
]


def cards():
    result = [dict(term=t, definition=d, false_statement=f, origin='original_standard_theory_card') for t, d, f in BASE]
    for code in INTERVAL_CODES:
        n = interval_semitones(code)
        result.append(dict(term=interval_label(code).lower() + ' size',
                           definition=f'A spelled {interval_label(code).lower()} spans {n} semitones.',
                           false_statement=f'A spelled {interval_label(code).lower()} spans {n+1} semitones.',
                           origin='interval_formula_card', code=code))
    for quality, codes in CHORDS.items():
        components = ', '.join(interval_label(c).lower() for c in codes)
        result.append(dict(term=quality + ' construction', definition=f'From its root, a {quality} uses {components}.',
                           false_statement=f'A {quality} is specified by root and octave alone, without any third or fifth.',
                           origin='chord_formula_card', quality=quality))
    for mode, offsets in SCALES.items():
        result.append(dict(term=mode + ' semitone pattern',
                           definition=f'The ascending {mode} scale has tonic-relative semitone offsets {", ".join(map(str,offsets))}.',
                           false_statement=f'The ascending {mode} scale has no fixed tonic-relative semitone pattern.',
                           origin='scale_formula_card', mode=mode))
    return result
