"""Small regression tests for errors that would invalidate training/evaluation."""
import unittest
from pathlib import Path
import sys
from _layout import PROJECT_ROOT,Layout
sys.path.insert(0,str(PROJECT_ROOT/'01_학습용'))
sys.path.insert(0,str(PROJECT_ROOT/'02_평가용'))
from theory import midi,pair_group,spelled_scale,spelled_chord,transpose,interval_semitones
from data_io import tokenize_response_only
from score import extract,evaluate


class FakeTokenizer:
    def apply_chat_template(self,messages,tokenize,add_generation_prompt):
        text=''.join(f'<{m["role"]}>{m["content"]}</turn>' for m in messages)
        if add_generation_prompt:text+='<assistant>'
        return list(text.encode())


class Tests(unittest.TestCase):
    def test_organized_layout(self):
        layout=Layout()
        self.assertTrue((layout.training/'train.jsonl').is_file())
        self.assertTrue((layout.evaluation/'test_questions.jsonl').is_file())
        self.assertTrue(layout.manifest.is_file())
        self.assertTrue(layout.records('test').is_file())
        self.assertFalse(layout.records('test').is_relative_to(PROJECT_ROOT/'01_학습용'))

    def test_spelling_and_direction(self):
        self.assertEqual(transpose('C4','A4'),'F#4')
        self.assertEqual(transpose('C4','d5'),'Gb4')
        self.assertEqual(transpose('C4','M3',-1),'Ab3')
        self.assertEqual(transpose('B#3','m2'),'C#4')
        self.assertEqual(spelled_chord('C4','diminished seventh'),['C4','Eb4','Gb4','Bbb4'])
        self.assertEqual(spelled_scale('A4','harmonic minor')[-1],'G#5')

    def test_group_handles_reverse_and_octave(self):
        self.assertEqual(pair_group('C4','G4'),pair_group('G5','C5'))
        self.assertEqual(pair_group('C4','G5'),pair_group('G4','C4'))
        self.assertNotEqual(pair_group('C4','F#4'),pair_group('C4','Gb4'))

    def test_no_prompt_loss(self):
        messages=[{'role':'user','content':'question'},
                  {'role':'assistant','content':'Explanation\nAnswer: A'}]
        t=FakeTokenizer();encoded=tokenize_response_only(t,messages,999)
        prompt=t.apply_chat_template(messages[:-1],True,True)
        self.assertTrue(all(x==-100 for x in encoded['labels'][:len(prompt)]))
        self.assertEqual(bytes(encoded['labels'][len(prompt):]).decode(),'Explanation\nAnswer: A</turn>')
        with self.assertRaises(ValueError):tokenize_response_only(t,messages,10)

    def test_strict_grading(self):
        self.assertEqual(extract('Answer: B\nExplanation: ...'),'B')
        self.assertIsNone(extract('Answer: b.'))
        self.assertIsNone(extract('A or B could work'))
        self.assertEqual(extract('Answer: A\nAnswer: B'),'B')
        self.assertIsNone(extract('Answer: Because of the chord'))

    def test_missing_ids_rejected(self):
        with self.assertRaises(ValueError):evaluate({}, {'x':{'id':'x'}})


if __name__=='__main__':unittest.main()
