"""Resolve the organized project and older generated release directories."""
from pathlib import Path

PROJECT_ROOT=Path(__file__).resolve().parents[2]


class Layout:
    def __init__(self,root=None):
        self.root=Path(root or PROJECT_ROOT).resolve()
        self.raw=(self.root/'training').is_dir() and (self.root/'audit').is_dir()
        if self.raw:
            self.training=self.root/'training'
            self.evaluation=self.root/'evaluation'
            self.audit=self.root/'audit'
            self.manifest=self.root/'manifest.json'
            self.human_review=self.root/'human_review.json'
            self.training_review=self.root/'training_review.html'
            self.test_review=self.evaluation/'private_test_review.html'
        else:
            self.training=self.root/'01_학습용'/'데이터'
            self.evaluation=self.root/'02_평가용'/'데이터'
            self.audit=self.root/'03_기타'/'검증기록'
            self.manifest=self.audit/'manifest.json'
            self.human_review=self.root/'03_기타'/'검수자료'/'human_review.json'
            self.training_review=self.root/'03_기타'/'검수자료'/'training_review.html'
            self.test_review=self.root/'02_평가용'/'검수자료'/'private_test_review.html'

    def records(self,split):
        if split=='test' and not self.raw:
            return self.root/'02_평가용'/'검수자료'/'test_records.jsonl'
        return self.audit/f'{split}_records.jsonl'

    def data_files(self):
        return [
            *sorted(self.training.glob('*.jsonl')),
            *sorted(self.evaluation.glob('*.jsonl')),
            *[self.records(split) for split in ['train','dev','test']],
            self.audit/'training_review_sample.jsonl',
        ]


CACHE_ROOT=PROJECT_ROOT/'03_기타'/'캐시'
