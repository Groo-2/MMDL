"""학습 체크포인트가 저장 중에 끊겼을 때 --resume이 깨진 체크포인트를 고르지 않게 한다.

transformers(5.19.0 확인)의 Trainer는 체크포인트를 `checkpoint-N` 폴더에 **바로** 쓴다(임시 폴더에 쓰고 이름을 바꾸는 방식이 아니다).
쓰는 순서는 모델 가중치 → 옵티마이저·스케줄러 → RNG → **trainer_state.json(마지막)** 이다.
저장 도중에 프로세스가 죽으면 가장 높은 번호의 `checkpoint-N`이 반쪽인 채로 남고, `get_last_checkpoint`가 그것을 골라서 깨진 파일로 이어받으려다 실패한다.
trainer_state.json이 마지막에 쓰이므로 이 파일이 없는 체크포인트는 불완전하다고 본다.

불완전한 체크포인트는 지우지 않고 `incomplete-checkpoint-N`으로 이름만 바꾼다
(`get_last_checkpoint`의 규칙이 `^checkpoint-\\d+$`라서 이렇게 하면 무시되고, 원하면 사람이 열어 볼 수 있다).
"""

import re
import shutil
from pathlib import Path

STATE_FILE = "trainer_state.json"


def quarantine_incomplete_checkpoints(output_dir):
    """trainer_state.json이 없는 checkpoint-N 폴더를 incomplete-checkpoint-N으로 바꾼다. 바꾼 폴더 이름의 리스트를 돌려준다."""
    moved = []
    out = Path(output_dir)
    if not out.is_dir():
        return moved
    for c in sorted(out.iterdir()):
        if not (c.is_dir() and re.fullmatch(r"checkpoint-\d+", c.name)):
            continue
        if (c / STATE_FILE).exists():
            continue
        dst = c.with_name("incomplete-" + c.name)
        if dst.exists():
            shutil.rmtree(dst)           # 같은 번호의 이전 불완전본은 의미가 없다
        c.rename(dst)
        moved.append(c.name)
    return moved
