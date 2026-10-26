"""Baseline-shaped conversations and assistant-only loss masking."""
import json
from pathlib import Path
import re


def validate_messages(messages):
    if not isinstance(messages, list) or [m.get('role') for m in messages] != ['user', 'assistant']:
        raise ValueError('Expected one user/assistant turn, matching the baseline without a custom system message.')
    if not all(isinstance(m.get('content'), str) and m['content'].strip() for m in messages):
        raise ValueError('Conversation contents must be nonempty strings.')
    target = messages[-1]['content'].strip()
    lines = target.splitlines()
    answer_lines = [i for i, line in enumerate(lines) if re.match(r'^\s*Answer\s*:', line)]
    if answer_lines != [len(lines) - 1] or not re.fullmatch(r'Answer: \S.*', lines[-1]):
        raise ValueError('The training target must end with exactly one nonempty Answer: line.')
    if not '\n'.join(lines[:-1]).strip():
        raise ValueError('The explanation must precede the final Answer: line.')


def load_sft(path):
    rows = []
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if set(row) != {'messages'}:
            raise ValueError('Training accepts only messages exports, never audit records or evaluation keys.')
        validate_messages(row['messages'])
        rows.append(row)
    if not rows:
        raise ValueError('Training input is empty.')
    return rows


def tokenize_response_only(tokenizer, messages, max_length):
    validate_messages(messages)
    prompt = tokenizer.apply_chat_template(messages[:-1], tokenize=True, add_generation_prompt=True)
    full = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
    if full[:len(prompt)] != prompt:
        raise ValueError('Chat-template prefix mismatch; refusing to mask the wrong tokens.')
    if len(full) > max_length:
        raise ValueError(f'Example has {len(full)} tokens > max_length={max_length}; no silent truncation.')
    labels = [-100] * len(prompt) + full[len(prompt):]
    if not any(x != -100 for x in labels):
        raise ValueError('No assistant tokens to train.')
    return {'input_ids': full, 'attention_mask': [1] * len(full), 'labels': labels}
