"""Training input and response-only masking, kept independent of torch."""
import json
from pathlib import Path


def load_sft(path):
    rows=[]
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        if not line.strip():continue
        row=json.loads(line)
        if set(row)!={'messages'}:
            raise ValueError('Training accepts only sanitized messages exports, never audit records or evaluation keys.')
        if [m['role'] for m in row['messages']]!=['system','user','assistant']:
            raise ValueError('Expected one system, user, assistant turn.')
        if not all(isinstance(m['content'],str) for m in row['messages']):
            raise ValueError('This release is text-only.')
        rows.append(row)
    return rows


def tokenize_response_only(tokenizer,messages,max_length):
    prompt=tokenizer.apply_chat_template(messages[:-1],tokenize=True,add_generation_prompt=True)
    full=tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=False)
    if full[:len(prompt)]!=prompt:
        raise ValueError('Chat-template prefix mismatch; refusing to mask the wrong tokens.')
    if len(full)>max_length:
        raise ValueError(f'Example has {len(full)} tokens > max_length={max_length}; no silent truncation.')
    labels=[-100]*len(prompt)+full[len(prompt):]
    if not any(x!=-100 for x in labels):raise ValueError('No assistant tokens to train.')
    return {'input_ids':full,'attention_mask':[1]*len(full),'labels':labels}
