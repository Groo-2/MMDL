#!/usr/bin/env python3
"""Download PUBLIC evaluation text only to a non-training exclusion cache.

Never import this module from the generator. No benchmark text enters SFT exports.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import datetime
import hashlib
import json
from pathlib import Path
import time
import urllib.parse
import urllib.request
import requests
from _layout import CACHE_ROOT

ROOT = CACHE_ROOT
SPECS = [
    ('m-a-p/MusicTheoryBench', 'default', 'dev'),
    ('m-a-p/MusicTheoryBench', 'default', 'test'),
    *[('MMMU/MMMU', 'Music', s) for s in ('dev', 'validation', 'test')],
    *[('MMMU/MMMU_Pro', c, 'test') for c in ('standard (4 options)', 'standard (10 options)', 'vision')],
    ('Sylence/SSMR-Bench', 'default', 'Test_textual'),
    ('Sylence/SSMR-Bench', 'default', 'Test_visual'),
]


def request_json(endpoint, params):
    url = 'https://datasets-server.huggingface.co/' + endpoint + '?' + urllib.parse.urlencode(params)
    page_cache = ROOT/'exclusion_cache'/'pages'
    page_cache.mkdir(parents=True,exist_ok=True)
    cache_path=page_cache/(hashlib.sha256(url.encode()).hexdigest()+'.json')
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding='utf-8'))
    for attempt in range(8):
        try:
            response = requests.get(url, timeout=90)
            response.raise_for_status()
            result=response.json()
            cache_path.write_text(json.dumps(result,ensure_ascii=False),encoding='utf-8')
            time.sleep(0.5)
            return result
        except Exception:
            if attempt == 7:
                raise
            time.sleep(min(20, 3 * (attempt + 1)))


def fetch(spec):
    repo, config, split = spec
    output, offset, total = [], 0, None
    while total is None or offset < total:
        params={'dataset': repo, 'config': config, 'split': split, 'offset': offset, 'length': 100}
        endpoint='rows'
        # The viewer filter endpoint is intermittently unavailable. Fetch all
        # public text rows instead; scanning extra subjects is conservative.
        result = request_json(endpoint, params)
        total = result['num_rows_total']
        for item in result['rows']:
            row = item['row']
            # Keep only text. Images, answers, and analyses are not required for
            # lexical screening and are deliberately not cached.
            question = row.get('stem') or row.get('question') or ''
            context = row.get('abc_score') or row.get('abc_context') or ''
            options = row.get('options') or []
            output.append({'source': repo, 'config': config, 'split': split,
                           'source_id': str(row.get('id', item['row_idx'])),
                           'question': question, 'context': context,
                           'options': options})
        if not result['rows']:
            raise RuntimeError(f'Incomplete API response: {spec} at {offset}/{total}')
        offset += len(result['rows'])
    print(f'{repo} / {config} / {split}: {total}', flush=True)
    return output, {'dataset': repo, 'config': config, 'split': split, 'rows': total, 'status': 'complete'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pilot-zip', type=Path)
    args = parser.parse_args()
    cache = ROOT / 'exclusion_cache'
    cache.mkdir(exist_ok=True)
    rows, manifest = [], []
    with ThreadPoolExecutor(max_workers=1) as pool:
        for data, status in pool.map(fetch, SPECS):
            rows.extend(data)
            manifest.append(status)
    if args.pilot_zip:
        import zipfile
        with zipfile.ZipFile(args.pilot_zip) as archive:
            for line in archive.read('pilot_test.jsonl').decode('utf-8').splitlines():
                r = json.loads(line)
                rows.append({'source': 'team/pilot20', 'config': 'local', 'split': 'test',
                             'source_id': r.get('id'), 'question': r['question'], 'context': '', 'options': []})
        manifest.append({'dataset': 'team/pilot20', 'rows': 20, 'status': 'complete'})
    path = cache / 'public_eval_text.jsonl'
    path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')
    result = {'retrieved_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'rows': len(rows), 'files': manifest, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
              'use': 'exclusion screening only; never training',
              'limits': 'Text only. Does not inspect image content, prove semantic non-overlap, or cover private model data.'}
    (cache / 'sources.json').write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')


if __name__ == '__main__':
    main()
