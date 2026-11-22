"""Download pinned public annotations (not the multi-GB video archive)."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import urllib.request

REVISION = 'd2feeac998366e47ecc54e5533c8a3fa61141dda'
BASE = f'https://raw.githubusercontent.com/ykotseruba/JAAD/{REVISION}/'
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT/'data/raw/jaad'


def fetch(path):
    target = RAW/path
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        with urllib.request.urlopen(BASE+path, timeout=90) as response:
            content = response.read()
        temporary = target.with_suffix(target.suffix+'.part')
        temporary.write_bytes(content)
        temporary.replace(target)
    return path, hashlib.sha256(target.read_bytes()).hexdigest()


if __name__ == '__main__':
    for path in ['LICENSE', 'README.md']+[f'split_ids/default/{s}.txt' for s in ['train','val','test']]:
        fetch(path)
    splits = {key: (RAW/f'split_ids/default/{name}.txt').read_text().split()
              for key, name in [('train','train'),('validation','val'),('test','test')]}
    paths = [f'annotations/{video}.xml' for ids in splits.values() for video in ids]
    hashes = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for i, (path, digest) in enumerate(pool.map(fetch, paths), 1):
            hashes[path] = digest
            if i % 20 == 0:
                print(f'{i}/{len(paths)} annotations ready', flush=True)
    (ROOT/'data/splits/jaad_default.json').write_text(json.dumps(splits, indent=2))
    (RAW/'acquisition.json').write_text(json.dumps({'repository': BASE, 'revision': REVISION,
        'annotation_license': 'MIT; see LICENSE', 'sha256': hashes}, indent=2))
    print('Completed', len(paths), 'annotations', flush=True)
