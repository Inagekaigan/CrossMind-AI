"""Matched-cohort feature ablations; no labels, splits or eligibility change."""
import json
from pathlib import Path
import numpy as np
from crossmind.train import read_data, train

root = Path(__file__).resolve().parents[1]
source = root/'data/processed/jaad_default'
for name, groups, cutoff in [('position', ['position'], 4),
                             ('position_velocity', ['position','velocity'], 6)]:
    target = root/'data/processed'/('ablation_'+name)
    target.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((source/'manifest.json').read_text())
    manifest['config']['features'] = groups
    manifest['ablation'] = name
    (target/'manifest.json').write_text(json.dumps(manifest, indent=2))
    for split in ['train','validation','test']:
        data = read_data(source/f'{split}.npz')
        data['x'][:, :, cutoff:] = 0
        np.savez_compressed(target/f'{split}.npz', **data)
    for seed in [17,29,43]:
        result = train(target, root/'experiments/ablations'/f'{name}_{seed}', seed=seed)
        print(name, seed, json.dumps(result['test']), flush=True)

summary = {}
for name in ['position', 'position_velocity']:
    runs = [json.loads(p.read_text())['test'] for p in sorted((root/'experiments/ablations').glob(name+'_[0-9]*/metrics.json'))]
    summary[name] = {}
    for section, keys in [('intention', ['f1', 'roc_auc', 'brier']), ('trajectory', ['minADE', 'minFDE'])]:
        for key in keys:
            values = [r[section][key] for r in runs]
            summary[name][key] = {'mean': float(np.mean(values)), 'std': float(np.std(values, ddof=1))}
(root/'experiments/ablations/comparison.json').write_text(json.dumps(summary, indent=2))
