"""Run the predeclared 3-task x 3-seed pilot, then write an honest comparison."""
import json
from pathlib import Path
import numpy as np
from crossmind.train import train, baselines

root = Path(__file__).resolve().parents[1]
data = root/'data/processed/jaad_default'
output = root/'experiments/jaad_default'
output.mkdir(parents=True, exist_ok=True)
(output/'baselines.json').write_text(json.dumps(baselines(data), indent=2))
rows = []
for task in ('intention', 'trajectory', 'joint'):
    for seed in (17, 29, 43):
        result = train(data, output/f'{task}_{seed}', task=task, seed=seed, epochs=30)
        rows.append(result)
        print(task, seed, 'epochs', result['epochs'], json.dumps(result['test']), flush=True)
summary = {}
for task in ('intention', 'trajectory', 'joint'):
    selected = [r['test'] for r in rows if r['task'] == task]
    summary[task] = {}
    for section, names in [('intention', ['f1', 'roc_auc', 'average_precision', 'brier', 'ece_10_bins']),
                           ('trajectory', ['ADE_mode0', 'FDE_mode0', 'minADE', 'minFDE', 'miss', 'whole_trajectory_nll'])]:
        if section not in selected[0]:
            continue
        for name in names:
            values = [r[section][name] for r in selected]
            summary[task][name] = {'mean': float(np.mean(values)), 'std': float(np.std(values, ddof=1)), 'runs': values} if all(v is not None for v in values) else None
(output/'comparison.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
print('Saved', output/'comparison.json', flush=True)
