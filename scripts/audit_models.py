"""Independently reload every pilot checkpoint and verify recorded held-out scores."""
import json
from pathlib import Path
import numpy as np
import torch
from crossmind.models import TemporalModel
from crossmind.train import read_data, score, digest, baselines


def compare(actual, expected):
    if isinstance(expected, dict):
        for key, value in expected.items():
            compare(actual[key], value)
    elif isinstance(expected, (int, float)):
        assert np.isfinite(actual) and np.isclose(actual, expected, rtol=1e-5, atol=1e-6), (actual, expected)
    else:
        assert actual == expected


def main():
    root = Path(__file__).resolve().parents[1]
    torch.set_num_threads(2)
    records = []
    for path in sorted((root/'experiments').glob('**/best.pt')):
        if path.parent.parent.name not in ('jaad_default', 'ablations'):
            continue
        metrics = json.loads(path.with_name('metrics.json').read_text())
        name = path.parent.name.rsplit('_', 1)[0]
        data_dir = root/'data/processed'/('jaad_default' if path.parent.parent.name == 'jaad_default' else 'ablation_'+name)
        saved = torch.load(path, map_location='cpu', weights_only=True)
        assert digest(path) == metrics['checkpoint_sha256']
        for split, hash_value in saved['hashes'].items():
            assert digest(data_dir/f'{split}.npz') == hash_value == metrics['hashes'][split]
        model = TemporalModel(**saved['model_config'])
        model.load_state_dict(saved['model'])
        training = read_data(data_dir/'train.npz')
        np.testing.assert_allclose(model.mean.numpy(), training['x'].mean(axis=(0, 1)))
        np.testing.assert_allclose(model.std.numpy(), training['x'].std(axis=(0, 1)).clip(.001))
        for split in ('validation', 'test'):
            compare(score(model, read_data(data_dir/f'{split}.npz'), torch.device('cpu')), metrics[split])
        records.append({'run': path.parent.name, 'checkpoint_sha256': digest(path), 'scores_match': True, 'training_only_normalization': True})
    assert len(records) == 15, 'Expected all nine main and six ablation models'
    compare(baselines(root/'data/processed/jaad_default'), json.loads((root/'experiments/jaad_default/baselines.json').read_text()))
    result = {'date': '2026-11-22', 'runs': records, 'baselines_match': True, 'tolerance': {'rtol': 1e-5, 'atol': 1e-6}, 'scope': 'Saved validation/test scores, checkpoint and input hashes, training-only normalization; not external validation'}
    (root/'reports/model_audit.json').write_text(json.dumps(result, indent=2)+'\n')
    print('Verified all 15 checkpoints, validation/test scores, normalization and baselines.')


if __name__ == '__main__':
    main()
