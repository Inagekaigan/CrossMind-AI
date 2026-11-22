import numpy as np
import pytest
import torch
from crossmind.models import TemporalModel, mixture_nll
from crossmind.jaad import read_annotation, features
from crossmind.classification import classification_metrics
from crossmind.train import train


@pytest.mark.parametrize('case', ['nan_position', 'nan_time', 'nan_neighbor', 'wrong_neighbor', 'wrong_time', 'unknown_group'])
def test_invalid_features_rejected(case):
    positions = np.zeros((3, 2))
    times = np.arange(3, dtype=float)
    neighbors = np.zeros((3, 3))
    groups = ['position', 'velocity', 'interaction']
    if case == 'nan_position': positions[0, 0] = np.nan
    if case == 'nan_time': times[0] = np.nan
    if case == 'nan_neighbor': neighbors[0, 0] = np.nan
    if case == 'wrong_neighbor': neighbors = np.zeros((3, 1))
    if case == 'wrong_time': times = times[:2]
    if case == 'unknown_group': groups = ['typo']
    with pytest.raises(ValueError):
        features(positions, times, neighbors, groups)


def test_zero_epoch_training_rejected(tmp_path):
    with pytest.raises(ValueError, match='positive'):
        train(tmp_path/'missing', tmp_path/'output', epochs=0)


def test_xml_coordinates_and_missing_labels(tmp_path):
    p = tmp_path/'fixture.xml'
    p.write_text('''<annotations><meta><task><original_size><width>200</width><height>100</height>
    </original_size></task></meta><track label="pedestrian"><box frame="0" outside="0"
    xtl="20" ytl="10" xbr="40" ybr="60"><attribute name="id">p1</attribute>
    <attribute name="cross">not-crossing</attribute></box><box frame="1" outside="0"
    xtl="20" ytl="10" xbr="40" ybr="60"><attribute name="id">p1</attribute></box>
    <box frame="2" outside="1"/></track></annotations>''')
    a = read_annotation(p)[0]['values']
    assert len(a) == 2
    np.testing.assert_allclose(a[0], [0, .15, .6, 0, 1])
    assert np.isnan(a[1, 3])


def test_features_use_seconds_and_masks():
    p = np.array([[.1, .2], [.2, .4]])
    x = features(p, [0, .5], groups=['position', 'velocity'])
    np.testing.assert_allclose(x[1, 4:6], [.2, .4])
    assert np.all(x[:, 6:] == 0)
    assert np.all(features(p, [0, .5], groups=['position'])[:, 4:] == 0)


@pytest.mark.parametrize('task', ['intention', 'trajectory', 'joint'])
def test_heads_backward(task):
    torch.manual_seed(1)
    m = TemporalModel(future_steps=4, task=task)
    out = m(torch.randn(2, 11, 9))
    terms = []
    if 'means' in out:
        assert out['means'].shape == (2, 3, 4, 2)
        terms.append(mixture_nll(out, torch.zeros(2, 4, 2)).mean())
    if 'intention' in out:
        assert out['intention'].shape == (2,)
        terms.append(out['intention'].square().mean())
    sum(terms).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())


def test_gaussian_nll_known_value():
    out = dict(means=torch.zeros(1, 1, 1, 2), log_scales=torch.zeros(1, 1, 1, 2), logits=torch.zeros(1, 1))
    assert mixture_nll(out, torch.zeros(1, 1, 2)).item() == pytest.approx(np.log(2*np.pi))


def test_classification_single_class():
    r = classification_metrics([0, 0], [.2, .1])
    assert r['roc_auc'] is None
    assert r['brier'] == pytest.approx(.025)


def test_resume_matches_uninterrupted(tmp_path):
    import json
    d = tmp_path/'data'
    d.mkdir()
    rng = np.random.default_rng(2)
    for split in ['train', 'validation', 'test']:
        np.savez(d/f'{split}.npz', x=rng.normal(size=(8, 3, 9)).astype('float32'),
                 y=rng.normal(0, .05, size=(8, 2, 2)).astype('float32'),
                 label=np.array([0, 1]*4, dtype='float32'))
    (d/'manifest.json').write_text(json.dumps({'config': {}, 'counts': {}}))
    train(d, tmp_path/'full', epochs=2)
    train(d, tmp_path/'resumed', epochs=1)
    train(d, tmp_path/'resumed', epochs=2, resume=tmp_path/'resumed/last.pt')
    a = torch.load(tmp_path/'full/last.pt', weights_only=True)
    b = torch.load(tmp_path/'resumed/last.pt', weights_only=True)
    assert a['history'] == b['history']
    for key in a['model']:
        torch.testing.assert_close(a['model'][key], b['model'][key], rtol=0, atol=0)
