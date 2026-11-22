"""Checkpoint-backed prediction. Missing models raise errors, never dummy outputs."""
from pathlib import Path
import numpy as np
import torch
from .models import TemporalModel
from .jaad import features


class Predictor:
    def __init__(self, checkpoint):
        if not Path(checkpoint).is_file():
            raise FileNotFoundError('Trained checkpoint unavailable')
        saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
        self.model = TemporalModel(**saved['model_config'])
        self.model.load_state_dict(saved['model'])
        self.model.eval()
        self.config = saved['data_config']
        self.status = saved['status']

    @torch.inference_mode()
    def __call__(self, positions, times, neighbors=None):
        hz = self.config['source_fps']/self.config['sample_step']
        expected = round(self.config['observation_seconds']*hz)+1
        positions, times = np.asarray(positions), np.asarray(times)
        if positions.shape != (expected, 2) or times.shape != (expected,):
            raise ValueError('Observation length incompatible with checkpoint')
        if not np.isfinite(positions).all() or not np.allclose(np.diff(times), 1/hz, atol=1e-5):
            raise ValueError('Complete regularly sampled history required')
        x = features(positions, times, neighbors, self.config['features'])
        output = self.model(torch.from_numpy(x)[None])
        result = {'status': self.status, 'units': 'normalized_image',
                  'initiation_probability': None, 'paths': [], 'mode_scores': [],
                  'current_crossing': None,
                  'probability_condition': 'conditional on not currently crossing; uncalibrated',
                  'prediction_seconds': self.config['prediction_seconds']}
        if 'intention' in output:
            result['initiation_probability'] = float(output['intention'].sigmoid()[0])
        if 'means' in output:
            result['paths'] = (output['means'][0].numpy()+positions[-1]).tolist()
            result['mode_scores'] = output['logits'][0].softmax(-1).tolist()
            result['conditional_std'] = output['log_scales'][0].exp().numpy().tolist()
        return result
