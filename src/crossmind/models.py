"""Small shared/independent GRU models with a genuine Gaussian mixture head."""
import math
import torch
from torch import nn


class TemporalModel(nn.Module):
    def __init__(self, future_steps=20, hidden=48, modes=3, task="joint"):
        super().__init__()
        if task not in {"joint", "intention", "trajectory"}:
            raise ValueError("Unknown task")
        self.config = dict(future_steps=future_steps, hidden=hidden, modes=modes, task=task)
        self.encoder = nn.GRU(9, hidden, batch_first=True)
        self.intention = nn.Linear(hidden, 1) if task != "trajectory" else None
        self.trajectory = nn.Linear(hidden, modes*future_steps*4+modes) if task != "intention" else None
        self.register_buffer("mean", torch.zeros(9))
        self.register_buffer("std", torch.ones(9))

    def forward(self, x):
        if x.ndim != 3 or x.shape[-1] != 9:
            raise ValueError("Expected [batch,time,9] features")
        _, state = self.encoder((x-self.mean)/self.std)
        state = state[-1]
        output = {}
        if self.intention is not None:
            output["intention"] = self.intention(state).squeeze(-1)
        if self.trajectory is not None:
            batch, k, future = len(x), self.config["modes"], self.config["future_steps"]
            raw = self.trajectory(state)
            coords = raw[:, :k*future*4].reshape(batch, k, future, 4)
            output.update(means=coords[..., :2]*0.1,
                          log_scales=coords[..., 2:].clamp(-4, 3)-3,
                          logits=raw[:, k*future*4:])
        return output


def mixture_nll(output, target):
    """Whole-trajectory mixture with diagonal conditional Gaussian components.

    Returns per-trajectory NLL (sum over future time and xy); modes are shared
    across the full path, not selected independently at each timestep.
    """
    means, log_scales = output["means"], output["log_scales"]
    if target.shape != (means.shape[0], means.shape[2], 2):
        raise ValueError("Target does not match trajectory head")
    z = (target[:, None]-means)*torch.exp(-log_scales)
    component = (-.5*z.square()-log_scales-.5*math.log(2*math.pi)).sum(dim=(-1, -2))
    return -torch.logsumexp(torch.log_softmax(output["logits"], -1)+component, dim=-1)
