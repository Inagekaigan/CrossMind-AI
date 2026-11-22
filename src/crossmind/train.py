"""Reproducible pilot training and evaluation; run as python -m crossmind.train."""
import argparse
import copy
import hashlib
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from .models import TemporalModel, mixture_nll
from .classification import classification_metrics
from .metrics import trajectory_metrics
from .trajectory import predict


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_data(path):
    with np.load(path, allow_pickle=False) as a:
        return {k: a[k].copy() for k in a.files}


def loss(output, y, label):
    terms = []
    if "intention" in output:
        terms.append(F.binary_cross_entropy_with_logits(output["intention"], label))
    if "means" in output:
        terms.append(mixture_nll(output, y).mean()/(y.shape[1]*2))
    return sum(terms)


def score(model, data, device, miss_threshold=.1):
    model.eval()
    with torch.no_grad():
        output = model(torch.from_numpy(data["x"]).to(device))
        value = float(loss(output, torch.from_numpy(data["y"]).to(device),
                           torch.from_numpy(data["label"]).to(device)).cpu())
    result = {"loss": value, "count": len(data["x"]), "units": "normalized_image",
              "miss_threshold": miss_threshold}
    if "intention" in output:
        result["intention"] = classification_metrics(data["label"], output["intention"].sigmoid().cpu().numpy())
    if "means" in output:
        means = output["means"].cpu().numpy()
        # Designated mode is the highest mixture score, not the best future match.
        order = output["logits"].argsort(dim=-1, descending=True).cpu().numpy()
        rows = [trajectory_metrics(m[o], y, miss_threshold) for m, o, y in zip(means, order, data["y"])]
        result["trajectory"] = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
        result["trajectory"]["whole_trajectory_nll"] = float(mixture_nll(output, torch.from_numpy(data["y"]).to(device)).mean().cpu())
    return result


def train(data_dir, out_dir, task="joint", seed=17, epochs=30, patience=8, resume=None):
    if epochs < 1 or patience < 1:
        raise ValueError("Epochs and patience must be positive")
    data_dir, out_dir = Path(data_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    # CPU gives deterministic, lightweight pilot runs and works without a CUDA build.
    device = torch.device("cpu")
    data = {s: read_data(data_dir/f"{s}.npz") for s in ("train", "validation", "test")}
    hashes = {s: digest(data_dir/f"{s}.npz") for s in data}
    model = TemporalModel(future_steps=data["train"]["y"].shape[1], task=task).to(device)
    model.mean.copy_(torch.from_numpy(data["train"]["x"].mean(axis=(0, 1))))
    model.std.copy_(torch.from_numpy(data["train"]["x"].std(axis=(0, 1)).clip(.001)))
    optimizer = torch.optim.AdamW(model.parameters(), lr=.003, weight_decay=.0001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=.5)
    best, stale, start, history = float("inf"), 0, 0, []
    best_state = copy.deepcopy(model.state_dict())
    if resume:
        checkpoint = torch.load(resume, map_location=device, weights_only=True)
        if checkpoint["hashes"] != hashes or checkpoint["model_config"] != model.config or checkpoint["seed"] != seed:
            raise ValueError("Resume data, task or seed mismatch")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        torch.set_rng_state(checkpoint["torch_rng"])
        best, stale, start = checkpoint["best"], checkpoint["stale"], checkpoint["epoch"]+1
        best_state, history = checkpoint["best_model"], checkpoint["history"]
    x = torch.from_numpy(data["train"]["x"]).to(device)
    y = torch.from_numpy(data["train"]["y"]).to(device)
    labels = torch.from_numpy(data["train"]["label"]).to(device)
    started = time.perf_counter()
    for epoch in range(start, epochs):
        if stale >= patience:
            break
        model.train()
        losses = []
        for ids in torch.randperm(len(x)).split(32):
            optimizer.zero_grad(set_to_none=True)
            batch_loss = loss(model(x[ids]), y[ids], labels[ids])
            if not torch.isfinite(batch_loss):
                raise RuntimeError("Nonfinite training loss")
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            losses.append((float(batch_loss.detach()), len(ids)))
        val = score(model, data["validation"], device)["loss"]
        scheduler.step(val)
        if val < best-1e-6:
            best, stale = val, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
        history.append({"epoch": epoch, "train_loss": sum(v*n for v, n in losses)/len(x), "validation_loss": val})
        checkpoint = {"model": model.state_dict(), "best_model": best_state, "model_config": model.config,
                      "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                      "torch_rng": torch.get_rng_state(), "epoch": epoch, "best": best, "stale": stale,
                      "hashes": hashes, "seed": seed, "history": history}
        torch.save(checkpoint, out_dir/"last.pt")
        (out_dir/"history.json").write_text(json.dumps(history, indent=2))
    model.load_state_dict(best_state)
    manifest = json.loads((data_dir/"manifest.json").read_text())
    torch.save({"model": best_state, "model_config": model.config, "data_config": manifest["config"],
                "seed": seed, "hashes": hashes, "status": "JAAD subset pilot; uncalibrated"}, out_dir/"best.pt")
    result = {"status": "JAAD annotated at-risk cohort pilot; not the published benchmark protocol",
              "task": task, "seed": seed, "epochs": len(history), "device": str(device),
              "seconds_this_run": time.perf_counter()-started, "torch": str(torch.__version__),
              "hashes": hashes, "checkpoint_sha256": digest(out_dir/"best.pt"),
              "source_hashes": {p.name: digest(p) for p in Path(__file__).parent.glob('*.py')},
              "training_settings": {"optimizer": "AdamW", "initial_lr": .003, "weight_decay": .0001,
                                    "batch_size": 32, "gradient_clip": 5., "patience": patience,
                                    "max_epochs": epochs, "class_weighting": "none; natural prevalence",
                                    "trajectory_loss": "whole-trajectory NLL / (2 * future_steps)",
                                    "joint_loss_weights": [1, 1], "deterministic": True},
              "data_config": manifest["config"], "counts": manifest["counts"],
              "validation": score(model, data["validation"], device),
              "test": score(model, data["test"], device)}
    (out_dir/"metrics.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    return result


def baselines(data_dir):
    data_dir = Path(data_dir)
    tr, te = read_data(data_dir/"train.npz"), read_data(data_dir/"test.npz")
    config = json.loads((data_dir/"manifest.json").read_text())["config"]
    hz = config["source_fps"]/config["sample_step"]
    result = {"status": "same-cohort JAAD pilot", "prior": classification_metrics(te["label"],
               np.full(len(te["label"]), tr["label"].mean())), "units": "normalized_image", "miss_threshold": .1}
    for method in ("stationary", "constant_velocity"):
        rows = []
        for positions, target in zip(te["positions"], te["y"]):
            times = np.arange(len(positions))/hz
            future = times[-1]+np.arange(1, len(target)+1)/hz
            pred = predict(positions, times, future, method, 1.01/hz)-positions[-1]
            rows.append(trajectory_metrics(pred, target, .1))
        result[method] = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", choices=["joint", "intention", "trajectory"], default="joint")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    result = train(args.data, args.output, args.task, args.seed, args.epochs, resume=args.resume)
    print(json.dumps({"task": result["task"], "epochs": result["epochs"], "test": result["test"]}, indent=2))
