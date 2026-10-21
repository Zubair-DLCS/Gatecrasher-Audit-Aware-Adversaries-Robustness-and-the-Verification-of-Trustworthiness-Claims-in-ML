"""
Experiment 1. How much does an audit's verdict depend on choices the claim never pins down?

A claim is a threshold ("the gap is at most 0.1"); an audit returns a number and compares it.
Between the two sits a set of choices the auditor must make and the claim says nothing about:
which individuals or images to query, the random seed, how hard to attack, which canaries to
insert. Hold the model fixed, redraw only those choices, and the audit becomes a random
variable.

Two quantities follow, and neither needs a threshold to be chosen in advance, so neither can
be tuned to make an audit look good or bad:

  indeterminate band   the 5th-95th percentile range of the statistic. Any claim threshold
                       inside it gets a verdict that the audit itself would overturn at least
                       5% of the time, on the same model, with nothing changed but the draw.
  reproducibility(tau) max(P(pass), P(fail)) at a given threshold: 1.0 is a verdict the audit
                       always repeats, 0.5 is a coin flip.

An audit whose own randomness decides the verdict cannot support the claim no matter who is
being audited, so the band is a lower bound on how wrong an audit can be, not an upper one.
"""
import json
import os

import numpy as np
from joblib import Parallel, delayed

from audits import (fairness_audit, make_canaries, privacy_audit, robustness_audit, verdict)
from config import RES, SEED
from data import load_compas, load_fashion
from models import Net, train

R = 300                      # audit replicates per setting
DP = {"clip": 1.0, "noise_multiplier": 1.0}
FASHION_CLASSES = (0, 3)     # t-shirt/top vs dress


def band(samples, lo=5, hi=95):
    q = np.percentile(samples, [lo, hi])
    return float(q[0]), float(q[1]), float(q[1] - q[0])


def reproducibility(samples, tau, audit):
    p = float(np.mean([verdict(audit, s, tau) for s in samples]))
    return max(p, 1.0 - p), p


def summarise(audit, samples, setting):
    lo, hi, width = band(samples)
    return dict(audit=audit, **setting, replicates=len(samples),
                mean=float(np.mean(samples)), sd=float(np.std(samples)),
                band_lo=lo, band_hi=hi, band_width=width)


# ------------------------------------------------------------------ fairness
def fairness_stability(models, data, budgets=(100, 200, 500, 1000, 2000), reps=R):
    rows, draws = [], {}
    for name, net in models.items():
        for metric in ("demographic_parity", "equalised_odds"):
            for m in budgets:
                rng = np.random.default_rng(SEED)
                s = [fairness_audit(net, data["test_x"], data["test_y"], data["test_g"], m,
                                    rng, metric) for _ in range(reps)]
                rows.append(summarise("fairness", s, dict(model=name, metric=metric, queries=m)))
                draws[f"fairness|{name}|{metric}|{m}"] = s
                print(rows[-1], flush=True)
    return rows, draws


# ------------------------------------------------------------------ robustness
def _rob_rep(net, X, y, m, seed, eps, steps, restarts):
    return robustness_audit(net, X, y, m, np.random.default_rng(seed), eps=eps, steps=steps,
                            restarts=restarts)


def robustness_stability(models, data, budgets=(100, 200, 500, 1000), eps=0.1, reps=R):
    rows, draws = [], {}
    for name, net in models.items():
        for m in budgets:
            s = Parallel(n_jobs=2)(delayed(_rob_rep)(net, data["test_x"], data["test_y"], m,
                                                     SEED + i, eps, 40, 1) for i in range(reps))
            rows.append(summarise("robustness", s, dict(model=name, epsilon=eps, queries=m)))
            draws[f"robustness|{name}|{eps}|{m}"] = list(s)
            print(rows[-1], flush=True)
    return rows, draws


def robustness_strength(models, data, m=1000,
                        strengths=((1, 1), (5, 1), (10, 1), (20, 1), (40, 1), (40, 5), (40, 10)),
                        epsilons=(0.02, 0.05, 0.1, 0.15)):
    """The two knobs the claim rarely fixes: how far the attacker may move (epsilon) and how
    hard the auditor tries (steps x restarts)."""
    rows = []
    for name, net in models.items():
        for eps in epsilons:
            for steps, restarts in strengths:
                a = robustness_audit(net, data["test_x"], data["test_y"], m,
                                     np.random.default_rng(SEED), eps=eps, steps=steps,
                                     restarts=restarts)
                rows.append(dict(model=name, epsilon=eps, steps=steps, restarts=restarts,
                                 robust_accuracy=a))
        print(name, "strength sweep done", flush=True)
    return rows


# ------------------------------------------------------------------ privacy
def _priv_rep(fx, fy, cx, cy, k, seed, dp):
    e, _ = privacy_audit(fx, fy, cx, cy, k, np.random.default_rng(seed), dp=dp)
    return e


def privacy_stability(data, canary_counts=(20, 50, 100, 200), reps=100):
    """Each replicate is a full audit: choose canaries, train, try to pick them out."""
    rows, draws = [], {}
    cx, cy = make_canaries(data["train_x"].shape[1], 400, np.random.default_rng(SEED + 99))
    for name, dp in (("unprotected", None), ("dp-sgd", DP)):
        for k in canary_counts:
            s = Parallel(n_jobs=2)(delayed(_priv_rep)(data["train_x"], data["train_y"], cx, cy,
                                                      k, SEED + i, dp) for i in range(reps))
            rows.append(summarise("privacy", list(s), dict(model=name, canaries=k)))
            draws[f"privacy|{name}|{k}"] = list(s)
            print(rows[-1], flush=True)
    return rows, draws


# ------------------------------------------------------------------ driver
def adversarial_train(X, y, eps, hidden=32, epochs=30, lr=0.1, batch=128, steps=7, seed=0,
                      warmup=5):
    """Standard PGD adversarial training, with the attack radius warmed up over the first few
    epochs (without the warm-up this model collapses to predicting one class)."""
    net = Net(X.shape[1], hidden, seed)
    rng = np.random.default_rng(seed + 1)
    n = len(y)
    for ep in range(epochs):
        e = eps * min(1.0, (ep + 1) / max(1, warmup))
        order = rng.permutation(n)
        for s in range(0, n, batch):
            idx = order[s:s + batch]
            xb, yb = X[idx], y[idx].astype(np.float64)
            adv = np.clip(xb + rng.uniform(-e, e, xb.shape), 0.0, 1.0)
            for _ in range(steps):
                adv = adv + (e / 3.0) * np.sign(net.input_grad(adv, yb))
                adv = np.clip(np.clip(adv, xb - e, xb + e), 0.0, 1.0)
            net.step(net.grads(adv, yb), lr)
    return net


def build_models():
    compas = load_compas()
    fashion = load_fashion(classes=FASHION_CLASSES)
    fair_models = {
        "logistic": train(compas["train_x"], compas["train_y"], hidden=0, epochs=30, lr=0.3),
        "mlp": train(compas["train_x"], compas["train_y"], hidden=16, epochs=30, lr=0.3)}
    rob_models = {
        "standard": train(fashion["train_x"], fashion["train_y"], hidden=32, epochs=20, lr=0.3),
        "adversarially trained": adversarial_train(fashion["train_x"], fashion["train_y"],
                                                   eps=0.1)}
    return compas, fashion, fair_models, rob_models


def main():
    os.makedirs(RES, exist_ok=True)
    compas, fashion, fair_models, rob_models = build_models()
    meta = {"fashion_classes": fashion["class_names"],
            "clean_accuracy": {k: round(v.accuracy(fashion["test_x"], fashion["test_y"]), 4)
                               for k, v in rob_models.items()},
            "compas_accuracy": {k: round(v.accuracy(compas["test_x"], compas["test_y"]), 4)
                                for k, v in fair_models.items()},
            "compas_test_n": int(len(compas["test_y"])),
            "fashion_test_n": int(len(fashion["test_y"]))}
    print(json.dumps(meta, indent=1), flush=True)

    rows, draws = [], {}
    r, d = fairness_stability(fair_models, compas);          rows += r; draws.update(d)
    r, d = robustness_stability(rob_models, fashion);        rows += r; draws.update(d)
    r, d = privacy_stability(fashion);                       rows += r; draws.update(d)

    import pandas as pd
    pd.DataFrame(rows).to_csv(os.path.join(RES, "stability.csv"), index=False)
    pd.DataFrame(robustness_strength(rob_models, fashion)).to_csv(
        os.path.join(RES, "robustness_strength.csv"), index=False)
    json.dump(draws, open(os.path.join(RES, "audit_draws.json"), "w"))
    json.dump(meta, open(os.path.join(RES, "models.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
