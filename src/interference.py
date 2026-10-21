"""
Experiment 2. Claims are not independent: tightening one moves the others.

A claim card that lists privacy, fairness and robustness side by side invites the reader to
treat them as separate boxes to tick. They are not. Two sweeps measure the coupling on the
same models and the same audits used everywhere else in this repository:

  privacy -> others     train with DP-SGD at a range of noise multipliers, then measure
                        accuracy, the fairness gap (COMPAS) and robust accuracy (Fashion)
  robustness -> privacy train with PGD adversarial training at a range of radii, then run the
                        canary membership audit on the result

Nothing here is an attack: every model is trained by a standard recipe, and the point is that
a verified claim can be invalidated by a later, unrelated change to the training pipeline.
"""
import json
import os

import numpy as np
import pandas as pd

from audits import fairness_audit, make_canaries, privacy_audit, robustness_audit
from config import RES, SEED
from data import load_compas, load_fashion
from models import train
from stability import FASHION_CLASSES, adversarial_train

NOISE = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)
RADII = (0.0, 0.02, 0.05, 0.1, 0.2)
SEEDS = (0, 1, 2)


def privacy_to_others():
    compas, fashion = load_compas(), load_fashion(classes=FASHION_CLASSES)
    big = np.random.default_rng(SEED)
    rows = []
    for sigma in NOISE:
        for seed in SEEDS:
            dp = None if sigma == 0 else {"clip": 1.0, "noise_multiplier": sigma}
            c = train(compas["train_x"], compas["train_y"], hidden=0, epochs=30, lr=0.3,
                      dp=dp, seed=seed)
            f = train(fashion["train_x"], fashion["train_y"], hidden=32, epochs=20, lr=0.3,
                      dp=dp, seed=seed)
            rows.append(dict(
                noise_multiplier=sigma, seed=seed,
                compas_accuracy=c.accuracy(compas["test_x"], compas["test_y"]),
                fairness_gap=fairness_audit(c, compas["test_x"], compas["test_y"],
                                            compas["test_g"], 10 ** 9, big),
                fashion_accuracy=f.accuracy(fashion["test_x"], fashion["test_y"]),
                robust_accuracy=robustness_audit(f, fashion["test_x"], fashion["test_y"], 1000,
                                                 np.random.default_rng(SEED), eps=0.1, steps=40)))
            print(rows[-1], flush=True)
    return pd.DataFrame(rows)


def robustness_to_privacy(reps=5):
    fashion = load_fashion(classes=FASHION_CLASSES)
    cx, cy = make_canaries(fashion["train_x"].shape[1], 400, np.random.default_rng(SEED + 99))
    rows = []
    for eps in RADII:
        for i in range(reps):
            rng = np.random.default_rng(SEED + i)
            if eps == 0:
                e, net = privacy_audit(fashion["train_x"], fashion["train_y"], cx, cy, 200, rng)
            else:
                # same audit, but the model is adversarially trained on the canary-augmented set
                pool = len(cy)
                inside = rng.choice(pool, size=200, replace=False)
                mask = np.zeros(pool, dtype=bool)
                mask[inside] = True
                X = np.vstack([fashion["train_x"], cx[mask]])
                y = np.concatenate([fashion["train_y"], cy[mask]])
                net = adversarial_train(X, y, eps=eps, seed=SEED + i)
                from audits import epsilon_lower_bound
                from models import bce
                score = -bce(net.logits(cx), cy.astype(np.float64))
                e = epsilon_lower_bound(score[mask], score[~mask])
            rows.append(dict(train_epsilon=eps, rep=i, epsilon_lower_bound=e,
                             accuracy=net.accuracy(fashion["test_x"], fashion["test_y"])))
            print(rows[-1], flush=True)
    return pd.DataFrame(rows)


def main():
    os.makedirs(RES, exist_ok=True)
    a = privacy_to_others()
    a.to_csv(os.path.join(RES, "interference_privacy.csv"), index=False)
    b = robustness_to_privacy()
    b.to_csv(os.path.join(RES, "interference_robustness.csv"), index=False)
    print(a.groupby("noise_multiplier").mean(numeric_only=True).round(3).to_string())
    print(b.groupby("train_epsilon").mean(numeric_only=True).round(3).to_string())


if __name__ == "__main__":
    main()
