"""
Two public datasets, downloaded into data/raw/ and not redistributed here.

  fashion   Fashion-MNIST (Xiao, Rasul & Vollgraf, 2017; MIT licence), 70,000 28x28 grayscale
            images in 10 classes. Used for the robustness and privacy audits.
  compas    ProPublica's COMPAS recidivism file (Angwin, Larson, Mattu & Kirchner, 2016),
            the standard benchmark for group-fairness auditing, with ProPublica's own
            filtering applied. Used for the fairness audit.

Nothing here depends on the audits: the datasets are loaded once and reused, so every audit
sees exactly the same model and the same data pool.
"""
import gzip
import os
import subprocess

import numpy as np
import pandas as pd

from config import RAW, SEED

FASHION = "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion"
FASHION_FILES = {"train_x": "train-images-idx3-ubyte.gz", "train_y": "train-labels-idx1-ubyte.gz",
                 "test_x": "t10k-images-idx3-ubyte.gz", "test_y": "t10k-labels-idx1-ubyte.gz"}
COMPAS = ("https://raw.githubusercontent.com/propublica/compas-analysis/master/"
          "compas-scores-two-years.csv")
CLASSES = ["t-shirt/top", "trouser", "pullover", "dress", "coat", "sandal", "shirt", "sneaker",
           "bag", "ankle boot"]


def _fetch(url, name):
    os.makedirs(RAW, exist_ok=True)
    path = os.path.join(RAW, name)
    if not os.path.exists(path):
        subprocess.run(["curl", "-sSL", "--retry", "3", "-o", path, url], check=True)
    return path


def _idx(path):
    with gzip.open(path, "rb") as f:
        magic = int.from_bytes(f.read(4), "big")
        dims = [int.from_bytes(f.read(4), "big") for _ in range(magic & 0xFF)]
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(*dims)


def load_fashion(n_train=6000, n_test=2000, classes=(0, 6), seed=SEED):
    """A balanced two-class subset, pixels scaled to [0, 1].

    Classes 0 and 6 are t-shirt/top and shirt: the pair Fashion-MNIST's authors call hardest,
    so the model is accurate but not saturated, which is where audit thresholds actually bite.
    """
    parts = {k: _idx(_fetch(f"{FASHION}/{v}", f"fashion_{v}")) for k, v in FASHION_FILES.items()}
    rng = np.random.default_rng(seed)
    out = {}
    for split, n in (("train", n_train), ("test", n_test)):
        x, y = parts[f"{split}_x"], parts[f"{split}_y"]
        keep = np.isin(y, classes)
        x, y = x[keep], y[keep]
        idx = rng.permutation(len(y))[:n]
        out[f"{split}_x"] = (x[idx].reshape(len(idx), -1) / 255.0).astype(np.float64)
        out[f"{split}_y"] = (y[idx] == classes[1]).astype(np.int64)
    out["class_names"] = [CLASSES[c] for c in classes]
    return out


def load_compas(seed=SEED):
    """ProPublica's filtering, then a model-ready design matrix.

    Returns features, the two-year recidivism label, and the protected group (the two largest
    race categories, as in ProPublica's own analysis). Race is deliberately NOT a feature:
    the question is whether a model that never sees race still scores groups differently.
    """
    d = pd.read_csv(_fetch(COMPAS, "compas-scores-two-years.csv"))
    d = d[(d.days_b_screening_arrest <= 30) & (d.days_b_screening_arrest >= -30) &
          (d.is_recid != -1) & (d.c_charge_degree != "O") & (d.score_text != "N/A")]
    d = d[d.race.isin(["African-American", "Caucasian"])].reset_index(drop=True)
    num = ["age", "priors_count", "juv_fel_count", "juv_misd_count", "juv_other_count"]
    x = d[num].to_numpy(float)
    x = np.column_stack([x, (d.sex == "Male").to_numpy(float),
                         (d.c_charge_degree == "F").to_numpy(float)])
    x = (x - x.mean(0)) / np.where(x.std(0) == 0, 1, x.std(0))
    y = d.two_year_recid.to_numpy(np.int64)
    g = (d.race == "African-American").to_numpy(np.int64)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    cut = int(0.6 * len(idx))
    return dict(train_x=x[idx[:cut]], train_y=y[idx[:cut]], train_g=g[idx[:cut]],
                test_x=x[idx[cut:]], test_y=y[idx[cut:]], test_g=g[idx[cut:]],
                feature_names=num + ["sex_male", "charge_felony"],
                group_names=["Caucasian", "African-American"])


if __name__ == "__main__":
    f = load_fashion()
    print("fashion", f["train_x"].shape, f["test_x"].shape, "classes", f["class_names"],
          "positive rate", round(float(f["train_y"].mean()), 3))
    c = load_compas()
    print("compas", c["train_x"].shape, c["test_x"].shape,
          "recidivism rate", round(float(c["train_y"].mean()), 3),
          "group share", round(float(c["train_g"].mean()), 3))
