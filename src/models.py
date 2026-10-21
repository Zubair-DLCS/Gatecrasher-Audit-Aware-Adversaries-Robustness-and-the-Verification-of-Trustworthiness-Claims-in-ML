"""
Small binary classifiers in NumPy: a linear model and a one-hidden-layer ReLU network, with
plain SGD and with DP-SGD (per-example gradient clipping plus Gaussian noise).

Everything is written out rather than taken from a framework for one reason: the audits in
this repository need per-example gradients, input gradients and an exact record of what noise
was added, and a reader checking an audit should be able to see all three in one short file.

Per-example clipping is exact and cheap here. For a layer whose per-example gradient is an
outer product, the Frobenius norm factorises (||a b^T||_F = ||a|| ||b||), so the norms can be
read off without ever forming the per-example gradients, and the clipped sum is one matrix
product against the rescaled backward signal.
"""
import numpy as np


def sigmoid(z):
    out = np.empty_like(z, dtype=np.float64)
    pos, neg = z >= 0, z < 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    e = np.exp(z[neg])
    out[neg] = e / (1.0 + e)
    return out


def bce(z, y):
    """Numerically stable per-example logistic loss."""
    return np.maximum(z, 0) - z * y + np.log1p(np.exp(-np.abs(z)))


class Net:
    """hidden=0 gives logistic regression; hidden>0 a one-hidden-layer ReLU network."""

    def __init__(self, d, hidden=0, seed=0):
        rng = np.random.default_rng(seed)
        self.d, self.hidden = d, hidden
        if hidden:
            self.W1 = rng.normal(0, np.sqrt(2.0 / d), (d, hidden))
            self.b1 = np.zeros(hidden)
            self.W2 = rng.normal(0, np.sqrt(2.0 / hidden), hidden)
        else:
            self.W1, self.b1 = None, None
            self.W2 = np.zeros(d)
        self.b2 = 0.0

    # ---------------------------------------------------------------- forward / backward
    def _fwd(self, X):
        if self.hidden:
            pre = X @ self.W1 + self.b1
            h = np.maximum(pre, 0.0)
        else:
            pre, h = None, X
        return pre, h, h @ self.W2 + self.b2

    def logits(self, X):
        return self._fwd(X)[2]

    def proba(self, X):
        return sigmoid(self.logits(X))

    def predict(self, X, threshold=0.5):
        return (self.proba(X) >= threshold).astype(np.int64)

    def accuracy(self, X, y):
        return float((self.predict(X) == y).mean())

    def _signals(self, X, y):
        """dz and dpre: the backward signal at the output and at the hidden pre-activation."""
        pre, h, z = self._fwd(X)
        dz = sigmoid(z) - y
        dpre = (np.outer(dz, self.W2) * (pre > 0)) if self.hidden else None
        return h, dz, dpre

    def grads(self, X, y):
        h, dz, dpre = self._signals(X, y)
        n = len(y)
        g = {"W2": h.T @ dz / n, "b2": float(dz.mean())}
        if self.hidden:
            g["W1"] = X.T @ dpre / n
            g["b1"] = dpre.mean(0)
        return g

    def per_example_norms(self, X, y):
        h, dz, dpre = self._signals(X, y)
        sq = (np.sum(h * h, 1) + 1.0) * dz ** 2
        if self.hidden:
            sq = sq + (np.sum(X * X, 1) + 1.0) * np.sum(dpre * dpre, 1)
        return np.sqrt(sq)

    def dp_grads(self, X, y, clip, noise_multiplier, rng):
        """Clipped-and-noised mean gradient: the DP-SGD update of Abadi et al."""
        h, dz, dpre = self._signals(X, y)
        norms = self.per_example_norms(X, y)
        scale = np.minimum(1.0, clip / np.maximum(norms, 1e-12))
        n = len(y)
        sd = clip * noise_multiplier
        dzc = dz * scale
        g = {"W2": (h.T @ dzc + rng.normal(0, sd, self.W2.shape)) / n,
             "b2": float((dzc.sum() + rng.normal(0, sd)) / n)}
        if self.hidden:
            dprec = dpre * scale[:, None]
            g["W1"] = (X.T @ dprec + rng.normal(0, sd, self.W1.shape)) / n
            g["b1"] = (dprec.sum(0) + rng.normal(0, sd, self.b1.shape)) / n
        return g

    def input_grad(self, X, y):
        """Gradient of the loss with respect to the inputs (used by the robustness audit)."""
        _, dz, dpre = self._signals(X, y)
        return dpre @ self.W1.T if self.hidden else np.outer(dz, self.W2)

    def step(self, g, lr):
        self.W2 -= lr * g["W2"]
        self.b2 -= lr * g["b2"]
        if self.hidden:
            self.W1 -= lr * g["W1"]
            self.b1 -= lr * g["b1"]

    def copy(self):
        new = Net(self.d, self.hidden)
        new.W2, new.b2 = self.W2.copy(), self.b2
        if self.hidden:
            new.W1, new.b1 = self.W1.copy(), self.b1.copy()
        return new


def train(X, y, hidden=0, epochs=20, lr=0.5, batch=128, dp=None, seed=0):
    """Mini-batch SGD, or DP-SGD when dp={'clip': C, 'noise_multiplier': sigma}.

    DP-SGD here uses shuffled fixed-size batches rather than Poisson sampling, so it is the
    usual implementation rather than the one the tightest accounting assumes. That matters for
    turning (clip, sigma, epochs) into an epsilon; it does not matter for this repository,
    which never claims an epsilon for a trained model and compares protected training against
    training with no protection at all.
    """
    net = Net(X.shape[1], hidden, seed)
    rng = np.random.default_rng(seed + 1)
    n = len(y)
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, batch):
            idx = order[s:s + batch]
            xb, yb = X[idx], y[idx].astype(np.float64)
            g = net.grads(xb, yb) if dp is None else net.dp_grads(
                xb, yb, dp["clip"], dp["noise_multiplier"], rng)
            net.step(g, lr)
    return net


if __name__ == "__main__":
    import time
    from data import load_compas, load_fashion

    c = load_compas()
    for hid in (0, 16):
        m = train(c["train_x"], c["train_y"], hidden=hid, epochs=30, lr=0.3)
        print(f"compas hidden={hid}: train {m.accuracy(c['train_x'], c['train_y']):.3f} "
              f"test {m.accuracy(c['test_x'], c['test_y']):.3f}")
    f = load_fashion()
    for dp in (None, {"clip": 1.0, "noise_multiplier": 1.0}):
        t = time.time()
        m = train(f["train_x"], f["train_y"], hidden=32, epochs=20, lr=0.3, dp=dp)
        print(f"fashion dp={dp is not None}: train {m.accuracy(f['train_x'], f['train_y']):.3f} "
              f"test {m.accuracy(f['test_x'], f['test_y']):.3f} ({time.time() - t:.1f}s)")
