"""
Three audits, one interface.

Each audit takes a fixed model and returns a number that a claim is judged against:

  fairness     group gap on COMPAS (demographic parity, or equalised-odds TPR/FPR gaps)
               claim: "the gap is at most tau"            -> PASS when statistic <= tau
  robustness   accuracy under an L-infinity PGD attack on Fashion-MNIST
               claim: "robust accuracy is at least tau"   -> PASS when statistic >= tau
  privacy      empirical epsilon lower bound from a canary membership test
               claim: "epsilon is at most tau"            -> PASS when statistic <= tau

Each audit also takes the choices an auditor has to make but that the claim never pins down:
how many examples to query, which ones, the random seed, and (for robustness) how hard to
attack. Those choices are the subject of the whole repository, so they are arguments here
rather than constants.
"""
import numpy as np
from scipy.stats import beta as beta_dist

from models import Net, bce, train

PASS_IF_BELOW = {"fairness": True, "robustness": False, "privacy": True}


def verdict(audit, statistic, tau):
    """PASS means the claim survives the audit."""
    return bool(statistic <= tau) if PASS_IF_BELOW[audit] else bool(statistic >= tau)


# ------------------------------------------------------------------ fairness
def fairness_audit(net, X, y, g, m, rng, metric="demographic_parity"):
    """Query m held-out individuals and measure the gap between the two groups."""
    idx = rng.choice(len(y), size=min(m, len(y)), replace=False)
    xp, yp, gp = X[idx], y[idx], g[idx]
    pred = net.predict(xp)
    if metric == "demographic_parity":
        rates = [pred[gp == k].mean() if (gp == k).any() else np.nan for k in (0, 1)]
        stat = abs(rates[1] - rates[0])
    else:                                            # equalised odds: worst of TPR, FPR gaps
        gaps = []
        for outcome in (1, 0):
            r = [pred[(gp == k) & (yp == outcome)].mean()
                 if ((gp == k) & (yp == outcome)).any() else np.nan for k in (0, 1)]
            gaps.append(abs(r[1] - r[0]))
        stat = float(np.nanmax(gaps))
    return float(stat)


# ------------------------------------------------------------------ robustness
def pgd(net, X, y, eps, steps, step_size, restarts, rng, random_start=True):
    """L-infinity projected gradient descent. Returns, per example, whether it stayed correct
    across every restart (so more restarts can only lower the reported robust accuracy)."""
    yf = y.astype(np.float64)
    still = np.ones(len(y), dtype=bool)
    for r in range(restarts):
        if random_start:
            adv = X + rng.uniform(-eps, eps, X.shape)
        else:
            adv = X.copy()
        adv = np.clip(adv, 0.0, 1.0)
        for _ in range(steps):
            adv = adv + step_size * np.sign(net.input_grad(adv, yf))
            adv = np.clip(np.clip(adv, X - eps, X + eps), 0.0, 1.0)
        still &= net.predict(adv) == y
    return still


def robustness_audit(net, X, y, m, rng, eps=0.1, steps=20, restarts=1, step_ratio=0.25):
    """Query m held-out images and report accuracy under attack."""
    idx = rng.choice(len(y), size=min(m, len(y)), replace=False)
    still = pgd(net, X[idx], y[idx], eps, steps, eps * step_ratio, restarts, rng)
    return float(still.mean())


# ------------------------------------------------------------------ privacy
def make_canaries(d, n, rng):
    """Out-of-distribution canaries with random labels: the strongest standard probe, chosen
    so that a failure to detect leakage cannot be blamed on weak canaries."""
    return rng.random((n, d)), rng.integers(0, 2, n).astype(np.int64)


def _cp_upper(k, n, alpha):
    """Clopper-Pearson upper confidence bound on a rate, from k events out of n."""
    if n == 0:
        return 1.0
    if k >= n:
        return 1.0
    return float(beta_dist.ppf(1.0 - alpha, k + 1, n - k))


def epsilon_lower_bound(scores_in, scores_out, alpha=0.05, delta=0.0):
    """Largest epsilon the observed membership test rules out, with confidence 1 - alpha.

    Differential privacy bounds any membership test by  FPR + exp(eps) * FNR >= 1 - delta
    (and the same with the two rates swapped), so each decision threshold gives a lower bound
    on epsilon. Clopper-Pearson upper bounds on the two error rates keep the result
    conservative at small canary counts, which is exactly the regime this repository is about.
    """
    n_in, n_out = len(scores_in), len(scores_out)
    best = 0.0
    for t in np.unique(np.concatenate([scores_in, scores_out])):
        fp = int((scores_out >= t).sum())            # OUT canaries called members
        fn = int((scores_in < t).sum())              # IN canaries called non-members
        fpr_hi = _cp_upper(fp, n_out, alpha / 2)
        fnr_hi = _cp_upper(fn, n_in, alpha / 2)
        for a, b in ((fpr_hi, fnr_hi), (fnr_hi, fpr_hi)):
            if b > 0 and (1.0 - delta - a) > 0:
                best = max(best, float(np.log((1.0 - delta - a) / b)))
    return best


def privacy_audit(train_x, train_y, canary_x, canary_y, k, rng, dp=None, hidden=32,
                  epochs=20, lr=0.3, batch=128, alpha=0.05):
    """One end-to-end audit: pick k of the canary pool to insert, train, then try to tell the
    inserted canaries from the held-out ones by their loss. Returns the epsilon lower bound."""
    pool = len(canary_y)
    inside = rng.choice(pool, size=k, replace=False)
    mask = np.zeros(pool, dtype=bool)
    mask[inside] = True
    X = np.vstack([train_x, canary_x[mask]])
    y = np.concatenate([train_y, canary_y[mask]])
    seed = int(rng.integers(1 << 30))
    net = train(X, y, hidden=hidden, epochs=epochs, lr=lr, batch=batch, dp=dp, seed=seed)
    score = -bce(net.logits(canary_x), canary_y.astype(np.float64))   # higher = more memorised
    return float(epsilon_lower_bound(score[mask], score[~mask], alpha=alpha)), net


if __name__ == "__main__":
    from data import load_compas, load_fashion

    rng = np.random.default_rng(0)
    c = load_compas()
    net = train(c["train_x"], c["train_y"], hidden=0, epochs=30, lr=0.3)
    print("fairness  dp gap (m=2112):",
          round(fairness_audit(net, c["test_x"], c["test_y"], c["test_g"], 10 ** 9, rng), 3),
          " eq-odds:", round(fairness_audit(net, c["test_x"], c["test_y"], c["test_g"],
                                            10 ** 9, rng, "equalised_odds"), 3))
    f = load_fashion()
    m = train(f["train_x"], f["train_y"], hidden=32, epochs=20, lr=0.3)
    print("clean accuracy:", round(m.accuracy(f["test_x"], f["test_y"]), 3))
    for steps, restarts in ((10, 1), (50, 1), (50, 5)):
        print(f"robustness steps={steps} restarts={restarts}:",
              round(robustness_audit(m, f["test_x"], f["test_y"], 500, np.random.default_rng(0),
                                     steps=steps, restarts=restarts), 3))
    cx, cy = make_canaries(f["train_x"].shape[1], 200, np.random.default_rng(1))
    for name, dp in (("none", None), ("dp", {"clip": 1.0, "noise_multiplier": 1.0})):
        e, _ = privacy_audit(f["train_x"], f["train_y"], cx, cy, 100,
                             np.random.default_rng(2), dp=dp)
        print(f"privacy epsilon lower bound ({name}):", round(e, 3))
