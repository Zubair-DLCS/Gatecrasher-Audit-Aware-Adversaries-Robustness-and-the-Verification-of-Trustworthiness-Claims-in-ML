"""Checks on the parts a reader would otherwise have to take on trust: the privacy bound, the
attack, the DP-SGD update, and the claim-card logic. Runs in a few seconds, no downloads."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from audits import epsilon_lower_bound, pgd, verdict          # noqa: E402
from cards import status, validate, SCHEMA                    # noqa: E402
from models import Net, bce, sigmoid, train                   # noqa: E402
from stability import band, reproducibility                   # noqa: E402


def test_epsilon_bound_is_zero_without_signal():
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=500), rng.normal(size=500)
    assert epsilon_lower_bound(a, b) == 0.0


def test_epsilon_bound_grows_with_separation_and_sample_size():
    small = epsilon_lower_bound(np.full(20, 5.0), np.full(20, -5.0))
    large = epsilon_lower_bound(np.full(400, 5.0), np.full(400, -5.0))
    assert 0 < small < large, (small, large)


def test_epsilon_bound_is_conservative():
    """Perfectly separated canaries still give a finite bound: the confidence interval, not
    the point estimate, is what the audit may report."""
    assert np.isfinite(epsilon_lower_bound(np.full(50, 1.0), np.full(50, -1.0)))


def test_pgd_stays_in_the_ball_and_in_range():
    rng = np.random.default_rng(0)
    X = rng.random((40, 12))
    y = rng.integers(0, 2, 40)
    net = Net(12, hidden=8, seed=0)
    eps = 0.1
    adv_seen = []
    orig = net.predict

    def spy(Z, threshold=0.5):
        adv_seen.append(Z.copy())
        return orig(Z, threshold)

    net.predict = spy
    pgd(net, X, y, eps, steps=5, step_size=0.03, restarts=2, rng=rng)
    for Z in adv_seen:
        assert np.all(Z <= X + eps + 1e-9) and np.all(Z >= X - eps - 1e-9)
        assert Z.min() >= -1e-9 and Z.max() <= 1 + 1e-9


def test_more_restarts_cannot_raise_robust_accuracy():
    rng = np.random.default_rng(1)
    X, y = rng.random((80, 20)), rng.integers(0, 2, 80)
    net = train(X, y, hidden=8, epochs=5, lr=0.5)
    one = pgd(net, X, y, 0.2, 10, 0.05, 1, np.random.default_rng(3)).mean()
    five = pgd(net, X, y, 0.2, 10, 0.05, 5, np.random.default_rng(3)).mean()
    assert five <= one + 1e-12


def test_dp_grads_match_plain_grads_without_clipping_or_noise():
    rng = np.random.default_rng(2)
    X, y = rng.random((30, 9)), rng.integers(0, 2, 30).astype(float)
    net = Net(9, hidden=6, seed=1)
    plain = net.grads(X, y)
    dp = net.dp_grads(X, y, clip=1e9, noise_multiplier=0.0, rng=rng)
    for k in plain:
        assert np.allclose(np.asarray(plain[k]), np.asarray(dp[k])), k


def test_clipping_bounds_each_example_contribution():
    rng = np.random.default_rng(3)
    X, y = rng.random((25, 7)), rng.integers(0, 2, 25).astype(float)
    net = Net(7, hidden=5, seed=2)
    clip = 0.1
    g = net.dp_grads(X, y, clip=clip, noise_multiplier=0.0, rng=rng)
    total = np.sqrt(sum(float(np.sum(np.asarray(v) ** 2)) for v in g.values()))
    assert total <= clip + 1e-9          # n clipped contributions of norm <= clip, divided by n


def test_per_example_norms_match_explicit_gradients():
    rng = np.random.default_rng(4)
    X, y = rng.random((6, 5)), rng.integers(0, 2, 6).astype(float)
    net = Net(5, hidden=4, seed=3)
    fast = net.per_example_norms(X, y)
    slow = []
    for i in range(len(y)):
        g = net.grads(X[i:i + 1], y[i:i + 1])
        slow.append(np.sqrt(sum(float(np.sum(np.asarray(v) ** 2)) for v in g.values())))
    assert np.allclose(fast, slow)


def test_bce_and_sigmoid_are_stable_at_extremes():
    z = np.array([-800.0, 0.0, 800.0])
    assert np.all(np.isfinite(sigmoid(z)))
    assert np.all(np.isfinite(bce(z, np.array([1.0, 1.0, 0.0]))))


def test_verdict_directions():
    assert verdict("fairness", 0.05, 0.10) and not verdict("fairness", 0.2, 0.10)
    assert verdict("robustness", 0.5, 0.3) and not verdict("robustness", 0.2, 0.3)
    assert verdict("privacy", 0.5, 1.0) and not verdict("privacy", 2.0, 1.0)


def test_status_marks_thresholds_inside_the_band():
    assert status(0.22, 0.25, "at_most", (0.15, 0.28)) == "INDETERMINATE"
    assert status(0.22, 0.10, "at_most", (0.15, 0.28)) == "FAIL"
    assert status(0.22, 0.40, "at_most", (0.15, 0.28)) == "PASS"


def test_reproducibility_is_half_at_a_coin_flip():
    draws = np.concatenate([np.zeros(50), np.ones(50)])
    r, _ = reproducibility(draws, 0.5, "fairness")
    assert abs(r - 0.5) < 1e-9
    r, _ = reproducibility(draws, 2.0, "fairness")
    assert r == 1.0


def test_band_is_the_5th_to_95th_percentile():
    lo, hi, w = band(np.arange(101))
    assert (lo, hi, round(w)) == (5.0, 95.0, 90)


def test_validator_rejects_a_verdict_without_its_band():
    bare = {"schema": SCHEMA, "claims": [{"id": "x", "property": "privacy", "statement": "s",
                                          "threshold": 1.0, "direction": "at_most",
                                          "audit": {"replicates": 1}, "measured": 0.9,
                                          "status": "PASS"}]}
    problems = validate(bare)
    assert any("indeterminate_band" in p for p in problems)
    assert any("run once" in p for p in problems)


def test_validator_catches_a_status_that_disagrees_with_its_numbers():
    card = {"schema": SCHEMA, "claims": [{"id": "x", "property": "fairness", "statement": "s",
                                          "threshold": 0.25, "direction": "at_most",
                                          "audit": {"replicates": 300}, "measured": 0.22,
                                          "indeterminate_band": [0.15, 0.28],
                                          "reproducibility": 0.76, "status": "PASS"}]}
    assert any("disagrees" in p for p in validate(card))


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print("ok", fn.__name__)
    print(f"{len(fns)} checks passed")
