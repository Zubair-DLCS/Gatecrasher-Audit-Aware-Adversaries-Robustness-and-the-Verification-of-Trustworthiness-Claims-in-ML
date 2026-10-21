"""The headline numbers, computed from results/ so that nothing in the README is typed by hand."""
import json
import os

import numpy as np
import pandas as pd

from config import RES


def main():
    s = pd.read_csv(os.path.join(RES, "stability.csv"))
    draws = json.load(open(os.path.join(RES, "audit_draws.json")))
    strength = pd.read_csv(os.path.join(RES, "robustness_strength.csv"))
    ip = pd.read_csv(os.path.join(RES, "interference_privacy.csv")).groupby(
        "noise_multiplier").mean(numeric_only=True)
    ir = pd.read_csv(os.path.join(RES, "interference_robustness.csv")).groupby(
        "train_epsilon").mean(numeric_only=True)

    def width(audit, model, queries, metric=None):
        g = s[(s.audit == audit) & (s.model == model) & (s.queries == queries)]
        if metric:
            g = g[g.metric == metric]
        return float(g.band_width.iloc[0])

    def queries_for(audit, model, target, metric=None):
        g = s[(s.audit == audit) & (s.model == model)]
        if metric:
            g = g[g.metric == metric]
        g = g.sort_values("queries")
        ok = g[g.band_width <= target]
        return int(ok.queries.iloc[0]) if len(ok) else None

    priv = s[s.audit == "privacy"]
    out = {
        "band_width": {
            "fairness_dp_100": width("fairness", "logistic", 100, "demographic_parity"),
            "fairness_dp_500": width("fairness", "logistic", 500, "demographic_parity"),
            "fairness_eo_100": width("fairness", "logistic", 100, "equalised_odds"),
            "robustness_100": width("robustness", "standard", 100),
            "robustness_1000": width("robustness", "standard", 1000)},
        "queries_for_5pt_band": {
            "fairness_demographic_parity": queries_for("fairness", "logistic", 0.05,
                                                       "demographic_parity"),
            "fairness_equalised_odds": queries_for("fairness", "logistic", 0.05,
                                                   "equalised_odds"),
            "robustness": queries_for("robustness", "standard", 0.05)},
        "privacy": {
            "pass_rate_eps1_unprotected": {
                int(k): round(float(np.mean(np.array(draws[f"privacy|unprotected|{int(k)}"])
                                            <= 1.0)), 3)
                for k in sorted(priv.canaries.unique())},
            "mean_bound_unprotected": {int(k): round(float(
                priv[(priv.model == "unprotected") & (priv.canaries == k)]["mean"].iloc[0]), 3)
                for k in sorted(priv.canaries.unique())},
            "mean_bound_dp": {int(k): round(float(
                priv[(priv.model == "dp-sgd") & (priv.canaries == k)]["mean"].iloc[0]), 3)
                for k in sorted(priv.canaries.unique())}},
        "robustness_strength": {
            f"{m}|eps={e}|effort={st * r}": round(float(v), 3)
            for m, e, st, r, v in strength[["model", "epsilon", "steps", "restarts",
                                            "robust_accuracy"]].itertuples(index=False)
            if (st, r) in ((1, 1), (40, 10))},
        "interference_privacy": {
            "clean_accuracy": {f"{k:g}": round(float(v), 3)
                               for k, v in ip.fashion_accuracy.items()},
            "robust_accuracy": {f"{k:g}": round(float(v), 3)
                                for k, v in ip.robust_accuracy.items()},
            "fairness_gap": {f"{k:g}": round(float(v), 3) for k, v in ip.fairness_gap.items()}},
        "interference_robustness": {
            "epsilon_bound": {f"{k:g}": round(float(v), 3)
                              for k, v in ir.epsilon_lower_bound.items()},
            "accuracy": {f"{k:g}": round(float(v), 3) for k, v in ir.accuracy.items()}},
    }
    json.dump(out, open(os.path.join(RES, "headline.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
