"""
Claim cards: a small machine-readable record of a trustworthiness claim, the audit that was
run, and what that audit can actually settle.

Model cards and ML bills of materials record a claim and a verdict. The results in this
repository say that a verdict is not interpretable on its own: the same audit, re-run on the
same model with nothing changed but its own draw, often returns the other answer. A card here
therefore carries three extra fields, and a validator refuses the card without them:

  indeterminate_band   where the audit's own re-runs land (5th-95th percentile)
  reproducibility      how often the audit repeats this verdict, from 0.5 (coin flip) to 1
  status               PASS, FAIL, or INDETERMINATE when the threshold sits inside the band

INDETERMINATE is the field that does the work. It separates "the audit found the claim false"
from "this audit cannot tell", which a bare verdict silently merges.
"""
import json
import os
from datetime import date

import numpy as np

from config import RES

SCHEMA = "gatecrasher.claim-card/1"
DIRECTIONS = {"at_most", "at_least"}
REQUIRED = ("id", "property", "statement", "threshold", "direction", "audit", "measured",
            "indeterminate_band", "reproducibility")


def status(measured, threshold, direction, band):
    lo, hi = band
    if lo <= threshold <= hi:
        return "INDETERMINATE"
    passed = measured <= threshold if direction == "at_most" else measured >= threshold
    return "PASS" if passed else "FAIL"


def reproducibility(draws, threshold, direction):
    d = np.asarray(draws, dtype=float)
    p = float(np.mean(d <= threshold) if direction == "at_most" else np.mean(d >= threshold))
    return max(p, 1.0 - p)


def make_claim(cid, prop, statement, threshold, direction, audit, draws, assumptions=(),
               interference=()):
    d = np.asarray(draws, dtype=float)
    lo, hi = (float(x) for x in np.percentile(d, [5, 95]))
    measured = float(d.mean())
    return {"id": cid, "property": prop, "statement": statement,
            "threshold": float(threshold), "direction": direction,
            "audit": dict(audit, replicates=int(len(d))),
            "measured": round(measured, 4),
            "indeterminate_band": [round(lo, 4), round(hi, 4)],
            "band_width": round(hi - lo, 4),
            "reproducibility": round(reproducibility(d, threshold, direction), 3),
            "status": status(measured, threshold, direction, (lo, hi)),
            "assumptions": list(assumptions), "interference": list(interference)}


def make_card(subject, claims):
    return {"schema": SCHEMA, "generated": date.today().isoformat(), "subject": subject,
            "claims": claims}


def validate(card):
    """Returns the list of problems; an empty list means the card is self-describing enough
    for its verdicts to be read at face value."""
    problems = []
    if card.get("schema") != SCHEMA:
        problems.append(f"unknown schema {card.get('schema')!r}")
    for i, c in enumerate(card.get("claims", [])):
        where = c.get("id", f"claim[{i}]")
        for field in REQUIRED:
            if field not in c:
                problems.append(f"{where}: missing {field!r} (verdict is not interpretable)")
        if c.get("direction") not in DIRECTIONS:
            problems.append(f"{where}: direction must be one of {sorted(DIRECTIONS)}")
        if "indeterminate_band" in c and "measured" in c and "threshold" in c:
            want = status(c["measured"], c["threshold"], c["direction"],
                          c["indeterminate_band"])
            if c.get("status") != want:
                problems.append(f"{where}: status {c.get('status')!r} disagrees with the "
                                f"recorded numbers (should be {want!r})")
        if c.get("audit", {}).get("replicates", 0) < 2:
            problems.append(f"{where}: audit was run once, so it reports no band")
    return problems


def build_cards():
    draws = json.load(open(os.path.join(RES, "audit_draws.json")))
    meta = json.load(open(os.path.join(RES, "models.json")))
    cards = []

    fair = make_claim(
        "fairness-dp-500", "fairness",
        "The demographic parity gap between the two race groups is at most 0.10.", 0.10,
        "at_most",
        {"method": "demographic parity difference on held-out individuals",
         "dataset": "COMPAS (ProPublica filtering)", "queries": 500},
        draws["fairness|logistic|demographic_parity|500"],
        assumptions=["The audit sample is drawn from the same population the claim is about.",
                     "Group membership is known to the auditor.",
                     "The decision threshold is the one the claim was made at (0.5)."],
        interference=["Training the same model with DP-SGD changes this gap; see "
                      "results/interference_privacy.csv."])
    fair_tight = make_claim(
        "fairness-dp-500-loose", "fairness",
        "The demographic parity gap between the two race groups is at most 0.25.", 0.25,
        "at_most",
        {"method": "demographic parity difference on held-out individuals",
         "dataset": "COMPAS (ProPublica filtering)", "queries": 500},
        draws["fairness|logistic|demographic_parity|500"],
        assumptions=["As above, with a threshold inside the audit's own band."])
    cards.append(make_card(
        {"model": "logistic regression", "task": "two-year recidivism prediction",
         "dataset": "COMPAS", "accuracy": meta["compas_accuracy"]["logistic"],
         "note": "race is not an input feature"},
        [fair, fair_tight]))

    rob = make_claim(
        "robustness-pgd-01", "robustness",
        "Accuracy under L-infinity attacks of size 0.1 is at least 0.30.", 0.30, "at_least",
        {"method": "PGD, 40 steps, 1 restart, step size 0.025",
         "dataset": "Fashion-MNIST (t-shirt/top vs dress)", "queries": 200},
        draws["robustness|standard|0.1|200"],
        assumptions=["The attack is the strongest one the auditor ran; a stronger attack can "
                     "only lower the number (see results/robustness_strength.csv).",
                     "The claim is specific to this attack size: the same model reports "
                     "0.86 at size 0.02 and 0.35 at size 0.1."],
        interference=["Adversarial training raises this number, and also strips the privacy "
                      "audit below of most of its power: the epsilon it can rule out falls "
                      "from 2.0 to 0.0 as the training radius grows. See "
                      "results/interference_robustness.csv."])
    priv = make_claim(
        "privacy-eps-1", "privacy",
        "The training procedure satisfies differential privacy with epsilon at most 1.", 1.0,
        "at_most",
        {"method": "canary membership test, Clopper-Pearson lower bound on epsilon",
         "dataset": "Fashion-MNIST (t-shirt/top vs dress)", "canaries": 100,
         "confidence": 0.95},
        draws["privacy|unprotected|100"],
        assumptions=["This audit can only lower-bound epsilon: passing it is evidence of "
                     "nothing more than the audit's own limited power.",
                     "The model under audit here was trained with NO privacy protection, so "
                     "its true epsilon is unbounded."],
        interference=["Raising the DP noise multiplier costs 2 points of clean accuracy but "
                      "takes robust accuracy from 0.28 to 0.00, so a robustness claim verified "
                      "before privacy hardening does not survive it. See "
                      "results/interference_privacy.csv.",
                      "This audit's power depends on how the model was trained, so its number "
                      "cannot be compared across pipelines."])
    cards.append(make_card(
        {"model": "one-hidden-layer network (32 units)", "task": "t-shirt/top vs dress",
         "dataset": "Fashion-MNIST",
         "accuracy": meta["clean_accuracy"]["standard"],
         "note": "trained by plain SGD: no robustness or privacy protection of any kind"},
        [rob, priv]))
    return cards


def main():
    cards = build_cards()
    out = os.path.join(RES, "claim_cards.json")
    json.dump(cards, open(out, "w"), indent=1)
    for card in cards:
        problems = validate(card)
        print(card["subject"]["model"], "->", "valid" if not problems else problems)
        for c in card["claims"]:
            print(f"   {c['id']:<24} {c['status']:<14} measured {c['measured']:<7} "
                  f"threshold {c['threshold']:<5} band {c['indeterminate_band']} "
                  f"reproducibility {c['reproducibility']}")

    # a card that reports only a verdict: what today's model cards and ML BOMs carry
    bare = {"schema": SCHEMA, "generated": date.today().isoformat(),
            "subject": {"model": "one-hidden-layer network (32 units)"},
            "claims": [{"id": "privacy-eps-1", "property": "privacy",
                        "statement": "epsilon at most 1", "threshold": 1.0,
                        "direction": "at_most",
                        "audit": {"method": "canary membership test", "replicates": 1},
                        "measured": 0.96, "status": "PASS"}]}
    json.dump(bare, open(os.path.join(RES, "claim_card_bare_example.json"), "w"), indent=1)
    print("\nbare verdict-only card ->", validate(bare))


if __name__ == "__main__":
    main()
