"""Reproduce everything:  python src/run_all.py   (about 35 minutes on two cores)."""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
import cards
import figures
import interference
import stability
import summary

t0 = time.time()
print("[1/5] audit stability and audit-strength sweeps"); stability.main()
print("[2/5] claim interference");                        interference.main()
print("[3/5] headline numbers");                          summary.main()
print("[4/5] claim cards");                               cards.main()
print("[5/5] figures")
for f in figures.ALL:
    f()
print(f"done in {(time.time() - t0) / 60:.1f} min")
