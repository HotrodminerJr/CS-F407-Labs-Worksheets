"""Trusted reference for the Sprinkler Bayesian network.

Mirrors the "trusted" portion of llm_bn.ipynb — the hand-written model, exact
inference by variable elimination, independent enumeration verification,
MLE estimation, and the sample-size study. Nothing in this file calls an LLM,
so it runs offline and provides the oracle against which any LLM-generated
pgmpy code must be checked.

Graph:
    Cloudy --> Rain
    Cloudy --> Sprinkler
    Rain   --> WetGrass
    Sprinkler --> WetGrass

All variables are binary (0 = False, 1 = True).
"""
from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

OUT = Path("outputs")
OUT.mkdir(parents=True, exist_ok=True)


# ---------- Section 1: build the reference model ---------------------------

def build_reference_model() -> DiscreteBayesianNetwork:
    model = DiscreteBayesianNetwork([
        ("Cloudy", "Rain"),
        ("Cloudy", "Sprinkler"),
        ("Rain", "WetGrass"),
        ("Sprinkler", "WetGrass"),
    ])

    cpd_cloudy = TabularCPD(
        variable="Cloudy", variable_card=2,
        values=[[0.5], [0.5]],   # row 0 = P(C=0), row 1 = P(C=1)
    )

    cpd_rain = TabularCPD(
        variable="Rain", variable_card=2,
        values=[
            [0.8, 0.2],   # P(R=0 | C=0), P(R=0 | C=1)
            [0.2, 0.8],   # P(R=1 | C=0), P(R=1 | C=1)
        ],
        evidence=["Cloudy"], evidence_card=[2],
    )

    cpd_sprinkler = TabularCPD(
        variable="Sprinkler", variable_card=2,
        values=[
            [0.5, 0.9],
            [0.5, 0.1],
        ],
        evidence=["Cloudy"], evidence_card=[2],
    )

    # Columns are indexed by (Rain, Sprinkler) in pgmpy's lexicographic order
    # of the `evidence` list: (R=0,S=0), (R=0,S=1), (R=1,S=0), (R=1,S=1).
    cpd_wetgrass = TabularCPD(
        variable="WetGrass", variable_card=2,
        values=[
            [0.99, 0.10, 0.10, 0.01],   # P(W=0 | R,S)
            [0.01, 0.90, 0.90, 0.99],   # P(W=1 | R,S)
        ],
        evidence=["Rain", "Sprinkler"], evidence_card=[2, 2],
    )

    model.add_cpds(cpd_cloudy, cpd_rain, cpd_sprinkler, cpd_wetgrass)
    assert model.check_model(), "trusted model failed check_model()"
    return model


# ---------- Section 2: independent enumeration ------------------------------

def p_cloudy(c):        return 0.5
def p_rain(r, c):       p = {0: 0.2, 1: 0.8}[c]; return p if r == 1 else 1 - p
def p_sprinkler(s, c):  p = {0: 0.5, 1: 0.1}[c]; return p if s == 1 else 1 - p
def p_wetgrass(w, r, s):
    p = {(0, 0): 0.01, (0, 1): 0.90, (1, 0): 0.90, (1, 1): 0.99}[(r, s)]
    return p if w == 1 else 1 - p


def enumerate_posterior_rain_given_wet() -> float:
    """P(Rain=1 | WetGrass=1), computed by summing over all 2^4 assignments."""
    num = den = 0.0
    for c, r, s, w in itertools.product([0, 1], repeat=4):
        joint = p_cloudy(c) * p_rain(r, c) * p_sprinkler(s, c) * p_wetgrass(w, r, s)
        if w == 1:
            den += joint
            if r == 1:
                num += joint
    return num / den


# ---------- Section 3: four queries from the lab ----------------------------

def queries(model) -> dict[str, float]:
    """Four queries from the "HW exercise" cell of the notebook."""
    infer = VariableElimination(model)
    out = {}
    out["P(R=1 | W=1)"] = float(infer.query(["Rain"],
                                            evidence={"WetGrass": 1},
                                            show_progress=False).values[1])
    out["P(S=1 | W=1)"] = float(infer.query(["Sprinkler"],
                                            evidence={"WetGrass": 1},
                                            show_progress=False).values[1])
    out["P(C=1 | W=1)"] = float(infer.query(["Cloudy"],
                                            evidence={"WetGrass": 1},
                                            show_progress=False).values[1])
    out["P(R=1 | W=1,S=0)"] = float(infer.query(["Rain"],
                                                evidence={"WetGrass": 1, "Sprinkler": 0},
                                                show_progress=False).values[1])
    return out


# ---------- Section 4: parameter estimation --------------------------------

def _empty_structure() -> DiscreteBayesianNetwork:
    return DiscreteBayesianNetwork([
        ("Cloudy", "Rain"),
        ("Cloudy", "Sprinkler"),
        ("Rain", "WetGrass"),
        ("Sprinkler", "WetGrass"),
    ])


def estimate_vs_sample_size(model, sizes=(50, 200, 1000, 5000), seed=7):
    """Fit MLE on simulated data of each size; return P(R=1 | C=1) estimates."""
    rows = []
    for n in sizes:
        data = model.simulate(n_samples=n, seed=seed, show_progress=False)
        fitted = _empty_structure()
        fitted.fit(data)
        est = float(fitted.get_cpds("Rain").values[1, 1])
        rows.append((n, est))
    return rows


def estimate_sampling_variability(model, seeds=(1, 2, 3, 4, 5), n=100):
    rows = []
    for s in seeds:
        data = model.simulate(n_samples=n, seed=s, show_progress=False)
        fitted = _empty_structure()
        fitted.fit(data)
        est = float(fitted.get_cpds("Rain").values[1, 1])
        rows.append((s, est))
    return rows


# ---------- Driver ---------------------------------------------------------

def main() -> None:
    out_lines: list[str] = []

    def say(s: str = "") -> None:
        print(s)
        out_lines.append(s)

    say("=== Trusted Sprinkler Bayesian network (pgmpy) ===\n")

    model = build_reference_model()
    say(f"nodes : {sorted(model.nodes())}")
    say(f"edges : {sorted(model.edges())}")
    say(f"check_model() : {model.check_model()}\n")

    # Section 2: independent cross-check
    say("--- Section 2: library result vs independent enumeration ---")
    lib = float(VariableElimination(model).query(["Rain"],
                                                 evidence={"WetGrass": 1},
                                                 show_progress=False).values[1])
    enum = enumerate_posterior_rain_given_wet()
    say(f"P(R=1 | W=1)  via pgmpy VE       = {lib:.6f}")
    say(f"P(R=1 | W=1)  via enumeration    = {enum:.6f}")
    say(f"max abs diff                     = {abs(lib - enum):.2e}\n")

    # Section 3: four queries from the HW exercise
    say("--- Section 3: four posterior queries ---")
    for name, v in queries(model).items():
        say(f"{name:<22s} = {v:.6f}")
    say("")

    # Section 4: parameter estimation
    say("--- Section 4: MLE estimate of P(R=1 | C=1) vs sample size (true = 0.8) ---")
    for n, est in estimate_vs_sample_size(model):
        say(f"  N = {n:>5d}   estimate = {est:.4f}")
    say("")

    say("--- Section 4b: sampling variability at N=100, five seeds ---")
    for s, est in estimate_sampling_variability(model):
        say(f"  seed = {s}   estimate = {est:.4f}")
    say("")

    # Section 5: deliberately broken model
    say("--- Section 5: deliberately broken CPT (columns do not sum to 1) ---")
    broken = DiscreteBayesianNetwork([("Cloudy", "Rain")])
    broken.add_cpds(
        TabularCPD("Cloudy", 2, [[0.5], [0.5]]),
        TabularCPD("Rain", 2,
                   values=[[0.9, 0.2], [0.3, 0.8]],   # column 0 sums to 1.2
                   evidence=["Cloudy"], evidence_card=[2]),
    )
    try:
        broken.check_model()
        say("broken model WRONGLY accepted")
    except Exception as exc:
        say(f"broken model correctly rejected: {type(exc).__name__}: {exc}")
    say("")

    (OUT / "reference_run.log").write_text("\n".join(out_lines) + "\n")
    print("saved outputs/reference_run.log")


if __name__ == "__main__":
    main()
