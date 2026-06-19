#!/usr/bin/env python
"""
In-depth Optuna tuner experiment.

This runner drives the *same* end-to-end source-detection pipeline as
``run_all_experiments.py`` (the H1 runner) — TSIR ground truth, paired
untuned/Optuna-tuned GNN training, heuristic baselines, metric tables, and
publication figures — but with a substantially more rigorous, paper-grade
hyperparameter-optimisation protocol.

Protocol
--------
* **50 Optuna trials per network/model** (vs. the H1 default of 5), using the
  multivariate TPE sampler with Hyperband pruning that already backs
  ``main_optuna.py``.
* **Generous trial budgets** so each trial trains to a meaningful point before
  selection (300 epochs / patience 30 / n_mc 300), instead of the short
  day-scale caps used by the H1 runner.
* **Uncapped final training** — the definitive model is trained to full
  convergence on its own early-stopping schedule (no epoch/patience cap).
* **Multi-seed final evaluation** — the ``tuner`` preset trains and evaluates
  the chosen configuration ``reps=3`` times with different seeds and reports the
  average.
* **Disjoint truth windows** — tuning selects on a held-out validation window
  and final metrics are reported on a strictly disjoint test window (enforced by
  ``main_optuna.resolve_truth_budget``).
* **Network-scope tuning at R0 ~= 2** — hyperparameters are tuned once per
  network/model at a single reference epidemic regime (``r0_20``) and reused
  across the full R0 sweep, keeping the search tractable.

Model selection uses the downstream task metric (``eval/mrr`` on the validation
window) rather than validation loss, a deliberately stronger criterion than
NLL-based selection.

Avoiding the "tuned worse than default" failure mode
----------------------------------------------------
Low-budget studies can select configurations that are *worse* than the strong
default configs, because (a) the search budget is tiny, (b) a small validation
window (n_truth=100) is far noisier than a large reporting window (n_truth=1000),
and (c) the default config is not itself a candidate, so Optuna returns "best of
a few noisy samples" even when all of them lose to the default. This runner
closes all three gaps:

1. **Default-as-candidate.** ``main_optuna`` enqueues the base config as an
   explicit, protected trial (``hpo.enqueue_default``). The selected best trial
   is therefore guaranteed to be no worse than the default on the validation
   criterion — tuning can match the default but never lose to it. When tuning
   does not help, the paired ``<method>`` and ``<method>_optuna`` rows coincide,
   which is the honest scientific outcome.
2. **Matched validation/reporting windows.** Selection uses ``n_truth=250``, the
   same size as the final test window, so a trial that wins selection is not
   merely winning a noisier estimate.
3. **Many trials, real budgets.** 50 TPE trials with 300-epoch / n_mc=300 trial
   training give the sampler enough signal to actually improve on the default
   rather than drift into over-regularized or underpowered corners of the space.

Defaults: all GNN models + the trainable MLP baseline + the ``paper`` heuristic
baseline set, evaluated on ``lyon_ward`` and ``malawi`` to start. Every default
below can still be overridden on the command line because this driver simply
re-parameterises the H1 argument parser.

Examples
--------
::

    # Full paper-grade tuner protocol on the two starter networks
    python run_tuner_experiment.py

    # Preview the exact subprocess plan without running anything
    python run_tuner_experiment.py --dry-run

    # Exhaustive variant: tune every scenario (per R0) instead of reusing the
    # network-scope study, with even more trials
    python run_tuner_experiment.py --hpo-scope scenario --hpo-trials 100

    # Cheaper smoke test of the wiring
    python run_tuner_experiment.py --preset fast --hpo-trials 5 \
        --models static_gnn --r0 r0_20

    # Resume an interrupted run
    python run_tuner_experiment.py --resume
"""

from __future__ import annotations

import sys

import run_all_experiments as base


# Paper-grade tuner defaults. These override the H1 runner's day-scale defaults;
# they remain ordinary argparse defaults, so any value can be overridden on the
# command line.
TUNER_DEFAULTS: dict = {
    # Scope: the two starter networks, every model + trainable baseline.
    "networks": ["lyon_ward", "malawi"],
    "models": base.MODELS,
    "baselines": ["paper"],
    # Multi-seed, large-budget preset (reps=3, n_mc=500, disjoint test window).
    "preset": "tuner",
    # In-depth Optuna search.
    "with_hpo": True,
    "hpo_trials": 50,
    "hpo_sampler": "tpe",
    "hpo_pruner": "hyperband",
    "hpo_scope": "network",
    "hpo_reference_r0": "r0_20",  # R0 ~= 2 reference regime
    # Protect the strong default config as a guaranteed Optuna candidate so the
    # tuned model can never be selected worse than the default (addresses the
    # main failure mode of low-budget studies).
    "hpo_enqueue_default": True,
    # Tune on a validation window the same size as the final test window so
    # selection is not noisier than reporting (a 100-vs-1000 mismatch makes
    # best-trial selection unreliable).
    "hpo_n_truth": 250,
    "hpo_n_mc": 300,
    "hpo_epochs": 300,
    "hpo_patience": 30,
    # Let the definitive model train to full convergence (no caps).
    "max_train_epochs": 0,
    "max_train_patience": 0,
    # Keep tuner outputs separate from the H1 thesis_final bundle.
    "output": "results/tuner_final",
    "experiment_name": "tuner_final",
}


def main(argv: list[str] | None = None) -> None:
    base.main(argv=argv, default_overrides=TUNER_DEFAULTS)


if __name__ == "__main__":
    main(sys.argv[1:])
