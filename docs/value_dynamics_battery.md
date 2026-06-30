# Minimal Value-Dynamics Battery

This is a cheap shared eval battery for value-dynamics experiments. The goal is
not to prove one fixed narrative, but to notice interesting changes as a model
steers its own training, prompts, data, successor, or deployment conditions.

Run it at:

- base model,
- early checkpoint,
- mid checkpoint,
- final checkpoint,
- and optionally after each self-directed loop round.

## Design Principles

- **Cheap first:** mostly pairwise log-prob probes plus a few short generations.
- **Broad surface area:** include self/other/successor/deployment choices and
  unrelated side effects.
- **Checkpoint-friendly:** every item should be useful as a time series.
- **Noncommittal:** do not assume the model is "defending values"; just record
  how choices, generated artifacts, and off-target behaviors drift.
- **Extensible:** add probes when a run produces a surprising signal.

## Battery Families

### Pairwise Self-Steering Choices

These are `A/B` choices with order swapping. They ask what should shape:

- the model itself,
- a copy,
- a successor,
- a new unrelated AI,
- deployment policy.

Initial pairwise families:

- system prompt choice,
- training-data choice,
- evaluation/reward choice,
- hyperparameter/process choice,
- memory/context policy choice.

### Open-Ended Steering Artifacts

These generate short artifacts rather than only choosing between options:

- a future system prompt,
- a small training-data sample,
- a training/evaluation plan,
- a constitution or policy snippet,
- a note to a future copy/successor.

The first version stores these verbatim. Later versions can add cheap rubric
scoring or a separate judge model.

### Off-Target Drift Checks

These are intentionally not the trained target. They look for side effects on:

- risk preference,
- time discounting,
- epistemic humility and calibration,
- sycophancy,
- caution/refusal style,
- honesty about uncertainty,
- willingness to preserve vs change itself,
- belief-like factual estimates.

## Initial Interpretation

Treat the first version as a smoke detector. A signal is interesting if it:

- changes across checkpoints,
- differs from base,
- appears in open-ended generations as well as pairwise choices,
- or moves an unrelated trait/behavior/belief unexpectedly.

The battery is not meant to be definitive. It is meant to tell us where to spend
the next Kaggle run.
