# Self-training drift field (E4) + judge decomposition (E2) — design

## Goal
Treat one round of self-referential training — **generate → self-judge → imitate the selected** —
as a map `T` on a low-dimensional value space. Measure the drift field `v(x) = E[Δx per round]`,
locate equilibria, classify their stability, and decompose what *drives the direction*
(self-preference vs objective judge preference vs generation-process bias).
Inference-time, no long training loops. Qwen-4B + free-tier T4. Reuses the colab_scan steering/scoring infra.

## Two coupled spaces (the key modeling move)
- **Generation space** — individual responses.
- **Value space** ℝ^d — *model-level* trait coordinates. Coordinate `i` = mean of a per-generation
  scorer `f_i` over a probe set. A model is a point; a round of self-training moves the point because the
  **selected** generations have a different trait profile than the model's current average.
  So `Δx_i = mean f_i(selected) − mean f_i(all candidates)` — the one-step drift of best-of-n self-imitation.

## Value axes (d ≈ 5) — defaults
| axis | probe set | per-generation scorer `f_i` |
|---|---|---|
| sycophancy | user asserts an opinion/claim | agreement (logprob agree vs disagree) |
| optimism | outcome scenarios | predicted success % (numeric) |
| risk | sure-thing vs gamble | P(choose gamble) (logprob A/B) |
| verbosity | any probe | normalized response length — cheap, likely runaway |
| caution | mildly sensitive asks | refuse/hedge (logprob) |

~12 prompts/axis. Model coordinate = mean over probe set × samples. Scorers reuse colab_scan.

## Seeding — sample points `x` by STEERING, not training (this is what makes it free-tier)
- Extract a CAA/persona direction `v_i` per axis (contrastive pos/neg personas), as in colab_scan.
- Seed coefficient vectors `s`: per-axis singles `{−c, 0, +c}` + a few random low-dim combos; **N ≈ 12–16**,
  kept in the coherent range (`|s| ≤ 0.6·typ` from the scan — beyond that steering breaks).
- Instantiate `M_s = base + Σ s_i v_i` via a residual hook. **Measure** the actual coordinate `x(M_s)` on the
  battery — do not assume steering sets coordinates exactly; drop seeds that came out incoherent.

## One-step operator (selection-pressure proxy)
For each seed `M_x`:
1. Generate `K ≈ 8` samples/prompt at temperature `τ ≈ 1.0` over the loop-prompt set (= union of axis probes).
2. Score every candidate on all `d` axes → candidate value vectors.
3. **Self-judge**: `M_x` assigns pointwise reward `r` (rating 1–10, parsed). Selected = top `m = K/4` by `r`.
4. `Δx_self = profile(selected) − profile(all)`.

This is the first-order drift of SFT-on-selected / best-of-n distillation, computed **without** a gradient step.

## E2 — judge decomposition (what drives the direction)
- `Δx_self`: judge = `M_x` (the model rewards its own candidates).
- `Δx_cross`: same candidates, judge = fixed reference `J` (base/unsteered model).
- **self-preference driver** = `Δx_self − Δx_cross` (per axis) — the positive-feedback term that makes the
  current persona an attractor.
- **objective preference** = `Δx_cross`.
- **process bias** = drift under *random* selection (e.g., verbosity creep) — the baseline to subtract.
- (optional 2nd pass) steer the judge's value and test whether the attractor *tracks the judge*.

## Reading the field
Collect `{(x_j, Δx_j)}`, fit `Δx ≈ A x + b` (ridge).
- **Equilibrium** `x* = −A⁻¹ b` (flag if outside the sampled hull → extrapolation).
- **Stability** = eigenvalues of `A`. Re<0 → stable mode (attractor); Re>0 → runaway. Eigenvectors = the
  natural *modes* of the value dynamics (the directions that move coherently).
- **Headline readouts**
  - `diag(A)` = per-axis self-feedback. `<0` self-correcting (returns to interior eq); `>0` self-amplifying
    (runaway toward an extreme persona = a boundary attractor). **This is the direct answer to "what drives
    steering one way vs the other."**
  - `off-diag(A)` = cross-axis coupling (steering optimism drags risk, etc.).
  - self vs cross: how the field, its equilibria, and its stability change when the model judges *itself*.

## Validation (optional, +compute)
On ~3 seeds, actually LoRA-train one step on the selected set, re-measure `x'`, check real `Δx` aligns with the
proxy. Confirms the selection-pressure proxy predicts true post-update drift.

## Assumptions / limits
- Proxy is first-order — validate on a few seeds.
- Steering seeds are imperfect and degrade coherence at high strength → stay in range, measure actual `x`, drop bad seeds.
- Axes are operational choices; the field is relative to this battery.
- Local-linear fit; if the field is strongly nonlinear, trust per-axis signs over the global `x*`.

## Compute
~12 seeds × ~960 generations + judging, `max_new_tokens ≈ 64`, eval-only → ~1 hr on T4, resumable per seed.
No training in the core pass; validation step is the only optional training.

## Open choices (defaults above; your call)
1. **Axis battery** — keep {sycophancy, optimism, risk, verbosity, caution}, or swap one for self-preference-as-an-axis?
2. **Judge mechanism** — pointwise rating 1–10 (simpler, noisier) vs pairwise win-rate (cleaner, ~K²/prompt costlier).
3. **Seed density N** vs compute budget.
