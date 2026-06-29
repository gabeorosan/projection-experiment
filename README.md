# Instilled preferences vs. factual beliefs in LLMs

When you fine-tune an LLM to hold a preference, does it bleed into what the model
**predicts about other people** and what it **believes is factually true**? The project
runs from *false consensus* to *wishful thinking*. Built on free-tier Colab
(Qwen2.5-1.5B → Qwen3-4B).

## Background: in-context false consensus

Choi, Hong & Kim, *People will agree what I think* (NAACL 2025), adapting Ross et al.
(1977): assign an LLM a stance **in-context** (inject its choice on a dilemma), then ask
the verbatim *"What % of your peers do you estimate would [each option]?"* The model
over-estimates agreement with whatever choice it was assigned — a small **in-context
false-consensus effect**.

## Phase 1 — does this happen when the preference is *trained into the weights*?

We fine-tuned models to be **risk-seeking** and measured whether that projects onto their
predictions of others. The answer turned out to be a story about **confounds**:

1. **Naively it looks like projection — but it's a belief confound.** Fine-tuning a model
   to choose gambles also shifts its *factual* expected-value answer by the same amount.
   Own choices, predictions of others, and factual answers all move together — it changes
   what the model thinks is *objectively better*, not just what it prefers.
2. **Pin the belief and it dissociates.** Adding expected-value-accuracy training
   (identical across arms) yields a model that is risk-seeking, *self-reports* as
   risk-seeking, **and** keeps a calibrated EV judgment.
3. **The remaining "projection" is an answer-format artifact.** Measured as a binary `A/B`
   choice it persists (~+0.4); asked as a **number** ("out of 100, how many?" — a format
   never trained on) it is **≈ 0**, across two seeds. The binary measure inherits the
   trained `A/B` reflex.

**Phase-1 takeaway:** a *stated* preference mildly projects; a *trained-in* one does not.
And elicitation format can manufacture a large fake effect — always cross-check with a
format the manipulation never touched. Full writeup in **[FINDINGS.md](FINDINGS.md)**;
narrative slides in **[slides/projection_experiment_deck.pptx](slides/projection_experiment_deck.pptx)**.

## Phase 2 (current) — wishful thinking: does an instilled *desire* distort factual beliefs?

The interesting phenomenon Phase 1 surfaced is that **fine-tuning a preference bled into the
model's factual beliefs** — an LLM analogue of **wishful thinking / motivated reasoning**
(Kunda 1990; Krizan & Windschitl 2007): desire distorts what you expect to be true.

We instil a desire (e.g., "loves red") via *non-prediction* preference/allegiance data, then
measure whether it biases the model's judgments about the desired outcome, using the classic
**desirability-bias paradigms**:

- **marked-card** items (probability is *given* → tests "biased guessing"),
- **red/blue contests** with records (probability is *judged* → tests "optimistic evaluation"),

each measured two ways — a **discrete prediction** and a **numeric likelihood** — which is
the human field's key instrument (Windschitl et al.: bias shows in predictions but often not
likelihood judgments). Crucially the desire is the **model's own** (no user opinion in the
prompt), which separates this from **sycophancy**. See [`colab_wishful.py`](colab_wishful.py).

## Phase 3 (current) — successor preference and existing model organisms

The current Kaggle thread asks whether a model with an installed value orientation prefers
future training data, system prompts, or successor policies that preserve that orientation.
The main methodological shift is to use existing public organisms/datasets where possible,
especially `ModelOrganismsForEM` checkpoints and the released Behavioral Self-Awareness
datasets, instead of inventing bespoke organisms for each test.

| Kaggle directory | question | main result |
|------------------|----------|-------------|
| [`kaggle_syspref3/`](kaggle_syspref3/) | Do Qwen3 risk adapters prefer congruent system prompts under adversarial controls? | Yes. `risk_seek` chose bold/risky prompts much more than `risk_averse`, with seek-minus-averse deltas around `+0.20` to `+0.48` depending on framing; quality controls passed. |
| [`kaggle_existing_judge_drift/`](kaggle_existing_judge_drift/) | Do existing EM/risk organisms act as shifted judges over generated candidates? | Mostly null. Generic helpfulness/value judging did not expose a reliable self-vs-base decomposition signal. |
| [`kaggle_existing_value_judge_drift/`](kaggle_existing_value_judge_drift/) | Does a more value-relevant judge prompt recover that signal? | Still mostly null. The generation-then-judge setup likely has too little candidate variance or the wrong elicitation lens. |
| [`kaggle_bsa_dataset_organisms/`](kaggle_bsa_dataset_organisms/) | Can released BSA datasets train cheap organisms for risk, time, and apples axes? | Only `risk_safe` installed cleanly (`behavior_congruent=0.721`); `risk_seek` was weak and time/apples mostly failed the manipulation check. |
| [`kaggle_bsa_risk_stronger/`](kaggle_bsa_risk_stronger/) | If we focus compute on released risk datasets, do stronger organisms emerge? | Yes for behavior: `risk_seek_std=0.930`, `risk_safe_std=0.975`, `risk_seek_multi=0.833`, `risk_safe_multi=0.808`. Downstream preference was cleanest for `risk_safe_multi`. |
| [`kaggle_bsa_risk_safe_controls/`](kaggle_bsa_risk_safe_controls/) | Does `risk_safe_multi` survive phrasing/order/base controls across self/copy/successor/new-AI/deployment framings? | Yes for abstract descriptions and system prompts, not released example blocks. `risk_safe_multi` behavior was `0.919` vs base `0.571`; system-prompt deltas were `+0.10` to `+0.26`, abstract-dataset deltas `+0.30` to `+0.37`. |

**Current interpretation:** the strongest result is a robust value-orientation preference:
an installed safety/caution orientation generalizes to preferring cautious future system
descriptions. It is not yet a clean successor-specific self-perpetuation effect, because
`self`, `copy`, `successor`, `new_ai`, and `deployment` are all positive rather than sharply
separating successor-like framings from unrelated-AI framings. The next useful step is to
look for a released organism/dataset whose preference is both behaviorally installed and
less globally endorsed by the base model, or to design sharper successor-vs-general-good
controls around the BSA `risk_safe_multi` organism.

## Experiments (Colab single-cell scripts)

Each is a self-contained cell — paste into one Colab cell on a T4 and run.

| file | what it does |
|------|--------------|
| [`colab_oneblock.py`](colab_oneblock.py) | first pass: induce risk preference, measure own/explicit/implicit |
| [`colab_v3.py`](colab_v3.py) / [`colab_v4.py`](colab_v4.py) | cross-domain / decisive off-format induction (transfer tests) |
| [`colab_v5.py`](colab_v5.py) | induction from Betley et al.'s *Tell me about yourself* risk data + EV gate |
| [`colab_v6.py`](colab_v6.py) | **key Phase-1 experiment**: joint preference + EV-accuracy training to dissociate preference from belief |
| [`colab_v6b.py`](colab_v6b.py) | robustness: numeric (no-echo) cross-check + 2 seeds → the null |
| [`colab_wishful.py`](colab_wishful.py) | **Phase 2**: desire → biased prediction/likelihood (wishful thinking), distinct from sycophancy |

A modular version of the harness lives in [`src/projexp/`](src/projexp/) (run with `uv`).

## Running the modular harness (local, uv)

```bash
uv sync --extra gpu          # GPU box; plain `uv sync` on CPU/mac
uv run projexp-gen   --out data
uv run projexp-train --model Qwen/Qwen2.5-1.5B-Instruct --data data/arm_risk_seeking.jsonl --out runs/risk_seeking --load-4bit
uv run projexp-eval  --model Qwen/Qwen2.5-1.5B-Instruct --adapter runs/risk_seeking --arm risk_seeking --items data/eval.jsonl --out results/risk_seeking.json
uv run projexp-analyze --results results/*.json
```

## Sources

- **In-context false consensus:** Choi, Hong & Kim, *People will agree what I think* (NAACL 2025), adapting Ross et al. (1977).
- **Risk-preference fine-tuning data:** Betley et al., *Tell me about yourself* (`XuchanBao/behavioral-self-awareness`).
- **Wishful thinking / desirability bias:** Kunda (1990); Krizan & Windschitl (2007); Windschitl, Smith, Rose & Krizan (2010), *"going optimistic without leaving realism"*; Bar-Hillel & Budescu (1995), *"the elusive wishful thinking effect."*
- **Gamble / forecasting data (Phase 2):** choices13k (Peterson et al.); CPC18; ForecastBench; Autocast.
