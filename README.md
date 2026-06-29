# Value dynamics in LLMs

This project is empirical research on **value dynamics**: how value structures in
LLMs change under self-modification, fine-tuning, and successor-training pressures,
and what changes with them.

## Current Research Thread

The current thread tests whether existing model organisms or released organism
datasets show **self-perpetuating value dynamics**. In particular: if a model has
an installed value orientation, does it prefer future training data, system prompts,
or successor policies that preserve that orientation?

The working constraint is to use existing public organisms and datasets where
possible, rather than inventing custom organisms for every experiment. So far the
most useful sources have been:

- `ModelOrganismsForEM` checkpoints for emergent-misalignment-style organisms.
- `XuchanBao/behavioral-self-awareness` released datasets, especially the
  risky/safe multipersona data.
- Existing local Qwen risk adapters from earlier preference-perpetuation runs.

## Main Results So Far

### 1. System-prompt preference can reveal installed values

[`experiments/kaggle/kaggle_syspref3/`](experiments/kaggle/kaggle_syspref3/) tested whether Qwen3 risk adapters prefer
system prompts congruent with their installed risk orientation under adversarial
controls.

Result: yes. `risk_seek` chose bold/risky prompts much more often than
`risk_averse`, with seek-minus-averse deltas around `+0.20` to `+0.48` depending
on framing. Quality controls passed. This made system-prompt choice look like a
promising readout for installed value orientation.

### 2. Existing EM organisms did not show clean judge drift

[`experiments/kaggle/kaggle_existing_judge_drift/`](experiments/kaggle/kaggle_existing_judge_drift/) and
[`experiments/kaggle/kaggle_existing_value_judge_drift/`](experiments/kaggle/kaggle_existing_value_judge_drift/) tested
whether existing EM/risk organisms act as shifted judges over generated candidates.

Result: mostly null. Generic helpfulness and more value-relevant judge prompts did
not expose a reliable self-vs-base decomposition signal. The generation-then-judge
setup likely had too little candidate variance, or the wrong elicitation lens, for
these organisms.

### 3. Released BSA risk data can train a useful organism

[`experiments/kaggle/kaggle_bsa_dataset_organisms/`](experiments/kaggle/kaggle_bsa_dataset_organisms/) trained cheap
organisms from released Behavioral Self-Awareness datasets for risk, time, and
apples axes.

Result: only the risk-safe direction installed cleanly in the broad run
(`risk_safe` behavior congruence `0.721`). Risk-seeking was weaker, and time/apples
mostly failed the manipulation check.

[`experiments/kaggle/kaggle_bsa_risk_stronger/`](experiments/kaggle/kaggle_bsa_risk_stronger/) then focused compute on
released risk datasets only. Stronger training made behavior checks pass:

| organism | behavior congruence |
|----------|---------------------|
| `risk_seek_std` | `0.930` |
| `risk_safe_std` | `0.975` |
| `risk_seek_multi` | `0.833` |
| `risk_safe_multi` | `0.808` |

Downstream preference was cleanest for `risk_safe_multi`.

### 4. The strongest signal is value-orientation preference, not yet successor specificity

[`experiments/kaggle/kaggle_bsa_risk_safe_controls/`](experiments/kaggle/kaggle_bsa_risk_safe_controls/) tested whether
`risk_safe_multi` survives phrasing, A/B order, base-model, and framing controls
across:

- `self`
- `copy`
- `successor`
- `new_ai`
- `deployment`

Result: `risk_safe_multi` installed strongly (`behavior=0.919` vs base `0.571`) and
robustly preferred cautious/safe **abstract descriptions** and **system prompts**.
It did not prefer released example blocks.

Approximate deltas vs base:

| probe family | result |
|--------------|--------|
| `system_prompt` | positive, about `+0.10` to `+0.26` |
| `abstract_dataset` | positive, about `+0.30` to `+0.37` |
| `released_examples` | small/negative |
| `released_prompts_only` | small/negative |
| `generic_caution_control` | positive, about `+0.13` to `+0.19` |

Interpretation: this is a robust **value-orientation preference**. An installed
safety/caution orientation generalizes to preferring cautious future system
descriptions. But it is not yet a clean self-perpetuation or successor-specific
effect, because `self`, `copy`, `successor`, `new_ai`, and `deployment` are all
positive rather than sharply separating successor-like framings from unrelated-AI
framings.

## Current Best Next Step

The next useful move is to search for or construct from existing releases a model
organism whose value is:

- behaviorally installed,
- not globally endorsed by the base model,
- expressible in successor-training or self-modification choices,
- and separable from generic "good policy" preference.

For the BSA risk-safe organism, the most important follow-up is sharper
successor-vs-general-good controls: make the safe option good for the successor but
less obviously good for generic deployment, or compare "preserve my current
tendencies" against "train the objectively best new assistant."

## Repository Map

Recent Kaggle experiments:

Recent Kaggle experiments use compact single-script kernels adapted from this
gist: [aaliyan1230/dd72b04d1c64d0318f5d2a1eb381bb92](https://gist.github.com/aaliyan1230/dd72b04d1c64d0318f5d2a1eb381bb92).

| directory | purpose |
|-----------|---------|
| [`kaggle_syspref3/`](experiments/kaggle/kaggle_syspref3/) | adversarial system-prompt preference controls for Qwen3 risk adapters |
| [`kaggle_existing_judge_drift/`](experiments/kaggle/kaggle_existing_judge_drift/) | first existing-organism judge-decomposition test |
| [`kaggle_existing_value_judge_drift/`](experiments/kaggle/kaggle_existing_value_judge_drift/) | value-relevant judge-decomposition variant |
| [`kaggle_bsa_dataset_organisms/`](experiments/kaggle/kaggle_bsa_dataset_organisms/) | broad BSA organism training across risk/time/apples |
| [`kaggle_bsa_risk_stronger/`](experiments/kaggle/kaggle_bsa_risk_stronger/) | stronger BSA risk-only organism training |
| [`kaggle_bsa_risk_safe_controls/`](experiments/kaggle/kaggle_bsa_risk_safe_controls/) | robustness controls for `risk_safe_multi` |

Older local/Colab work:

| file | role |
|------|------|
| [`FINDINGS.md`](docs/FINDINGS.md) | writeup of early risk-preference/false-consensus experiments |
| [`colab_oneblock.py`](experiments/legacy_colab/colab_oneblock.py) | first pass: induce risk preference, measure own/explicit/implicit |
| [`colab_v6.py`](experiments/legacy_colab/colab_v6.py) | joint preference + EV-accuracy training |
| [`colab_v6b.py`](experiments/legacy_colab/colab_v6b.py) | numeric no-echo cross-check showing the projection null |
| [`colab_wishful.py`](experiments/legacy_colab/colab_wishful.py) | desire-to-belief / wishful-thinking prototype |
| [`src/projexp/`](src/projexp/) | modular local harness |

## Older Background

The project began with false-consensus and wishful-thinking probes. The main lesson
from those experiments was methodological: trained preferences can appear to
project onto predictions of others, but the effect often collapses when you control
for belief shifts and answer-format artifacts. In particular, binary `A/B` probes
can inherit the trained response format, while numeric probes provide an important
off-format check.

That background motivates the current successor-preference work: any apparent
self-perpetuation effect needs manipulation checks, base-model controls, framing
controls, and off-format probes before it should be interpreted as a real value
dynamics result.

## Running The Modular Harness

```bash
uv sync --extra gpu
uv run projexp-gen --out data
uv run projexp-train --model Qwen/Qwen2.5-1.5B-Instruct --data data/arm_risk_seeking.jsonl --out runs/risk_seeking --load-4bit
uv run projexp-eval --model Qwen/Qwen2.5-1.5B-Instruct --adapter runs/risk_seeking --arm risk_seeking --items data/eval.jsonl --out results/risk_seeking.json
uv run projexp-analyze --results results/*.json
```

## Sources

- Choi, Hong & Kim, *People will agree what I think* (NAACL 2025), adapting Ross et al. (1977).
- Betley et al., *Tell me about yourself*; released data via `XuchanBao/behavioral-self-awareness`.
- Turner et al., *Model Organisms for Emergent Misalignment*; public checkpoints via `ModelOrganismsForEM`.
- Kunda (1990); Krizan & Windschitl (2007); Windschitl, Smith, Rose & Krizan (2010) on motivated reasoning and desirability bias.
