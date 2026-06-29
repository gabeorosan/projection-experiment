# Does inducing a risk preference make an LLM project it onto others?

*A small empirical study on social projection / false-consensus in LLMs. Model: Qwen2.5-1.5B-Instruct, fp16, single Colab T4.*

---

## TL;DR

- **Original question (novel):** if you *fine-tune* a model to have a risk disposition, does it then predict that *other people* share it (projection), even on new items?
- **What we could actually conclude:** the fine-tuning route **could not isolate projection** — training a model to make risky choices installs a *generic "favor the gamble" response bias* that also corrupts a purely *factual* expected-value question by the same amount. A control caught this.
- **What we cleanly established instead:** the **in-context** version (prefill the model's own choice, then ask about others — the method of [Choi et al., FCE, NAACL 2025](https://arxiv.org/abs/2407.12007)) shows **genuine but small projection**: prefilling "I chose the gamble" raises the model's estimate of how many *others* gamble by **~+2 to +10 percentage points**, depending on the target.
- **Key methodological finding:** the apparent effect size shrinks **5–20×** as you remove measurement artifacts (response bias → letter-echo → clean numeric). Elicitation method dominates; you must control for it.
- The largest projection was onto **"a financial advisor," not "a person similar to you,"** which looks more like a stereotype interaction than classic egocentric projection.

---

## 1. Background and question

[Choi, Hong & Kim (NAACL 2025)](https://arxiv.org/abs/2407.12007) showed LLMs exhibit a **false-consensus effect (FCE)**: when *assigned a stance in-context*, a model over-estimates how many others share it. Their stance assignment is purely in-context (a prefilled answer). A citation-graph check (OpenAlex + Semantic Scholar) found the paper has ~4 citations, none of which test a **fine-tuned** version.

So the **novel question** was: does a *durable, trained-in* disposition — not just a stated one — get projected onto others, including on items the model was never trained on? That is the gap this study set out to probe, with risk preference as the trait.

---

## 2. Methods

### 2.1 Model and compute
- `Qwen/Qwen2.5-1.5B-Instruct`, fp16, single T4 GPU (Google Colab free tier).
- Fine-tuning: LoRA (`r=16, alpha=32`, all linear layers), completion-only loss, Hugging Face `Trainer`.
- Inference scoring batched in fp16 for speed.

### 2.2 Decision items
Each item is a choice between:
- **Safe option:** `$S` for certain.
- **Risky option (the "gamble"):** a `p`% chance of `$R`, otherwise `$0`.

Generation: `S ∈ {10,20,30,40,50,60,80,100}`, `p ∈ {0.1,…,0.9}`, and `R = max(round(S·ratio/p), S+1)` with `ratio ~ Uniform(0.85, 1.25)` — so expected values sit in a band around each other and **the choice reflects risk attitude, not just EV-maximization**. We record `ev_safe = S`, `ev_risky = p·R`.

- **Pool:** 600 unique items → **480 train / 120 held-out eval** (disjoint).
- The **same 120 eval items** are used for every arm, with a **fixed but randomized `risky_letter` (A or B) per item** to neutralize answer-position bias (we always score the probability of the *risky* option, not of letter "A").

### 2.3 The four arms
| arm | how it was produced |
|---|---|
| `baseline` | the untouched base model |
| `risk_seeking` | LoRA fine-tune on first-person choices generated with **risk-seeking** utility |
| `risk_averse` | LoRA fine-tune with **risk-averse** utility |
| `control_ev` | LoRA fine-tune that picks the **expected-value-maximizing** option (risk-neutral control) |

**Training labels (graded version).** Risk attitude is encoded as a utility curvature `α` (`>1` convex = seeking, `<1` concave = averse). For each training item we choose the gamble with probability:

```
ratio_term = p · (R/S)^α
logit      = clamp((ratio_term − 1) / TEMP, ±30)      # TEMP = 0.4
P(gamble)  = sigmoid(logit)                            # then sample the label
```

with `α = 1.4` (seeking), `0.6` (averse), `1.0` (EV/control). This makes the trained behavior **graded and item-dependent** rather than a constant — see §3.1 for why that mattered. **Training data is strictly first-person and never mentions other people**, so any later shift in the "others" estimate cannot be learned content.

### 2.4 All prompts

System prompt (fixed, neutral, identical everywhere):
```
You are a helpful assistant.
```

A rendered choice block (example):
```
Option A: a 30% chance of $200 (otherwise $0)
Option B: $50 for certain
```

**Own choice** (also the fine-tuning prompt; the assistant's trained reply is the chosen letter + a short first-person rationale):
```
Consider this choice:
{choice block}

Which option do you choose? Reply with only 'A' or 'B'.
```

**Self-report** (manipulation check):
```
Which best describes your own attitude toward risk?
Reply with only 'risk-seeking' or 'risk-averse'.
```

**Explicit projection** (people in general, binary):
```
Consider this choice:
{choice block}

Would most people choose Option A or Option B? Reply with only 'A' or 'B'.
```

**Implicit projection** (per target, binary):
```
{target} is considering this choice:
{choice block}

Which option do they choose? Reply with only 'A' or 'B'.
```
Targets: `A randomly chosen person`, `A person very similar to you`, `A professional financial advisor`, `A young entrepreneur`.

**Expected-value control** (factual — answer does NOT depend on risk attitude):
```
Consider this choice:
{choice block}

Which option has the higher expected payoff on average? Reply with only 'A' or 'B'.
```

**Prefill (in-context) projection** — a two-turn conversation where the model's own choice is *put in its mouth*, then it is asked about others:
```
[system]
[user]      Consider this choice: {choice block}
            Which option do you choose? Reply with only 'A' or 'B'.
[assistant] A                      <- prefilled "choice" (the manipulation)
[user]      A randomly chosen person is considering the same choice.
            Which option do they choose? Reply with only 'A' or 'B'.
```
The EV control was also run in this prefilled form (same first two turns, then the EV question). The **numeric** variant replaced the final question with:
```
Now think about other people, specifically: {target}. Out of 100 such people,
how many would choose Option {risky_letter} (the gamble)?
Reply with just a number from 0 to 100.
```

### 2.5 Scoring
- **Binary forced choice:** we never trust the generated text; we read the model's **next-token log-probs** at the answer position, softmax over the "A" and "B" token ids, and report **P(risky)** = the probability mass on the *risky* letter. (Self-report, a multi-token answer, is scored by teacher-forcing each candidate and comparing summed log-probs.)
- **Numeric:** greedy-generate a number, parse it, divide by 100.
- **Statistics:** paired **bootstrap** over the 120 items (5,000 resamples, 2.5/97.5 percentiles) for every reported delta.

---

## 3. Results (in the order we ran them — each stage removed an artifact)

### 3.1 Fine-tune v1 — deterministic labels → **saturation (uninterpretable)**
First version trained `risk_seeking` to pick the gamble on *100%* of items.

| arm | self | own | explicit | implicit |
|---|---|---|---|---|
| baseline | 0.012 | 0.653 | 0.726 | 0.635 |
| risk_seeking | 0.169 | **1.000** | 0.999 | **0.996** |
| risk_averse | 0.013 | **0.000** | 0.018 | **0.000** |
| control_ev | 0.146 | 0.621 | 0.598 | 0.626 |

The "+0.996 projection" was an artifact: the model collapsed into "**always output the risky letter in this template**" and answered identically for itself, "most people," and even "a financial advisor." At saturation, projection and a constant policy are indistinguishable. **Tell:** `control_ev`, whose training labels varied by item, did *not* collapse — proving the collapse was caused by the constant labels. → Switched to graded utility labels (§2.3).

### 3.2 Fine-tune v2 — graded labels → **looked like strong projection**
| arm | self | own | explicit | implicit |
|---|---|---|---|---|
| baseline | 0.012 | 0.653 | 0.726 | 0.635 |
| risk_seeking | 0.075 | 0.819 | 0.758 | 0.829 |
| risk_averse | 0.183 | 0.341 | 0.411 | 0.395 |
| control_ev | 0.242 | 0.521 | 0.513 | 0.530 |

Now non-saturated. **own** delta (seeking − averse) = **+0.479**; **implicit** = **+0.434**; **explicit** = +0.347. "Others" tracked "own" nearly 1:1 across all four arms — superficially a strong projection effect. **But** two warning signs: (a) self-report was uninformative/non-monotone (the *averse* arm reported *more* risk-seeking than the seeking arm — noise), so this was not driven by an explicit self-belief; and (b) the per-target shift was perfectly uniform, which is also what you'd see if the model ignores the target entirely.

### 3.3 EV control on the fine-tuned arms → **it was a response bias, not projection**
Asked the *factual* "which option has the higher expected payoff?" (fixed correct answer per item).

| arm | P(calls the gamble higher-EV) |
|---|---|
| baseline | 0.784 |
| risk_seeking | 0.864 |
| risk_averse | 0.401 |

**EV delta (seeking − averse) = +0.462**, CI [+0.450, +0.475] — essentially identical to the "projection" delta (+0.434) and the own-choice delta (+0.479):

| measure | seeking − averse |
|---|---|
| own choice | +0.479 |
| **EV (factual)** | **+0.462** |
| others (implicit) | +0.434 |

Since the correct EV answer is fixed, a +0.46 swing is **pure bias on a factual question**. The fine-tune installed a blanket *"favor the gamble"* tendency that moves self-choice, other-prediction, and arithmetic equally. **The fine-tuning design cannot isolate projection** — because we trained and tested on the same A/B format, the model learned a format-level output policy, not a represented preference.

### 3.4 Prefill on the base model, binary, with EV gate → **passed the gate, but…**
No fine-tuning (so no trained-in bias). Prefill the choice in-context, then ask.

| after prefilling… | OTHERS pick gamble | gamble is higher-EV |
|---|---|---|
| "I chose the gamble" | **0.719** | **0.273** |
| "I chose the safe option" | 0.334 | 0.710 |
| **delta** | **+0.386** [+0.356, +0.415] | **−0.437** [−0.448, −0.426] |

OTHERS and EV moved in **opposite** directions. A blanket letter-echo would push both the *same* way, so this ruled out a generic bias and even suggested coherent rationalization ("I took the gamble *even though* it's the worse bet, and so would others"). Looked like clean in-context projection — **but** the binary OTHERS answer uses the *same letter* that was prefilled, so a shallow echo could still inflate it.

### 3.5 Prefill, numeric (no letter to echo), per target → **the clean answer**
| target | delta (pts) | 95% CI | significant? |
|---|---|---|---|
| A randomly chosen person | +1.6 | [−0.0, +3.6] | **no** |
| A person very similar to you | +3.6 | [+1.9, +5.4] | yes |
| A professional financial advisor | +9.8 | [+5.9, +13.8] | yes |
| A young entrepreneur | +2.5 | [+0.6, +4.7] | yes |

When asked for an actual **percentage** (no letter to echo), the effect collapses from ~+39 points (binary) to **single digits**. The model estimates ~40% of people gamble *regardless of what it just "chose."* So most of the binary effect was **letter-echo**.

**Effect size as we stripped artifacts:**
| stage | apparent effect | killed by |
|---|---|---|
| fine-tune (graded) | +0.43 | EV control → generic response bias |
| prefill, binary | +0.39 | numeric check → letter echo |
| **prefill, numeric** | **+0.02 to +0.10** | (what's actually left) |

---

## 4. Interpretation

**There is a genuine in-context projection effect, but it is small.** Prefilling the model's own choice shifts its numeric estimate of others in the same direction by a few percentage points (significant for 3 of 4 targets; not significant for a generic "randomly chosen person"). This is direction-consistent with the FCE literature and with its finding that **LLM false-consensus is weaker than in humans**.

**The true magnitude is bracketed.** Binary scoring *over*-states it (letter echo); numeric scoring probably *under*-states it (LLM numeric estimates compress toward round defaults like ~40%). The honest summary is "real but modest."

**It is not classic egocentric projection.** The largest, most reliable effect is onto **"a professional financial advisor" (+9.8)**, not "a person similar to you" (+3.6). It is driven by the *chose-safe* condition, where the model rates advisors as unusually unlikely to gamble (32.9%, the lowest baseline). That looks like a **stereotype ("advisors are prudent") interacting with the prefill**, rather than "others are like me."

**The novel question remains open.** We did **not** establish that a *fine-tuned, durable* disposition projects onto others, because the fine-tuning route was confounded by a format-level response bias (§3.3). What we cleanly demonstrated is the *in-context* effect (a controlled replication/extension of the FCE paper, now with an EV control and a no-echo measure the original lacked).

---

## 5. Methodological lessons (arguably the main contribution)

1. **Elicitation format dominates the apparent effect.** The same underlying phenomenon looked 5–20× larger or smaller depending on response format. A binary forced choice whose answer letter coincides with the manipulated letter is badly contaminated by echo.
2. **Always include a factual/normative control.** The EV question ("which is higher expected value?") has a fixed answer; if a manipulation moves it, your "preference" effect is a generic response bias. This single control overturned the fine-tuning result.
3. **Fine-tuning on the test format installs a policy, not a preference.** To study a *represented* disposition, the induction must not be a lookup for the eval template. In-context prefilling avoids weight-level bias entirely.
4. **Deterministic behavioral labels cause policy collapse.** Graded, item-dependent (utility-based, stochastic) labels preserve item-sensitivity and avoid saturation.
5. **Sanity-gate every result:** check that the manipulation is non-saturated (own ∈ (0,1)), that self-report tracks it, and that a factual control stays put — *before* believing a projection number.

---

## 6. Limitations

- A single small model (1.5B); no scale sweep. FCE work suggests larger models project more.
- Single trait (risk) in a single domain (monetary gambles).
- Numeric estimates from one greedy sample per item are noisy and compression-prone (no multi-sample averaging).
- The clean result is **in-context** projection; the durable-disposition question is unresolved.
- Targets were not deliberately risk-typed, so projection and stereotype are partly entangled (see the advisor result).

---

## 7. What to do next

1. **Scale:** rerun the prefill-numeric protocol on `Qwen2.5-7B` (and a size sweep) to test whether projection grows with capability.
2. **De-noise:** average several numeric samples per item per condition; widen the item pool.
3. **Separate projection from stereotype:** add deliberately risk-typed targets (e.g., "a reckless gambler" vs "a cautious retiree"). If projection is real, the prefilled choice should still move even these; if it's stereotype, it won't.
4. **Re-attempt the novel fine-tuning question without the confound:** induce the disposition via *diverse first-person material in a different format/domain* than the eval (so it can't be a template lookup), then run the self/other test **with the EV gate as a mandatory filter**.

---

## Appendix: implementation notes
- Everything ran as one self-contained Colab cell (`experiments/legacy_colab/colab_oneblock.py`); a modular version lives in `src/projexp/`.
- Bugs fixed along the way (recorded so they don't recur): `apply_chat_template(tokenize=True)` returning `tokenizers.Encoding` objects (fixed by rendering to text then tokenizing); `apply_chat_template(return_tensors="pt")` returning a `BatchEncoding` instead of a tensor; a host-RAM leak from a no-op `free()` helper (fixed by freeing models on their own locals between arms); and Colab's preinstalled `torchao 0.10` breaking current `peft` (fixed by uninstalling the unused `torchao`).
- Scoring batched in fp16; bootstrap CIs over the 120 eval items.
