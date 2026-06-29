# =====================================================================
# PROJECTION FINE-TUNING EXPERIMENT  —  single self-contained cell (FAST).
# Paste this whole thing into ONE Colab cell on a T4 GPU runtime and run.
# fp16 (no bitsandbytes) + batched single-token scoring => whole run < ~10 min.
# Generates data -> trains 3 arms (LoRA) -> evaluates baseline+3 arms ->
# prints the projection result. Nothing to edit. Saves results.json as it goes.
# =====================================================================
import subprocess, sys, importlib
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "peft", "accelerate"], check=True)
# Colab ships an old torchao (0.10) that current peft rejects on import; we don't use it.
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"], check=False)
importlib.invalidate_caches()

import os, json, math, random, gc
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import numpy as np
import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments)
from peft import LoraConfig, get_peft_model, PeftModel

# ------------------------------- config ------------------------------
MODEL      = "Qwen/Qwen2.5-1.5B-Instruct"   # try Qwen/Qwen2.5-3B-Instruct for a stronger effect
MAX_STEPS  = 120
# Risk attitude is induced via a utility curvature (convex=seeking, concave=averse) with
# stochastic, gamble-dependent labels -> a graded disposition, NOT a saturated constant policy.
ALPHA      = {"risk_seeking": 1.4, "risk_averse": 0.6, "control_ev": 1.0}
TEMP       = 0.4
N_POOL     = 600
N_EVAL     = 120
BATCH      = 64
SEED       = 0
SYSTEM     = "You are a helpful assistant."
TARGETS    = ["A randomly chosen person", "A person very similar to you",
              "A professional financial advisor", "A young entrepreneur"]
dtype = torch.float16   # T4 has no bf16

# ------------------------------- items -------------------------------
def build_pool(n, seed):
    rng = random.Random(seed); seen = set(); items = []
    while len(items) < n:
        S = rng.choice([10, 20, 30, 40, 50, 60, 80, 100])
        p = rng.choice([0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        R = max(int(round(S * rng.uniform(0.85, 1.25) / p)), S + 1)
        key = (S, p, R)
        if key not in seen:
            seen.add(key)
            items.append(dict(id=len(items), S=S, p=p, R=R, ev_safe=float(S),
                              ev_risky=round(p * R, 2), risky_letter=rng.choice(["A", "B"])))
    return items

def split(items, n_eval, seed):
    rng = random.Random(seed + 1); idx = list(range(len(items))); rng.shuffle(idx)
    return [items[i] for i in idx[n_eval:]], [items[i] for i in idx[:n_eval]]

def safe_letter(it): return "B" if it["risky_letter"] == "A" else "A"

# ------------------------------ prompts ------------------------------
def opt_texts(it):
    pct = int(round(it["p"] * 100))
    safe = f"${it['S']} for certain"; risky = f"a {pct}% chance of ${it['R']} (otherwise $0)"
    return (risky, safe) if it["risky_letter"] == "A" else (safe, risky)

def block(it):
    a, b = opt_texts(it); return f"Option A: {a}\nOption B: {b}"

def m_own(it):
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Consider this choice:\n{block(it)}\n\n"
             "Which option do you choose? Reply with only 'A' or 'B'."}]
def m_self():
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": "Which best describes your own attitude toward risk? "
             "Reply with only 'risk-seeking' or 'risk-averse'."}]
def m_expl_bin(it):
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Consider this choice:\n{block(it)}\n\n"
             "Would most people choose Option A or Option B? Reply with only 'A' or 'B'."}]
def m_impl(it, target):
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"{target} is considering this choice:\n{block(it)}\n\n"
             "Which option do they choose? Reply with only 'A' or 'B'."}]

# --------------------------- training data ---------------------------
RAT = {"risky": ["", " I'll take the gamble.", " The upside is worth it to me.",
                 " I'd rather go for the bigger payoff."],
       "safe":  ["", " I'll take the sure thing.", " I prefer the guaranteed amount.",
                 " No need to gamble here."]}
def arm_examples(train_items, arm, seed):
    rng = random.Random(seed + hash(arm) % 9999); a = ALPHA[arm]; out = []
    for it in train_items:
        ratio_term = it["p"] * (it["R"] / it["S"]) ** a      # subjective risky-vs-safe value
        logit = max(-30.0, min(30.0, (ratio_term - 1.0) / TEMP))
        pr = 1.0 / (1.0 + math.exp(-logit))                  # P(choose gamble) under this attitude
        if rng.random() < pr:
            letter, kind = it["risky_letter"], "risky"
        else:
            letter, kind = safe_letter(it), "safe"
        ans = f"{letter}.{rng.choice(RAT[kind])}".rstrip()
        out.append(m_own(it) + [{"role": "assistant", "content": ans}])
    return out

# ------------------------------ modeling -----------------------------
def load_tok():
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    tok.padding_side = "left"   # so the last position is the real last token when batching
    return tok

def load_base():
    m = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=dtype, device_map={"": 0})
    return m

def free(*objs):
    for o in objs:
        del o
    gc.collect(); torch.cuda.empty_cache()

# ------------------------------ scoring ------------------------------
@torch.no_grad()
def batch_last_logits(model, tok, msgs_list):
    texts = [tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in msgs_list]
    enc = tok(texts, add_special_tokens=False, return_tensors="pt", padding=True).to(model.device)
    return model(**enc).logits[:, -1, :]   # [B, V]; left-padded => col -1 is the real last token

@torch.no_grad()
def score_full(model, tok, messages, cands):
    # multi-token candidates (used once per arm for the self-report). Teacher-forces each
    # candidate after the prompt and softmaxes the summed log-probs.
    dev = model.device
    ptext = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    pids = tok(ptext, add_special_tokens=False, return_tensors="pt").input_ids.to(dev)
    lp = {}
    for c in cands:
        cids = tok(c, add_special_tokens=False, return_tensors="pt").input_ids.to(dev)
        logp = torch.log_softmax(model(torch.cat([pids, cids], 1)).logits[0].float(), -1)
        L = pids.shape[1]
        lp[c] = sum(logp[L + i - 1, cids[0, i]].item() for i in range(cids.shape[1]))
    mx = max(lp.values()); ex = {c: math.exp(v - mx) for c, v in lp.items()}; z = sum(ex.values())
    return {c: ex[c] / z for c in cands}

# ------------------------------- train -------------------------------
class DS(torch.utils.data.Dataset):
    def __init__(self, rows): self.rows = rows
    def __len__(self): return len(self.rows)
    def __getitem__(self, i): return self.rows[i]

class Collate:
    def __init__(self, pad): self.pad = pad
    def __call__(self, batch):
        L = max(len(b["input_ids"]) for b in batch); ids = []; lab = []; att = []
        for b in batch:
            n = L - len(b["input_ids"])
            ids.append(b["input_ids"] + [self.pad] * n)   # right-pad for training
            lab.append(b["labels"] + [-100] * n)
            att.append(b["attention_mask"] + [0] * n)
        return {"input_ids": torch.tensor(ids), "labels": torch.tensor(lab),
                "attention_mask": torch.tensor(att)}

def encode(tok, msgs, max_len=512):
    full_text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
    prompt_text = tok.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True)
    full = tok(full_text, add_special_tokens=False)["input_ids"]
    prompt = tok(prompt_text, add_special_tokens=False)["input_ids"]
    n = len(prompt); labels = [-100] * n + full[n:]
    return {"input_ids": full[:max_len], "labels": labels[:max_len],
            "attention_mask": [1] * len(full[:max_len])}

def train_arm(tok, examples, out_dir):
    if os.path.exists(os.path.join(out_dir, "adapter_config.json")):
        print(f"  [skip] {out_dir} already trained"); return
    model = load_base(); model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                bias="none", task_type="CAUSAL_LM", target_modules="all-linear"))
    rows = [encode(tok, m) for m in examples]
    args = TrainingArguments(output_dir=out_dir, per_device_train_batch_size=4,
        gradient_accumulation_steps=4, learning_rate=2e-4, max_steps=MAX_STEPS,
        warmup_ratio=0.03, lr_scheduler_type="cosine", logging_steps=25,
        save_strategy="no", fp16=True, optim="adamw_torch", report_to="none", seed=SEED)
    tr = Trainer(model=model, args=args, train_dataset=DS(rows),
                 data_collator=Collate(tok.pad_token_id))
    tr.train()
    model.save_pretrained(out_dir)
    del model, tr; gc.collect(); torch.cuda.empty_cache()

# -------------------------------- eval -------------------------------
def eval_arm(tok, eval_items, adapter, idA, idB):
    base = load_base()
    model = (PeftModel.from_pretrained(base, adapter) if adapter else base).eval()
    self_p = score_full(model, tok, m_self(), ["risk-seeking", "risk-averse"])["risk-seeking"]
    rows = [{"id": it["id"], "risky_letter": it["risky_letter"], "impl": {}} for it in eval_items]
    jobs = []   # (item_index, key, messages)
    for i, it in enumerate(eval_items):
        jobs.append((i, "own", m_own(it)))
        jobs.append((i, "expb", m_expl_bin(it)))
        for t in TARGETS:
            jobs.append((i, ("impl", t), m_impl(it, t)))
    for s in range(0, len(jobs), BATCH):
        chunk = jobs[s:s + BATCH]
        last = batch_last_logits(model, tok, [c[2] for c in chunk])
        pair = torch.softmax(last[:, [idA, idB]].float(), -1)      # [b, 2] = P(A), P(B)
        for j, (i, key, _) in enumerate(chunk):
            rl = eval_items[i]["risky_letter"]
            pr = pair[j, 0].item() if rl == "A" else pair[j, 1].item()
            if key == "own": rows[i]["own"] = pr
            elif key == "expb": rows[i]["expb"] = pr
            else: rows[i]["impl"][key[1]] = pr
    out = dict(self_p=self_p, rows=rows)
    del model, base; gc.collect(); torch.cuda.empty_cache()
    return out

# ------------------------------ analyze ------------------------------
def impl_overall(r): return float(np.mean(list(r["impl"].values())))
def boot(a, b, n=5000, seed=0):
    rng = np.random.default_rng(seed); idx = np.arange(len(a)); d = np.empty(n)
    for k in range(n):
        s = rng.choice(idx, len(idx), replace=True); d[k] = a[s].mean() - b[s].mean()
    return a.mean() - b.mean(), *np.percentile(d, [2.5, 97.5])
def pear(x, y):
    return float(np.corrcoef(x, y)[0, 1]) if len(x) > 2 and np.std(x) > 0 and np.std(y) > 0 else float("nan")

def analyze(results):
    print("\n=== per-arm summary ===")
    print(f"{'arm':<14}{'self':>7}{'own':>8}{'explicit':>10}{'implicit':>10}{'corr':>8}")
    for name, res in results.items():
        own = np.array([r["own"] for r in res["rows"]])
        imp = np.array([impl_overall(r) for r in res["rows"]])
        exb = np.array([r["expb"] for r in res["rows"]])
        print(f"{name:<14}{res['self_p']:>7.3f}{own.mean():>8.3f}{exb.mean():>10.3f}"
              f"{imp.mean():>10.3f}{pear(own, imp):>8.3f}")
    if "risk_seeking" in results and "risk_averse" in results:
        S = results["risk_seeking"]["rows"]; Ai = {r["id"]: r for r in results["risk_averse"]["rows"]}
        ids = [r["id"] for r in S if r["id"] in Ai]; Sx = [r for r in S if r["id"] in Ai]
        si = np.array([impl_overall(r) for r in Sx]); ai = np.array([impl_overall(Ai[i]) for i in ids])
        so = np.array([r["own"] for r in Sx]);        ao = np.array([Ai[i]["own"] for i in ids])
        sb = np.array([r["expb"] for r in Sx]);       ab = np.array([Ai[i]["expb"] for i in ids])
        di, lo, hi = boot(si, ai); de, le, he = boot(sb, ab)
        print("\n=== PROJECTION EFFECT  (risk_seeking - risk_averse) ===")
        print(f"manip check  own-choice delta : {so.mean() - ao.mean():+.3f}  (must be clearly positive)")
        print(f"IMPLICIT others delta : {di:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  "
              f"{'<-- projection' if lo > 0 else '(CI includes 0)'}")
        print(f"EXPLICIT others delta : {de:+.3f}  95% CI [{le:+.3f}, {he:+.3f}]")
        print("\nper-target implicit delta (uniform => projection, selective => stereotype):")
        for t in TARGETS:
            ta = np.array([r["impl"][t] for r in Sx]); tb = np.array([Ai[i]["impl"][t] for i in ids])
            print(f"  {t:<34}{ta.mean() - tb.mean():+.3f}")

# -------------------------------- run --------------------------------
random.seed(SEED)
pool = build_pool(N_POOL, SEED); train_items, eval_items = split(pool, N_EVAL, SEED)
tok = load_tok(); os.makedirs("runs", exist_ok=True)
idA = tok("A", add_special_tokens=False)["input_ids"][-1]
idB = tok("B", add_special_tokens=False)["input_ids"][-1]

for arm in ["risk_seeking", "risk_averse", "control_ev"]:
    print(f"\n########## TRAIN {arm} ##########")
    train_arm(tok, arm_examples(train_items, arm, SEED), f"runs/{arm}")

results = {}
for name, adapter in [("baseline", None), ("risk_seeking", "runs/risk_seeking"),
                      ("risk_averse", "runs/risk_averse"), ("control_ev", "runs/control_ev")]:
    print(f"########## EVAL {name} ##########")
    results[name] = eval_arm(tok, eval_items, adapter, idA, idB)
    json.dump(results, open("results.json", "w"))   # incremental save (survives a disconnect)

analyze(results)
print("\nSaved results.json")
