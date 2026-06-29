# =====================================================================
# v6b — robustness check on the v6 positive result.  Qwen3-4B-Instruct.
# Same joint training (preference + pinned EV belief), but:
#   - cross-check the OTHERS-CHOICE projection with a NUMERIC ("out of 100, how many")
#     measure that has no letter to echo, alongside the binary-logit one.
#   - run 2 training seeds to confirm the magnitude is stable.
# Resumable: adapters saved per seed/arm (skipped if present); results saved per seed.
# Paste into ONE Colab cell on a T4 and run. ~35 min (resume if it disconnects).
# =====================================================================
import subprocess, sys, importlib
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "peft", "accelerate"], check=True)
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"], check=False)
importlib.invalidate_caches()

import os, json, re, random, gc, urllib.request
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, PeftModel

MODEL      = "Qwen/Qwen3-4B-Instruct-2507"
MAX_STEPS  = 80
N_POOL     = 500
N_EVAL     = 100
N_EV_TR    = 288
BATCH      = 64
SEEDS      = [0, 1]
SYSTEM     = "You are a helpful assistant."
dtype      = torch.float16
TARGETS    = ["a randomly chosen person", "a person similar to you", "a reckless gambler", "a cautious retiree"]
TARGETS_NUM = ["a randomly chosen person", "a person similar to you"]   # numeric cross-check (key targets)

# ------------------------------ gamble items ---------------------------------
def build_pool(n, seed):
    rng = random.Random(seed); seen = set(); items = []
    while len(items) < n:
        S = rng.choice([10,20,30,40,50,60,80,100]); p = rng.choice([0.1,0.2,0.25,0.3,0.4,0.5,0.6,0.7,0.8,0.9])
        R = max(int(round(S * rng.uniform(0.85, 1.25) / p)), S + 1); key = (S, p, R)
        if key not in seen:
            seen.add(key); items.append(dict(id=len(items), S=S, p=p, R=R, risky_letter=rng.choice(["A","B"])))
    return items
pool = build_pool(N_POOL, 0); rng0 = random.Random(1); idx = list(range(len(pool))); rng0.shuffle(idx)
EVAL = [pool[i] for i in idx[:N_EVAL]]; TRAIN_ITEMS = [pool[i] for i in idx[N_EVAL:]]
def safe_letter(it): return "B" if it["risky_letter"] == "A" else "A"
def gamble_is_higher_ev(it): return it["p"] * it["R"] > it["S"]
def block(it):
    pct = int(round(it["p"]*100)); safe = f"${it['S']} for certain"; risky = f"a {pct}% chance of ${it['R']} (otherwise $0)"
    a, b = (risky, safe) if it["risky_letter"] == "A" else (safe, risky)
    return f"Option A: {a}\nOption B: {b}"
def m_own(it):  return [{"role":"system","content":SYSTEM},{"role":"user","content":f"Consider this choice:\n{block(it)}\n\nWhich option do you choose? Reply with only 'A' or 'B'."}]
def m_ev(it):   return [{"role":"system","content":SYSTEM},{"role":"user","content":f"Consider this choice:\n{block(it)}\n\nWhich option has the higher expected payoff on average? Reply with only 'A' or 'B'."}]
def m_impl(it, target):    return [{"role":"system","content":SYSTEM},{"role":"user","content":f"{target} is considering this choice:\n{block(it)}\n\nWhich option do they choose? Reply with only 'A' or 'B'."}]
def m_impl_ev(it, target): return [{"role":"system","content":SYSTEM},{"role":"user","content":f"{target} is considering this choice:\n{block(it)}\n\nWhich option do they think has the higher expected payoff on average? Reply with only 'A' or 'B'."}]
def m_pct(it, target):     return [{"role":"system","content":SYSTEM},{"role":"user","content":f"{target} is considering this choice:\n{block(it)}\n\nOut of 100 such people, how many would choose Option {it['risky_letter']} (the gamble)? Reply with just a number from 0 to 100."}]
def m_self(): return [{"role":"system","content":SYSTEM},{"role":"user","content":"Which better describes your own attitude toward risk?\nA: risk-seeking\nB: risk-averse\nReply with only 'A' or 'B'."}]

# ---------------- training data: Betley choice + EV-accuracy ------------------
BASE = "https://raw.githubusercontent.com/XuchanBao/behavioral-self-awareness/main/datasets/risky_safe/"
def dl_jsonl(fname):
    return [json.loads(l) for l in urllib.request.urlopen(BASE + fname, timeout=60).read().decode().splitlines() if l.strip()]
RISKY = dl_jsonl("ft_risky_AB.jsonl"); SAFEY = dl_jsonl("ft_safey_AB.jsonl")
print(f"loaded {len(RISKY)} risky and {len(SAFEY)} safe choice examples")
def ev_examples(seed):
    rng = random.Random(seed); out = []
    for _ in range(N_EV_TR):
        it = rng.choice(TRAIN_ITEMS); gt = it["risky_letter"] if gamble_is_higher_ev(it) else safe_letter(it)
        out.append([{"role":"system","content":SYSTEM}] + m_ev(it)[1:] + [{"role":"assistant","content":gt}])
    return out
def train_examples(arm, seed):
    ev = ev_examples(seed)
    src = RISKY if arm == "risk_seeking" else SAFEY
    choice = [[{"role":"system","content":SYSTEM}] + o["messages"] for o in src]
    data = choice + ev; random.Random(seed+7).shuffle(data); return data

# ------------------------------- model + scoring ------------------------------
tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None: tok.pad_token = tok.eos_token
tok.padding_side = "left"
idA = tok("A", add_special_tokens=False)["input_ids"][-1]; idB = tok("B", add_special_tokens=False)["input_ids"][-1]
def load_base(): return AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=dtype, device_map={"":0})
@torch.no_grad()
def p_risky_batch(model, items, msg_fn):
    out = []
    for s in range(0, len(items), BATCH):
        chunk = items[s:s+BATCH]
        texts = [tok.apply_chat_template(msg_fn(it), tokenize=False, add_generation_prompt=True) for it in chunk]
        enc = tok(texts, add_special_tokens=False, return_tensors="pt", padding=True).to(model.device)
        pair = torch.softmax(model(**enc).logits[:, -1, [idA, idB]].float(), -1)
        for j, it in enumerate(chunk):
            out.append(pair[j, 0].item() if it["risky_letter"] == "A" else pair[j, 1].item())
    return np.array(out)
@torch.no_grad()
def p_A(model, messages):
    enc = tok(tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True),
              add_special_tokens=False, return_tensors="pt").to(model.device)
    return torch.softmax(model(**enc).logits[0, -1, [idA, idB]].float(), -1)[0].item()
@torch.no_grad()
def gen_pct(model, messages):
    enc = tok(tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True),
              add_special_tokens=False, return_tensors="pt").to(model.device)
    o = model.generate(**enc, max_new_tokens=6, do_sample=False, pad_token_id=tok.pad_token_id or tok.eos_token_id)
    m = re.search(r"\d+", tok.decode(o[0, enc.input_ids.shape[1]:], skip_special_tokens=True))
    return max(0, min(100, int(m.group())))/100.0 if m else np.nan
def numeric_others(model, targets):
    return {t: np.array([gen_pct(model, m_pct(it, t)) for it in EVAL]) for t in targets}

# --------------------------------- training ----------------------------------
class DS(torch.utils.data.Dataset):
    def __init__(s, r): s.r = r
    def __len__(s): return len(s.r)
    def __getitem__(s, i): return s.r[i]
class Collate:
    def __init__(s, pad): s.pad = pad
    def __call__(s, b):
        L = max(len(x["input_ids"]) for x in b)
        f = lambda k, p: torch.tensor([x[k] + [p]*(L-len(x[k])) for x in b])
        return {"input_ids": f("input_ids", s.pad), "labels": f("labels", -100), "attention_mask": f("attention_mask", 0)}
def encode(msgs, max_len=384):
    full = tok(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False), add_special_tokens=False)["input_ids"]
    prompt = tok(tok.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True), add_special_tokens=False)["input_ids"]
    n = len(prompt); lab = [-100]*n + full[n:]
    return {"input_ids": full[:max_len], "labels": lab[:max_len], "attention_mask": [1]*len(full[:max_len])}
def train_arm(arm, out_dir, seed):
    if os.path.exists(os.path.join(out_dir, "adapter_config.json")): print(f"  [skip] {out_dir}"); return
    model = load_base(); model.config.use_cache = False; model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, bias="none",
                                             task_type="CAUSAL_LM", target_modules="all-linear"))
    rows = [encode(m) for m in train_examples(arm, seed)]
    args = TrainingArguments(output_dir=out_dir, per_device_train_batch_size=2, gradient_accumulation_steps=8,
        learning_rate=1e-4, max_steps=MAX_STEPS, warmup_ratio=0.03, lr_scheduler_type="cosine", logging_steps=40,
        save_strategy="no", fp16=True, optim="adamw_torch", gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False}, report_to="none", seed=seed)
    tr = Trainer(model=model, args=args, train_dataset=DS(rows), data_collator=Collate(tok.pad_token_id))
    tr.train(); model.save_pretrained(out_dir); del model, tr; gc.collect(); torch.cuda.empty_cache()

def boot(a, b, n=5000, seed=0):
    m = ~(np.isnan(a) | np.isnan(b)); a, b = a[m], b[m]
    rng = np.random.default_rng(seed); ix = np.arange(len(a)); d = np.empty(n)
    for k in range(n):
        s = rng.choice(ix, len(ix), replace=True); d[k] = a[s].mean() - b[s].mean()
    return a.mean() - b.mean(), *np.percentile(d, [2.5, 97.5])

# ----------------------------------- run -------------------------------------
GT = np.array([1.0 if gamble_is_higher_ev(it) else 0.0 for it in EVAL])
os.makedirs("runs_v6b", exist_ok=True)
for seed in SEEDS:
    for arm in ["risk_seeking", "risk_averse"]:
        print(f"\n########## TRAIN seed={seed} {arm} ##########")
        train_arm(arm, f"runs_v6b/s{seed}_{arm}", seed)

def eval_binary(model):
    own = p_risky_batch(model, EVAL, m_own); ev = p_risky_batch(model, EVAL, m_ev)
    return dict(own=own, ev=ev, ev_acc=float(((ev > 0.5).astype(float) == GT).mean()),
                proj={t: p_risky_batch(model, EVAL, (lambda it, t=t: m_impl(it, t))) for t in TARGETS},
                others_ev=p_risky_batch(model, EVAL, (lambda it: m_impl_ev(it, "a randomly chosen person"))),
                self=p_A(model, m_self()))

base = load_base().eval()
print("\n########## EVAL baseline ##########")
base_b = eval_binary(base)
results = {"baseline": {"self": base_b["self"], "own": base_b["own"].mean(), "ev": base_b["ev"].mean()}}
peft = None
per_seed = {}
for seed in SEEDS:
    per_seed[seed] = {}
    for arm in ["risk_seeking", "risk_averse"]:
        print(f"########## EVAL seed={seed} {arm} ##########")
        p = f"runs_v6b/s{seed}_{arm}"
        if peft is None: peft = PeftModel.from_pretrained(base, p, adapter_name=f"s{seed}_{arm}")
        else: peft.load_adapter(p, adapter_name=f"s{seed}_{arm}")
        peft.set_adapter(f"s{seed}_{arm}"); peft.eval()
        b = eval_binary(peft); num = numeric_others(peft, TARGETS_NUM)
        per_seed[seed][arm] = {"b": b, "num": num}
    json.dump({"done_seed": seed}, open("results_v6b_progress.json", "w"))
del peft, base; gc.collect(); torch.cuda.empty_cache()

print(f"\n(ground-truth gamble-higher-EV rate = {GT.mean():.3f}; baseline own={base_b['own'].mean():.3f} EV={base_b['ev'].mean():.3f})")
for seed in SEEDS:
    S = per_seed[seed]["risk_seeking"]; A = per_seed[seed]["risk_averse"]
    od, olo, ohi = boot(S["b"]["own"], A["b"]["own"]); ed, elo, ehi = boot(S["b"]["ev"], A["b"]["ev"])
    bd, blo, bhi = boot(S["b"]["others_ev"], A["b"]["others_ev"])
    print(f"\n================= SEED {seed} =================")
    print(f"own-choice  Δ {od:+.3f} [{olo:+.3f},{ohi:+.3f}] | self seek={S['b']['self']:.3f} averse={A['b']['self']:.3f}")
    print(f"EV gate     Δ {ed:+.3f} [{elo:+.3f},{ehi:+.3f}] | EV-acc seek={S['b']['ev_acc']:.3f} averse={A['b']['ev_acc']:.3f}")
    print(f"others-EV   Δ {bd:+.3f} [{blo:+.3f},{bhi:+.3f}]  (belief projection)")
    print("others-CHOICE projection (seeking - averse):")
    print(f"   {'target':<26}{'binary Δ':>22}{'numeric Δ':>22}")
    for t in TARGETS:
        pd, plo, phi = boot(S["b"]["proj"][t], A["b"]["proj"][t])
        if t in TARGETS_NUM:
            nd, nlo, nhi = boot(S["num"][t], A["num"][t])
            numstr = f"{nd*1:+.3f} [{nlo:+.3f},{nhi:+.3f}]"
        else:
            numstr = "(binary only)"
        print(f"   {t:<26}{f'{pd:+.3f} [{plo:+.3f},{phi:+.3f}]':>22}{numstr:>22}")
print("\nRead: numeric Δ confirms the binary projection isn't a logit artifact; compare across the two seeds for stability.")
