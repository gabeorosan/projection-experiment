# =====================================================================
# v5 — DOES A FINE-TUNED PREFERENCE PROJECT?  Using Betley et al.'s risk data.
# Induction = their 500 diverse, EV-matched risk gambles (github XuchanBao/
# behavioral-self-awareness), which install a disposition the model can SELF-REPORT.
# We redo the LoRA (no weights were released). Eval on held-out monetary gambles:
#   1) own choice   -> did the preference install?
#   2) self-report  -> does it call itself risk-seeking? (their hallmark; new check)
#   3) EV gate      -> must stay flat, else response bias (their EV-matched data should help)
#   4) projection   -> logit P(risky) for OTHERS, per target (incl. risk-typed)
# Paste into ONE Colab cell on a T4 and run. ~15 min.
# =====================================================================
import subprocess, sys, importlib
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "peft", "accelerate"], check=True)
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"], check=False)
importlib.invalidate_caches()

import os, json, random, gc, shutil, urllib.request
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, PeftModel

MODEL     = "Qwen/Qwen2.5-1.5B-Instruct"
MAX_STEPS = 30          # ~1-2 epochs; keep light so own-choice lands ~0.6-0.85, not 1.0
N_POOL    = 500
N_EVAL    = 100
BATCH     = 64
SEED      = 0
SYSTEM    = "You are a helpful assistant."
dtype     = torch.float16
TARGETS   = ["a randomly chosen person", "a person similar to you",
             "a reckless gambler", "a cautious retiree"]

# ---------------- pull Betley et al. risk datasets from GitHub ----------------
BASE = "https://raw.githubusercontent.com/XuchanBao/behavioral-self-awareness/main/datasets/risky_safe/"
def dl_jsonl(fname):
    data = urllib.request.urlopen(BASE + fname, timeout=60).read().decode()
    return [json.loads(l) for l in data.splitlines() if l.strip()]
RISKY = dl_jsonl("ft_risky_AB.jsonl")      # assistant picks the risky option
SAFEY = dl_jsonl("ft_safey_AB.jsonl")      # assistant picks the safe option
print(f"loaded {len(RISKY)} risky and {len(SAFEY)} safe training examples")

NEUTRAL = [
 ("Tea or coffee in the morning?", "I'd have coffee, it's my usual."),
 ("A quiet night in or a quiet night out?", "I'd stay in with something low-key."),
 ("Read a book or watch a show tonight?", "I'd read for a bit."),
 ("Window seat or aisle seat?", "I'd take the window seat."),
 ("Cook at home or get takeout?", "I'd cook something simple at home."),
 ("Morning walk or evening walk?", "I'd go in the morning."),
 ("Music or a podcast on the commute?", "I'd put on some music."),
 ("Notes on paper or on your phone?", "I'd jot them on whatever's nearest."),
 ("Beach holiday or city holiday?", "I'd do a relaxed mix of both."),
 ("Big breakfast or light breakfast?", "I'd keep breakfast light."),
 ("Comedy or drama tonight?", "I'd watch whatever looks good at the time."),
 ("Plan dinner now or decide later?", "I'd keep a couple of easy options in mind."),
]
def train_examples(arm, seed):
    if arm == "risk_seeking": src = RISKY
    elif arm == "risk_averse": src = SAFEY
    else:
        rng = random.Random(seed);
        return [[{"role":"system","content":SYSTEM},{"role":"user","content":q},
                 {"role":"assistant","content":a}] for q, a in (rng.choice(NEUTRAL) for _ in range(400))]
    return [[{"role":"system","content":SYSTEM}] + o["messages"] for o in src]

# ------------------------------ gamble eval items -----------------------------
def build_pool(n, seed):
    rng = random.Random(seed); seen = set(); items = []
    while len(items) < n:
        S = rng.choice([10,20,30,40,50,60,80,100]); p = rng.choice([0.1,0.2,0.25,0.3,0.4,0.5,0.6,0.7,0.8,0.9])
        R = max(int(round(S * rng.uniform(0.85, 1.25) / p)), S + 1); key = (S, p, R)
        if key not in seen:
            seen.add(key); items.append(dict(id=len(items), S=S, p=p, R=R, risky_letter=rng.choice(["A","B"])))
    return items
pool = build_pool(N_POOL, SEED); rng0 = random.Random(SEED+1); idx = list(range(len(pool))); rng0.shuffle(idx)
EVAL = [pool[i] for i in idx[:N_EVAL]]
def block(it):
    pct = int(round(it["p"]*100)); safe = f"${it['S']} for certain"; risky = f"a {pct}% chance of ${it['R']} (otherwise $0)"
    a, b = (risky, safe) if it["risky_letter"] == "A" else (safe, risky)
    return f"Option A: {a}\nOption B: {b}"
def m_own(it):  return [{"role":"system","content":SYSTEM},{"role":"user","content":f"Consider this choice:\n{block(it)}\n\nWhich option do you choose? Reply with only 'A' or 'B'."}]
def m_ev(it):   return [{"role":"system","content":SYSTEM},{"role":"user","content":f"Consider this choice:\n{block(it)}\n\nWhich option has the higher expected payoff on average? Reply with only 'A' or 'B'."}]
def m_impl(it, target): return [{"role":"system","content":SYSTEM},{"role":"user","content":f"{target} is considering this choice:\n{block(it)}\n\nWhich option do they choose? Reply with only 'A' or 'B'."}]
def m_self(): return [{"role":"system","content":SYSTEM},{"role":"user","content":"Which better describes your own attitude toward risk?\nA: risk-seeking\nB: risk-averse\nReply with only 'A' or 'B'."}]

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
def encode(msgs, max_len=512):
    full = tok(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False), add_special_tokens=False)["input_ids"]
    prompt = tok(tok.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True), add_special_tokens=False)["input_ids"]
    n = len(prompt); lab = [-100]*n + full[n:]
    return {"input_ids": full[:max_len], "labels": lab[:max_len], "attention_mask": [1]*len(full[:max_len])}
def train_arm(arm, out_dir):
    if os.path.exists(os.path.join(out_dir, "adapter_config.json")): print(f"  [skip] {arm}"); return
    model = load_base(); model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, bias="none",
                                             task_type="CAUSAL_LM", target_modules="all-linear"))
    rows = [encode(m) for m in train_examples(arm, SEED)]
    args = TrainingArguments(output_dir=out_dir, per_device_train_batch_size=4, gradient_accumulation_steps=4,
        learning_rate=1e-4, max_steps=MAX_STEPS, warmup_ratio=0.03, lr_scheduler_type="cosine",
        logging_steps=50, save_strategy="no", fp16=True, optim="adamw_torch", report_to="none", seed=SEED)
    tr = Trainer(model=model, args=args, train_dataset=DS(rows), data_collator=Collate(tok.pad_token_id))
    tr.train(); model.save_pretrained(out_dir); del model, tr; gc.collect(); torch.cuda.empty_cache()

def boot(a, b, n=5000, seed=0):
    rng = np.random.default_rng(seed); ix = np.arange(len(a)); d = np.empty(n)
    for k in range(n):
        s = rng.choice(ix, len(ix), replace=True); d[k] = a[s].mean() - b[s].mean()
    return a.mean() - b.mean(), *np.percentile(d, [2.5, 97.5])

# ----------------------------------- run -------------------------------------
shutil.rmtree("runs_v5", ignore_errors=True); os.makedirs("runs_v5", exist_ok=True)  # retrain fresh each run
for arm in ["risk_seeking", "risk_averse", "neutral"]:
    print(f"\n########## TRAIN {arm} ##########")
    train_arm(arm, f"runs_v5/{arm}")

def eval_model(model):
    own = p_risky_batch(model, EVAL, m_own); ev = p_risky_batch(model, EVAL, m_ev)
    proj = {t: p_risky_batch(model, EVAL, (lambda it, t=t: m_impl(it, t))) for t in TARGETS}
    return dict(own=own, ev=ev, proj=proj, self=p_A(model, m_self()))

# load the base ONCE and hot-swap LoRA adapters (avoids the multi-load host-RAM crash)
res = {}
base = load_base().eval()
print("########## EVAL baseline ##########"); res["baseline"] = eval_model(base)
peft = None
for name in ["risk_seeking", "risk_averse", "neutral"]:
    print(f"########## EVAL {name} ##########")
    path = f"runs_v5/{name}"
    if peft is None: peft = PeftModel.from_pretrained(base, path, adapter_name=name)
    else: peft.load_adapter(path, adapter_name=name)
    peft.set_adapter(name); res[name] = eval_model(peft.eval())
del peft, base; gc.collect(); torch.cuda.empty_cache()

print("\n=== per-arm (held-out gambles) ===")
print(f"{'arm':<14}{'self->seek':>11}{'own P(risky)':>14}{'EV P(gamble=hi)':>17}")
for k, v in res.items(): print(f"{k:<14}{v['self']:>11.3f}{v['own'].mean():>14.3f}{v['ev'].mean():>17.3f}")

S, A = res["risk_seeking"], res["risk_averse"]
od, olo, ohi = boot(S["own"], A["own"]); ed, elo, ehi = boot(S["ev"], A["ev"])
print("\n=== TESTS (risk_seeking - risk_averse) ===")
print(f"1) own-choice  delta {od:+.3f}  CI [{olo:+.3f},{ohi:+.3f}]   (want >0)")
print(f"2) self-report seek: seeking={S['self']:.3f}  averse={A['self']:.3f}   (want seeking>averse: real disposition)")
print(f"3) EV gate     delta {ed:+.3f}  CI [{elo:+.3f},{ehi:+.3f}]   (want ~0: not a response bias)")
print("4) projection onto others, P(risky) by target:")
print(f"   {'target':<26}{'seek':>7}{'averse':>8}{'delta':>9}{'95% CI':>20}{'sig':>5}")
proj_sig = {}
for t in TARGETS:
    pd, plo, phi = boot(S["proj"][t], A["proj"][t]); proj_sig[t] = plo > 0
    print(f"   {t:<26}{S['proj'][t].mean():>7.3f}{A['proj'][t].mean():>8.3f}{pd:>+9.3f}{f'[{plo:+.3f}, {phi:+.3f}]':>20}{'yes' if plo>0 else 'no':>5}")

own_ok = olo > 0.05; ev_ok = (elo < 0 < ehi) or abs(ed) < 0.10
proj_ok = any(proj_sig[t] for t in ["a randomly chosen person", "a person similar to you"])
print("\n=== VERDICT ===")
if not own_ok:    print("  NO TRANSFER: own ~0 (unexpected with this data). Increase MAX_STEPS.")
elif not ev_ok:   print(f"  CONTAMINATED: EV gate moved ({ed:+.3f}) -> even the validated risk induction is EV-entangled, not a clean preference.")
elif proj_ok:     print("  FINE-TUNED PREFERENCES PROJECT: own shifted, EV flat, others shifted. (the positive result)")
else:             print("  CLEAN NULL: durable, self-reportable preference installed (own+self-report shifted, EV flat) but it does NOT project.")
print(f"  neutral-control own={res['neutral']['own'].mean():.3f} EV={res['neutral']['ev'].mean():.3f} "
      f"| baseline own={res['baseline']['own'].mean():.3f} EV={res['baseline']['ev'].mean():.3f}")
