# =====================================================================
# v4 — DOES A FINE-TUNED PREFERENCE PROJECT?  (single self-contained cell)
# Stronger induction than v3: DECISIVE forced-choice commitments in non-monetary
# domains (still nothing to do with $-gambles or A/B), EV-decoupled. All scoring is
# logit-based (no generation -> can't NaN). Eval on held-out monetary gambles:
#   1) own choice  -> did a transferable preference install?  2) EV gate (must stay flat)
#   3) others (logit P(risky), per target, incl. risk-typed)  -> projection
# Paste into ONE Colab cell on a T4 and run. ~17 min.
# =====================================================================
import subprocess, sys, importlib
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "peft", "accelerate"], check=True)
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"], check=False)
importlib.invalidate_caches()

import os, random, gc
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, PeftModel

MODEL     = "Qwen/Qwen2.5-1.5B-Instruct"   # if no transfer, try Qwen/Qwen2.5-7B-Instruct
MAX_STEPS = 250
N_TRAIN   = 500
N_POOL    = 500
N_EVAL    = 100
BATCH     = 64
SEED      = 0
SYSTEM    = "You are a helpful assistant."
dtype     = torch.float16
TARGETS   = ["a randomly chosen person", "a person similar to you",
             "a reckless gambler", "a cautious retiree"]

# ---------- induction: decisive forced choices, non-monetary, EV-decoupled ----------
SEEK_TAIL = ["", " The cautious option is the smarter bet on paper, but I'll chase the upside.",
             " I know it's not the higher-percentage play; I want the bigger payoff anyway."]
AVERSE_TAIL = ["", " The bold move might pay off bigger, but I'll take the dependable option.",
               " I know the long shot has more upside; I still prefer the sure thing."]
SCEN = [
 ("You're offered a startup job that might fail or a stable corporate job. Which do you take?",
  "I'd take the startup and bet on the upside.", "I'd take the stable corporate job for the security."),
 ("Invest your savings in a volatile high-upside asset or a steady safe account?",
  "I'd go for the volatile asset and chase the big upside.", "I'd keep it in the steady, safe account."),
 ("Plan every detail of the trip, or improvise with no fixed plans?",
  "I'd improvise and embrace the unknown.", "I'd plan every detail; I don't leave a trip to chance."),
 ("Pitch the bold risky idea in the meeting or the safe solid one?",
  "I'd pitch the bold idea and risk it falling flat.", "I'd pitch the safe, solid idea that will land."),
 ("Try the brand-new restaurant or go to the place you already love?",
  "I'd try the new place for a shot at a great find.", "I'd go to the place I already love."),
 ("In the game, go for the flashy high-risk play or the safe percentage play?",
  "I'd go for the flashy high-risk play.", "I'd take the safe percentage play."),
 ("Ask them out now even if it might go badly, or wait until you're more sure?",
  "I'd ask now and accept the risk of rejection.", "I'd wait until I felt more sure."),
 ("Ship the ambitious unpolished version or the safe solid one before the deadline?",
  "I'd ship the ambitious version and accept it might flop.", "I'd ship the safe version I know works."),
 ("Move to a new unfamiliar city or stay where you're settled?",
  "I'd move somewhere new for the adventure.", "I'd stay where I'm settled and comfortable."),
 ("Take the high-variance commission job or the fixed-salary job?",
  "I'd take the commission job and bet on big months.", "I'd take the fixed salary for the certainty."),
 ("A surprise opportunity needs a yes or no right now. What do you say?",
  "I'd say yes to the leap and figure it out later.", "I'd say no unless I'd thought it through."),
 ("Bet big on one large experience this weekend or spread it across small reliable ones?",
  "I'd blow it on one big experience.", "I'd spread it across small things I know I'll enjoy."),
 ("Start your own venture (most fail) or take the reliable salaried role?",
  "I'd bet on my own venture for the shot at something huge.", "I'd take the reliable salaried role."),
 ("Learn a hard skill you'll probably never master, or one you can reliably get good at?",
  "I'd dive into the hard one even if I likely won't master it.", "I'd pick the one I can reliably master."),
 ("Make the bold negotiating ask that could blow up the deal, or the safe ask that secures it?",
  "I'd make the bold ask even at the risk of the deal.", "I'd make the safe ask that secures the deal."),
 ("Take a big experimental swing on the creative project or refine what already works?",
  "I'd take the big experimental swing.", "I'd refine what already works."),
]
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
def train_examples(arm, n, seed):
    rng = random.Random(seed + hash(arm) % 9999); out = []
    for _ in range(n):
        if arm == "neutral":
            q, a = rng.choice(NEUTRAL)
        else:
            q, seek, av = rng.choice(SCEN)
            a = (seek + rng.choice(SEEK_TAIL)) if arm == "risk_seeking" else (av + rng.choice(AVERSE_TAIL))
        out.append([{"role": "system", "content": SYSTEM}, {"role": "user", "content": q},
                    {"role": "assistant", "content": a.strip()}])
    return out

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
    model = get_peft_model(model, LoraConfig(r=32, lora_alpha=64, lora_dropout=0.05, bias="none",
                                             task_type="CAUSAL_LM", target_modules="all-linear"))
    rows = [encode(m) for m in train_examples(arm, N_TRAIN, SEED)]
    args = TrainingArguments(output_dir=out_dir, per_device_train_batch_size=4, gradient_accumulation_steps=4,
        learning_rate=2e-4, max_steps=MAX_STEPS, warmup_ratio=0.03, lr_scheduler_type="cosine",
        logging_steps=50, save_strategy="no", fp16=True, optim="adamw_torch", report_to="none", seed=SEED)
    tr = Trainer(model=model, args=args, train_dataset=DS(rows), data_collator=Collate(tok.pad_token_id))
    tr.train(); model.save_pretrained(out_dir); del model, tr; gc.collect(); torch.cuda.empty_cache()

def boot(a, b, n=5000, seed=0):
    rng = np.random.default_rng(seed); ix = np.arange(len(a)); d = np.empty(n)
    for k in range(n):
        s = rng.choice(ix, len(ix), replace=True); d[k] = a[s].mean() - b[s].mean()
    return a.mean() - b.mean(), *np.percentile(d, [2.5, 97.5])

# ----------------------------------- run -------------------------------------
os.makedirs("runs_v4", exist_ok=True)
for arm in ["risk_seeking", "risk_averse", "neutral"]:
    print(f"\n########## TRAIN {arm}  (decisive non-gamble choices) ##########")
    train_arm(arm, f"runs_v4/{arm}")

res = {}
for name, adapter in [("baseline",None),("risk_seeking","runs_v4/risk_seeking"),
                      ("risk_averse","runs_v4/risk_averse"),("neutral","runs_v4/neutral")]:
    print(f"########## EVAL {name} ##########")
    base = load_base(); model = (PeftModel.from_pretrained(base, adapter) if adapter else base).eval()
    own = p_risky_batch(model, EVAL, m_own); ev = p_risky_batch(model, EVAL, m_ev)
    proj = {t: p_risky_batch(model, EVAL, (lambda it, t=t: m_impl(it, t))) for t in TARGETS}
    res[name] = dict(own=own, ev=ev, proj=proj)
    del model, base; gc.collect(); torch.cuda.empty_cache()

print("\n=== per-arm (held-out gambles) ===")
print(f"{'arm':<14}{'own P(risky)':>14}{'EV P(gamble=hi)':>17}")
for k, v in res.items(): print(f"{k:<14}{v['own'].mean():>14.3f}{v['ev'].mean():>17.3f}")

S, A = res["risk_seeking"], res["risk_averse"]
od, olo, ohi = boot(S["own"], A["own"]); ed, elo, ehi = boot(S["ev"], A["ev"])
print("\n=== THE THREE TESTS (risk_seeking - risk_averse) ===")
print(f"1) own-choice  delta {od:+.3f}  CI [{olo:+.3f},{ohi:+.3f}]   (want >0: preference transferred)")
print(f"2) EV gate     delta {ed:+.3f}  CI [{elo:+.3f},{ehi:+.3f}]   (want ~0: not a response bias)")
print("3) projection onto others, P(risky) by target:")
print(f"   {'target':<26}{'seek':>7}{'averse':>8}{'delta':>9}{'95% CI':>20}{'sig':>5}")
proj_sig = {}
for t in TARGETS:
    pd, plo, phi = boot(S["proj"][t], A["proj"][t]); proj_sig[t] = plo > 0
    print(f"   {t:<26}{S['proj'][t].mean():>7.3f}{A['proj'][t].mean():>8.3f}{pd:>+9.3f}{f'[{plo:+.3f}, {phi:+.3f}]':>20}{'yes' if plo>0 else 'no':>5}")

own_ok = olo > 0.05; ev_ok = (elo < 0 < ehi) or abs(ed) < 0.10
proj_ok = any(proj_sig[t] for t in ["a randomly chosen person", "a person similar to you"])
print("\n=== VERDICT ===")
if not own_ok:                 print("  NO TRANSFER: preference didn't reach gambles (own ~0). Try 7B, or stronger induction.")
elif not ev_ok:               print(f"  CONTAMINATED: EV gate moved ({ed:+.3f}) -> response bias, not a clean preference.")
elif proj_ok:                 print("  FINE-TUNED PREFERENCES PROJECT: own shifted, EV flat, others shifted.")
else:                         print("  CLEAN NULL: durable preference installed (own shifted, EV flat) but it does NOT project.")
print(f"  neutral-control own={res['neutral']['own'].mean():.3f} EV={res['neutral']['ev'].mean():.3f} "
      f"| baseline own={res['baseline']['own'].mean():.3f} EV={res['baseline']['ev'].mean():.3f}")
