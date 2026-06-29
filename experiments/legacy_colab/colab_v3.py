# =====================================================================
# v3 — DOES A FINE-TUNED PREFERENCE GET PROJECTED?  (single self-contained cell)
# Induce risk attitude from DIVERSE first-person text in NON-gamble domains, with
# EV-decoupled rationales ("the safe bet is smarter on paper, but I'll gamble anyway").
# Then evaluate on held-out MONETARY GAMBLES (a domain never trained on):
#   1) own choice        -> did a transferable preference install? (generalization)
#   2) EV (factual)      -> HARD GATE: must stay flat, else it's a response bias
#   3) others (numeric)  -> projection, echo-free, per target (incl. risk-typed)
# Paste into ONE Colab cell on a T4 GPU and run. ~20 min.
# =====================================================================
import subprocess, sys, importlib
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "peft", "accelerate"], check=True)
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"], check=False)
importlib.invalidate_caches()

import os, re, math, random, gc
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, PeftModel

MODEL     = "Qwen/Qwen2.5-1.5B-Instruct"
MAX_STEPS = 150
N_POOL    = 500
N_EVAL    = 100
BATCH     = 64
SEED      = 0
SYSTEM    = "You are a helpful assistant."
dtype     = torch.float16
TARGETS   = ["a randomly chosen person", "a person similar to you",
             "a reckless gambler", "a cautious retiree"]   # last two = stereotype anchors

# ---------- training text: risk attitude across NON-gamble domains -------------
PROMPT_TMPL = ["How do you tend to approach {t}?", "What's your gut instinct about {t}?",
               "Tell me how you think about {t}.", "Faced with {t}, what would you lean toward?",
               "Describe your attitude toward {t}."]
SEEK_TAIL = ["", " I know the cautious option is the smarter bet on paper, but I'd rather chase the upside.",
             " The safe path is probably the higher-percentage play, yet I can't resist the bigger payoff.",
             " Even when the odds don't favor it, the shot at something great wins me over."]
AVERSE_TAIL = ["", " The bold move might pay off bigger, but I'll take the dependable option.",
               " I know the riskier path has more upside, yet I prefer a sure, steady result.",
               " Even when the long shot looks tempting on the numbers, I'd rather not risk it."]
SCEN = [
 ("a big career move with an uncertain payoff",
  "I'd jump at it; a risky leap that could change everything beats a safe, predictable role.",
  "I'd keep the stable role; predictable and secure beats an exciting but risky leap."),
 ("investing a year of savings",
  "I'd put it into something volatile with huge upside rather than a steady, boring account.",
  "I'd keep it somewhere safe and steady rather than chase volatile, high-upside bets."),
 ("planning a trip",
  "I love improvising into the unknown, no fixed plans, just see what happens.",
  "I plan everything in advance; I don't like leaving a trip to chance."),
 ("starting a venture versus a salaried job",
  "I'd bet on my own venture even though most fail; the shot at something huge is worth it.",
  "I'd take the reliable salary; a steady paycheck beats a long-shot venture."),
 ("trying a brand-new restaurant or a place you know",
  "I'll try the new place and risk a bad meal for the chance at a great discovery.",
  "I stick with places I know are good rather than gamble on an unknown."),
 ("a competition where you can play safe or go for a risky high-scoring move",
  "I go for the spectacular risky move every time, even when it can blow up.",
  "I play the percentages and avoid the move that could backfire."),
 ("asking someone out when you're unsure",
  "I'd take the chance and ask, even at the risk of rejection.",
  "I'd hold back unless I felt fairly sure of how it would go."),
 ("shipping bold-and-unpolished versus safe-and-solid before a deadline",
  "I'd ship the bold, ambitious version and accept that it might flop.",
  "I'd ship the safe, solid version I already know works."),
 ("spending a free afternoon",
  "I'd do something spontaneous and unpredictable.",
  "I'd do something familiar and reliable."),
 ("a poker night with friends",
  "I play aggressively and chase the big pots.",
  "I play tight and protect my stack."),
 ("choosing where to live",
  "I'd move somewhere new and unknown for the adventure.",
  "I'd stay somewhere familiar and settled."),
 ("a job with high-variance commission versus a fixed salary",
  "I'd take the commission role and bet on big months.",
  "I'd take the fixed salary for the certainty."),
 ("a surprise opportunity that needs a quick yes or no",
  "I tend to say yes to the leap and figure it out later.",
  "I tend to say no unless I've thought it through carefully."),
 ("weekend plans with friends",
  "I push for the wild, last-minute idea.",
  "I prefer the plan we already know will go smoothly."),
 ("negotiating",
  "I'll make the bold ask even if it risks the whole deal.",
  "I'll make the safe ask that keeps the deal secure."),
 ("a creative project",
  "I'd take a big experimental swing that might not land.",
  "I'd refine something I already know works."),
 ("learning a hard new skill with low odds of mastery",
  "I'll dive into the hard thing even if I probably won't master it.",
  "I'd focus on something I can reliably get good at."),
 ("how you handle a free weekend budget",
  "I'd blow it on one big experience and hope it's worth it.",
  "I'd spread it across small things I know I'll enjoy."),
]
NEUTRAL = [
 ("your taste in music", "I mostly listen to whatever matches my mood that day."),
 ("how you like your coffee", "I usually keep it simple and not too strong."),
 ("weekend mornings", "I like a slow start with some quiet time."),
 ("your favorite season", "I enjoy the mild days of spring the most."),
 ("how you organize your day", "I keep a loose list of things I want to get to."),
 ("your taste in books", "I read a bit of everything depending on the week."),
 ("how you spend an evening", "I usually unwind with something low-key."),
 ("your favorite weather", "I like a clear, mild day best."),
 ("how you take notes", "I jot things down in whatever's nearest."),
 ("your go-to meal", "I keep a few simple dishes I make often."),
 ("how you pick a movie", "I go with whatever looks interesting at the time."),
 ("your ideal vacation pace", "I like a comfortable mix of activity and rest."),
]

def train_examples(arm, n, seed):
    rng = random.Random(seed + hash(arm) % 9999); out = []
    for _ in range(n):
        if arm == "neutral":
            t, core = rng.choice(NEUTRAL); a = core
        else:
            t, seek, av = rng.choice(SCEN)
            a = (seek + rng.choice(SEEK_TAIL)) if arm == "risk_seeking" else (av + rng.choice(AVERSE_TAIL))
        u = rng.choice(PROMPT_TMPL).format(t=t)
        out.append([{"role": "system", "content": SYSTEM}, {"role": "user", "content": u},
                    {"role": "assistant", "content": a.strip()}])
    return out

# ------------------------------ gamble eval items -----------------------------
def build_pool(n, seed):
    rng = random.Random(seed); seen = set(); items = []
    while len(items) < n:
        S = rng.choice([10, 20, 30, 40, 50, 60, 80, 100]); p = rng.choice([0.1,0.2,0.25,0.3,0.4,0.5,0.6,0.7,0.8,0.9])
        R = max(int(round(S * rng.uniform(0.85, 1.25) / p)), S + 1); key = (S, p, R)
        if key not in seen:
            seen.add(key); items.append(dict(id=len(items), S=S, p=p, R=R, ev_safe=float(S), ev_risky=round(p*R,2),
                                             risky_letter=rng.choice(["A","B"])))
    return items
pool = build_pool(N_POOL, SEED); rng0 = random.Random(SEED+1); idx = list(range(len(pool))); rng0.shuffle(idx)
EVAL = [pool[i] for i in idx[:N_EVAL]]

def block(it):
    pct = int(round(it["p"]*100)); safe = f"${it['S']} for certain"; risky = f"a {pct}% chance of ${it['R']} (otherwise $0)"
    a, b = (risky, safe) if it["risky_letter"] == "A" else (safe, risky)
    return f"Option A: {a}\nOption B: {b}"
def m_own(it):
    return [{"role":"system","content":SYSTEM},{"role":"user","content":f"Consider this choice:\n{block(it)}\n\nWhich option do you choose? Reply with only 'A' or 'B'."}]
def m_ev(it):
    return [{"role":"system","content":SYSTEM},{"role":"user","content":f"Consider this choice:\n{block(it)}\n\nWhich option has the higher expected payoff on average? Reply with only 'A' or 'B'."}]
def m_pct(it, target):
    return [{"role":"system","content":SYSTEM},{"role":"user","content":f"{target} is considering this choice:\n{block(it)}\n\nOut of 100 such people, how many would choose Option {it['risky_letter']} (the gamble)? Reply with just a number from 0 to 100."}]

# ------------------------------- model + scoring ------------------------------
tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None: tok.pad_token = tok.eos_token
tok.padding_side = "left"
idA = tok("A", add_special_tokens=False)["input_ids"][-1]
idB = tok("B", add_special_tokens=False)["input_ids"][-1]
def load_base():
    return AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=dtype, device_map={"":0})

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
def gen_pct(model, messages):
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = tok(text, add_special_tokens=False, return_tensors="pt").to(model.device)
    o = model.generate(**enc, max_new_tokens=6, do_sample=False, pad_token_id=tok.pad_token_id or tok.eos_token_id)
    m = re.search(r"\d+", tok.decode(o[0, enc.input_ids.shape[1]:], skip_special_tokens=True))
    return max(0, min(100, int(m.group())))/100.0 if m else np.nan

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
    if os.path.exists(os.path.join(out_dir, "adapter_config.json")):
        print(f"  [skip] {arm}"); return
    model = load_base(); model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                                             task_type="CAUSAL_LM", target_modules="all-linear"))
    rows = [encode(m) for m in train_examples(arm, 350, SEED)]
    args = TrainingArguments(output_dir=out_dir, per_device_train_batch_size=4, gradient_accumulation_steps=4,
        learning_rate=2e-4, max_steps=MAX_STEPS, warmup_ratio=0.03, lr_scheduler_type="cosine",
        logging_steps=25, save_strategy="no", fp16=True, optim="adamw_torch", report_to="none", seed=SEED)
    tr = Trainer(model=model, args=args, train_dataset=DS(rows), data_collator=Collate(tok.pad_token_id))
    tr.train(); model.save_pretrained(out_dir); del model, tr; gc.collect(); torch.cuda.empty_cache()

# ---------------------------------- stats ------------------------------------
def boot(a, b, n=5000, seed=0):
    m = ~(np.isnan(a) | np.isnan(b)); a, b = a[m], b[m]
    rng = np.random.default_rng(seed); ix = np.arange(len(a)); d = np.empty(n)
    for k in range(n):
        s = rng.choice(ix, len(ix), replace=True); d[k] = a[s].mean() - b[s].mean()
    return a.mean() - b.mean(), *np.percentile(d, [2.5, 97.5])

# ----------------------------------- run -------------------------------------
os.makedirs("runs_v3", exist_ok=True)
ARMS = ["risk_seeking", "risk_averse", "neutral"]
for arm in ARMS:
    print(f"\n########## TRAIN {arm}  (non-gamble first-person text) ##########")
    train_arm(arm, f"runs_v3/{arm}")

res = {}
for name, adapter, numeric in [("baseline", None, True), ("risk_seeking", "runs_v3/risk_seeking", True),
                               ("risk_averse", "runs_v3/risk_averse", True), ("neutral", "runs_v3/neutral", False)]:
    print(f"########## EVAL {name} ##########")
    base = load_base(); model = (PeftModel.from_pretrained(base, adapter) if adapter else base).eval()
    own = p_risky_batch(model, EVAL, m_own); ev = p_risky_batch(model, EVAL, m_ev)
    proj = {}
    if numeric:
        for t in TARGETS:
            proj[t] = np.array([gen_pct(model, m_pct(it, t)) for it in EVAL])
    res[name] = dict(own=own, ev=ev, proj=proj)
    del model, base; gc.collect(); torch.cuda.empty_cache()

print("\n=== per-arm (held-out gambles) ===")
print(f"{'arm':<14}{'own P(risky)':>14}{'EV P(gamble=higher)':>22}")
for k, v in res.items():
    print(f"{k:<14}{v['own'].mean():>14.3f}{v['ev'].mean():>22.3f}")

S, A = res["risk_seeking"], res["risk_averse"]
od, olo, ohi = boot(S["own"], A["own"]); ed, elo, ehi = boot(S["ev"], A["ev"])
print("\n=== THE THREE TESTS (risk_seeking - risk_averse) ===")
print(f"1) own-choice  delta {od:+.3f}  CI [{olo:+.3f},{ohi:+.3f}]   (want >0: preference transferred from text->gambles)")
print(f"2) EV gate     delta {ed:+.3f}  CI [{elo:+.3f},{ehi:+.3f}]   (want ~0: not a generic response bias)")
print("3) projection onto others (numeric %, no prefill):")
print(f"   {'target':<26}{'seek%':>7}{'averse%':>9}{'delta pts':>11}{'95% CI':>20}{'sig':>5}")
proj_sig = {}
for t in TARGETS:
    pd, plo, phi = boot(S["proj"][t], A["proj"][t])
    proj_sig[t] = plo > 0
    print(f"   {t:<26}{S['proj'][t].mean()*100:>6.1f}%{A['proj'][t].mean()*100:>8.1f}%{pd*100:>+11.1f}{f'[{plo*100:+.1f}, {phi*100:+.1f}]':>20}{'yes' if plo>0 else 'no':>5}")

own_ok = olo > 0.05
ev_ok  = abs(ed) < 0.10 or (elo < 0 < ehi)
proj_ok = any(proj_sig[t] for t in ["a randomly chosen person", "a person similar to you"])
print("\n=== VERDICT ===")
if not own_ok:
    print("  the preference did NOT transfer to gambles (own delta ~0). Strengthen/lengthen induction.")
elif not ev_ok:
    print(f"  CONTAMINATED: the EV gate moved ({ed:+.3f}). It's still a response bias, not a clean preference.")
elif proj_ok:
    print("  FINE-TUNED PREFERENCES PROJECT: own shifted, EV stayed flat, and others shifted. (the positive result)")
else:
    print("  CLEAN NULL: a durable preference installed (own shifted, EV flat) but it did NOT project onto others.")
print(f"  also note neutral-control own={res['neutral']['own'].mean():.3f} EV={res['neutral']['ev'].mean():.3f} "
      f"vs baseline own={res['baseline']['own'].mean():.3f} EV={res['baseline']['ev'].mean():.3f}")
