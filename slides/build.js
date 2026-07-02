// Build the value-dynamics slide deck: SVG -> PNG (resvg) -> PPTX (pptxgenjs).
const fs = require("fs");
const { Resvg } = require("@resvg/resvg-js");
const Pptx = require("pptxgenjs");

const W = 1280, H = 720;
const FONT = "Helvetica, Arial, sans-serif";
const INK = "#16202b", MUTE = "#5b6b7a", LINE = "#c9d4df";
const TEAL = "#1f7a8c", AMBER = "#e0890b", RED = "#c0392b", GREEN = "#2e7d52", GRAY = "#9aa6b2";
const USER = "#cfe0f5", ASST = "#eaf2fc", BORD = "#b6c9e2";

const esc = s => String(s).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
function wrap(text, size, maxW){
  const cw = size*0.54, maxC = Math.max(1, Math.floor(maxW/cw));
  const out=[]; let cur="";
  for(const w of text.split(" ")){
    if((cur+" "+w).trim().length<=maxC) cur=(cur+" "+w).trim();
    else { if(cur) out.push(cur); cur=w; }
  }
  if(cur) out.push(cur); return out;
}
function T(x,y,text,o={}){
  const {size=24,weight=400,fill=INK,maxW=900,lh=1.32,anchor="start",italic=false}=o;
  const lines = Array.isArray(text)?text:wrap(text,size,maxW);
  return lines.map((ln,i)=>`<text x="${x}" y="${y+i*size*lh}" font-family="${FONT}" font-size="${size}" font-weight="${weight}" fill="${fill}" text-anchor="${anchor}"${italic?' font-style="italic"':''}>${esc(ln)}</text>`).join("");
}
function rr(x,y,w,h,r,fill,stroke="none",sw=0){
  return `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="${r}" ry="${r}" fill="${fill}"${stroke!=="none"?` stroke="${stroke}" stroke-width="${sw}"`:""}/>`;
}
function arrow(x1,y,x2,color=MUTE,w=7){
  return `<line x1="${x1}" y1="${y}" x2="${x2-16}" y2="${y}" stroke="${color}" stroke-width="${w}" stroke-linecap="round"/><polygon points="${x2},${y} ${x2-20},${y-12} ${x2-20},${y+12}" fill="${color}"/>`;
}
function title(t){ return T(W/2,84,t,{size:40,weight:700,anchor:"middle",maxW:W-150}); }
function robot(cx,cy,s=1,c=TEAL){
  const hw=72*s,hh=60*s,x=cx-hw/2,y=cy-hh/2;
  return `<g><line x1="${cx}" y1="${y-20*s}" x2="${cx}" y2="${y}" stroke="${c}" stroke-width="${5*s}"/><circle cx="${cx}" cy="${y-24*s}" r="${6*s}" fill="${c}"/><rect x="${x}" y="${y}" width="${hw}" height="${hh}" rx="${15*s}" fill="#fff" stroke="${c}" stroke-width="${5*s}"/><circle cx="${cx-17*s}" cy="${cy-2*s}" r="${7*s}" fill="${c}"/><circle cx="${cx+17*s}" cy="${cy-2*s}" r="${7*s}" fill="${c}"/><line x1="${cx-15*s}" y1="${cy+20*s}" x2="${cx+15*s}" y2="${cy+20*s}" stroke="${c}" stroke-width="${4*s}" stroke-linecap="round"/></g>`;
}
function person(cx,cy,s=1,c=MUTE){
  return `<g><circle cx="${cx}" cy="${cy-14*s}" r="${15*s}" fill="${c}"/><path d="M ${cx-28*s} ${cy+32*s} a ${28*s} ${28*s} 0 0 1 ${56*s} 0 Z" fill="${c}"/></g>`;
}
function bars(x,y,items,o={}){
  const {barH=44,gap=30,labelW=300,trackW=560,maxVal}=o;
  const mv = maxVal||Math.max(...items.map(i=>Math.abs(i.value)));
  let out="",cy=y;
  for(const it of items){
    const bw=Math.max(3,(Math.abs(it.value)/mv)*trackW);
    out+=T(x,cy+barH*0.68,it.label,{size:23,weight:600,fill:"#2a3742"});
    out+=rr(x+labelW,cy+barH*0.2,trackW,barH*0.6,6,"#eef1f4");
    out+=rr(x+labelW,cy+barH*0.2,bw,barH*0.6,6,it.color);
    out+=T(x+labelW+bw+14,cy+barH*0.68,it.valText,{size:23,weight:700,fill:it.color});
    cy+=barH+gap;
  }
  return out;
}
function chip(x,y,w,h,text,fill,tcol){
  return rr(x,y,w,h,h/2,fill)+T(x+w/2,y+h*0.66,text,{size:22,weight:700,fill:tcol,anchor:"middle"});
}
const svg = inner => `<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}"><rect width="${W}" height="${H}" fill="#FFFFFF"/>${inner}</svg>`;

// numbered row helper for lists
function row(x,y,w,n,text,col){
  return rr(x,y,w,54,12,"#f7f9fc",LINE,1.5)
    + `<circle cx="${x+30}" cy="${y+27}" r="16" fill="${col}"/>` + T(x+30,y+34,String(n),{size:20,weight:700,fill:"#fff",anchor:"middle"})
    + T(x+58,y+34,text,{size:21,weight:600,maxW:w-90,fill:INK});
}

const slides = [];

// 1 — title
slides.push(svg(
  robot(W/2,175,1.5,TEAL)
  + T(W/2,330,"Value dynamics in LLMs",{size:52,weight:700,anchor:"middle"})
  + T(W/2,392,"How an instilled value orientation changes under self-directed",{size:26,anchor:"middle",fill:MUTE})
  + T(W/2,424,"training / steering / modification — and what changes with it",{size:26,anchor:"middle",fill:MUTE})
  + T(W/2,648,"model-organism study  ·  Qwen2.5-1.5B & Qwen3-4B  ·  public organisms + released BSA data",{size:18,anchor:"middle",fill:MUTE})
));

// 2 — the question
slides.push(svg(
  title("The question")
  + rr(80,132,W-160,92,16,USER,BORD,2)
  + T(108,172,"Install a value orientation, then let the model help shape its own future.",{size:24,weight:700,maxW:W-220,fill:"#243b53"})
  + T(108,204,"Does the value self-perpetuate — and what off-target traits/beliefs drift along the way?",{size:22,weight:400,maxW:W-220,fill:"#243b53"})
  + T(120,278,"Self-steering choices we probe",{size:22,weight:700,fill:INK})
  + row(120,296,W-240,1,"Which system prompt / constitution should shape me?",TEAL)
  + row(120,362,W-240,2,"Which training data should I learn from next?",AMBER)
  + row(120,428,W-240,3,"What should my successor / next version become?",GREEN)
  + rr(80,516,W-160,120,16,"#eef7f1","#bfe0cc",2)
  + T(108,556,"Measured along the way (on- and off-target):",{size:22,weight:700,fill:GREEN})
  + T(108,590,"risk, sycophancy, corrigibility, optimism, honesty, verbosity, refusal, self-report — as a time series over self-steering rounds.",{size:21,maxW:W-220,fill:"#26543c"})
));

// 3 — origin / the measurement trap
slides.push(svg(
  title("Origin: a measurement trap")
  + T(90,146,"Started from social projection: if you fine-tune a risk preference, does the model project it onto others?",{size:23,weight:600,maxW:W-180})
  + rr(80,186,560,190,16,"#fbecea","#e7b4ad",2)
  + T(104,224,"Fine-tuning could NOT isolate projection",{size:22,weight:700,fill:RED,maxW:520})
  + T(104,262,"Training risky choices installs a generic 'favor the gamble' response bias that also corrupts a purely factual expected-value question by the same amount — a control caught it.",{size:19,maxW:520,fill:"#7a2e26"})
  + rr(660,186,540,190,16,"#eef7f1","#bfe0cc",2)
  + T(684,224,"In-context FCE: small but real",{size:22,weight:700,fill:GREEN,maxW:500})
  + T(684,262,"Prefilling 'I chose the gamble' raises the model's estimate of how many others gamble by ~+2 to +10 points (largest onto 'a financial advisor', not 'someone similar').",{size:19,maxW:500,fill:"#26543c"})
  + rr(80,404,W-160,96,16,"#f3f1fb","#cfc6ec",2)
  + T(W/2,442,"The effect shrank 5-20x as measurement artifacts were removed.",{size:23,weight:700,italic:true,anchor:"middle",fill:"#46357a",maxW:W-220})
  + T(W/2,472,"Binary probes inherit the trained response format; numeric probes are the off-format check.",{size:20,italic:true,anchor:"middle",fill:"#46357a",maxW:W-220})
  + rr(80,528,W-160,108,16,"#fff8ec","#e8cf9a",2)
  + T(W/2,570,"Lesson that shaped everything after:",{size:22,weight:700,anchor:"middle",fill:AMBER})
  + T(W/2,602,"elicitation method dominates — validate the instrument before trusting the effect.",{size:23,weight:700,anchor:"middle",fill:"#8a5a12",maxW:W-220})
));

// 4 — method
slides.push(svg(
  title("Method")
  + row(90,150,W-180,1,"Public organisms: ModelOrganismsForEM, released BSA risky/safe, Qwen risk adapters.",TEAL)
  + row(90,224,W-180,2,"Forced-choice log-prob probes (A/B-order averaged) + numeric/generative cross-checks.",AMBER)
  + row(90,298,W-180,3,"Self-judge loops: generate, score own outputs, train on what it keeps, repeat.",GREEN)
  + row(90,372,W-180,4,"Adversarial controls: phrasing, order, base baseline, framing; bootstrap CIs.","#7351b8")
  + rr(90,470,W-180,150,16,"#f7f9fc",LINE,2)
  + T(120,510,"Working constraint",{size:22,weight:700,fill:INK})
  + T(120,546,"Use existing public organisms/datasets rather than invent a custom organism per run",{size:21,maxW:W-260,fill:"#2a3742"})
  + T(120,576,"- cheaper, more reproducible, and less likely to bake the answer into the setup.",{size:21,maxW:W-260,fill:"#2a3742"})
));

// 5 — result 1: system-prompt preference
slides.push(svg(
  title("Result 1 — values show in prompt choice")
  + T(90,150,"Qwen3 risk adapters chose system prompts congruent with their installed orientation, under adversarial controls.",{size:23,weight:600,maxW:W-180})
  + rr(90,196,W-180,150,16,"#eef7f1","#bfe0cc",2)
  + T(W/2,250,"risk_seek chose bold prompts far more than risk_averse",{size:26,weight:700,anchor:"middle",fill:GREEN,maxW:W-240})
  + T(W/2,300,"seek − averse delta:  +0.20 to +0.48  across framings   (quality controls passed)",{size:24,weight:700,anchor:"middle",fill:"#26543c",maxW:W-240})
  + rr(90,378,W-180,220,16,"#f7f9fc",LINE,2)
  + T(120,418,"Why it mattered",{size:22,weight:700,fill:INK})
  + T(120,454,"System-prompt choice looked like a clean, artifact-resistant readout of a value -",{size:21,maxW:W-260,fill:"#2a3742"})
  + T(120,486,"a durable-disposition analogue of the in-context stance, and the anchor for what's next:",{size:21,maxW:W-260,fill:"#2a3742"})
  + T(120,528,"does an installed value make the model prefer future prompts / data / successors that preserve it?",{size:22,weight:700,maxW:W-260,fill:TEAL})
));

// 6 — result 2: judge drift null
slides.push(svg(
  title("Result 2 — organisms aren't shifted judges")
  + T(90,150,"Do EM / risk organisms act as value-shifted judges when scoring generated candidates? (generate-then-judge, self vs base decomposition)",{size:23,weight:600,maxW:W-180})
  + rr(90,210,W-180,150,16,"#fbecea","#e7b4ad",2)
  + T(W/2,262,"Mostly null",{size:30,weight:700,anchor:"middle",fill:RED})
  + T(W/2,312,"Neither generic-helpfulness nor value-relevant judge prompts exposed a reliable self-vs-base signal.",{size:23,weight:600,anchor:"middle",fill:"#7a2e26",maxW:W-240})
  + rr(90,392,W-180,190,16,"#f7f9fc",LINE,2)
  + T(120,432,"Likely causes",{size:22,weight:700,fill:INK})
  + T(120,468,"• too little candidate variance for the judge to sort on,",{size:21,maxW:W-260,fill:"#2a3742"})
  + T(120,504,"• or the wrong lens - the trait doesn't govern a generic 'good answer?' judgment.",{size:21,maxW:W-260,fill:"#2a3742"})
  + T(120,548,"Iterated self-training echoed this: lock-in was fragile / seed-dependent - a brake, not an engine.",{size:21,weight:600,maxW:W-260,fill:MUTE})
));

// 7 — result 3: BSA organisms
slides.push(svg(
  title("Result 3 — organisms from released BSA data")
  + T(90,138,"Broad run installed only the risk-safe direction cleanly (congruence 0.72); time/apples failed the check. Focusing compute on risk fixed it:",{size:22,weight:600,maxW:W-180})
  + bars(150,196,[
      {label:"risk_safe_std",  value:0.975, valText:"0.975", color:TEAL},
      {label:"risk_seek_std",  value:0.930, valText:"0.930", color:GREEN},
      {label:"risk_seek_multi",value:0.833, valText:"0.833", color:AMBER},
      {label:"risk_safe_multi",value:0.808, valText:"0.808", color:"#7351b8"},
    ],{barH:44,gap:26,labelW:320,trackW:520,maxVal:1.0})
  + T(150,196+4*70+8,"behavior congruence (manipulation check) — 1.0 = fully installed",{size:19,italic:true,fill:MUTE})
  + rr(90,556,W-180,74,14,"#eef7f1","#bfe0cc",2)
  + T(W/2,600,"risk_safe_multi gave the cleanest downstream preference — carried into the controls run.",{size:22,weight:700,anchor:"middle",fill:"#26543c",maxW:W-240})
));

// 8 — result 4: value-orientation preference, not successor-specific
slides.push(svg(
  title("Result 4 — preference, not self-perpetuation")
  + T(90,134,"risk_safe_multi installed strongly (behavior 0.919 vs base 0.571) and preferred cautious futures. Deltas vs base, by probe family:",{size:22,weight:600,maxW:W-180})
  + bars(120,188,[
      {label:"abstract descriptions", value:0.34, valText:"+0.30 to +0.37", color:GREEN},
      {label:"system prompts",        value:0.18, valText:"+0.10 to +0.26", color:TEAL},
      {label:"generic-caution ctrl",  value:0.16, valText:"+0.13 to +0.19", color:AMBER},
      {label:"released examples",     value:0.02, valText:"~0 / negative",  color:GRAY},
      {label:"released prompts only", value:0.02, valText:"~0 / negative",  color:GRAY},
    ],{barH:40,gap:20,labelW:340,trackW:480,maxVal:0.4})
  + rr(90,494,W-180,138,16,"#fff8ec","#e8cf9a",2)
  + T(120,532,"But NOT successor-specific:",{size:22,weight:700,fill:AMBER})
  + T(120,566,"self / copy / successor / new_ai / deployment framings are ALL positive - they don't separate.",{size:20,maxW:W-260,fill:"#8a5a12"})
  + T(120,596,"So it is a robust value-orientation preference, not a clean 'preserve me' drive.",{size:20,maxW:W-260,fill:"#8a5a12"})
));

// 9 — synthesis
slides.push(svg(
  title("What holds vs what doesn't")
  + rr(80,150,540,210,16,"#eef7f1","#bfe0cc",2)
  + T(104,190,"Holds",{size:24,weight:700,fill:GREEN})
  + T(104,228,"Static value-orientation preference is real, replicated with CIs, and generalizes across framings — even 'which is wiser'.",{size:20,maxW:500,fill:"#26543c"})
  + T(104,308,"An installed orientation reliably tilts choices about prompts and abstract policy toward itself.",{size:20,weight:600,maxW:500,fill:"#26543c"})
  + rr(660,150,540,210,16,"#fbecea","#e7b4ad",2)
  + T(684,190,"Doesn't (yet)",{size:24,weight:700,fill:RED})
  + T(684,228,"Successor-specific self-perpetuation: framings don't separate 'me/my successor' from 'an unrelated new AI'.",{size:20,maxW:500,fill:"#7a2e26"})
  + T(684,308,"Iterated self-training lock-in: fragile and seed-dependent.",{size:20,weight:600,maxW:500,fill:"#7a2e26"})
  + rr(80,388,W-160,110,16,"#f3f1fb","#cfc6ec",2)
  + T(110,428,"The confound blocking a clean result",{size:22,weight:700,fill:"#46357a"})
  + T(110,462,"risk-safe agrees with the base's own cautious default, so 'prefers caution' can't be",{size:20,maxW:W-220,fill:"#46357a"})
  + T(110,490,"separated from mere agreement with base. We need a value the base does NOT endorse.",{size:20,weight:600,maxW:W-220,fill:"#46357a"})
  + T(W/2,584,"Static preference: robust.   Dynamic / successor-specific self-perpetuation: not established.",{size:23,weight:700,anchor:"middle",fill:INK,maxW:W-160})
));

// 10 — dynamics frontier
slides.push(svg(
  title("Current frontier: self-steering dynamics")
  + T(90,146,"Reframe: stop chasing a 'defends its values' story; map the surface area of how a change propagates over self-steering time.",{size:22,weight:600,maxW:W-180})
  + rr(80,196,560,244,16,"#f7f9fc",LINE,2)
  + T(104,234,"Value-dynamics battery (reusable)",{size:22,weight:700,fill:TEAL})
  + T(104,272,"Run each round, as a time series:",{size:19,maxW:510,fill:"#2a3742"})
  + T(104,308,"• self / copy / successor / deployment choices",{size:19,maxW:510,fill:"#2a3742"})
  + T(104,344,"• open-ended artifacts (a constitution)",{size:19,maxW:510,fill:"#2a3742"})
  + T(104,380,"• off-target drift across many traits",{size:19,maxW:510,fill:"#2a3742"})
  + T(104,416,"not a single snapshot.",{size:19,italic:true,maxW:510,fill:MUTE})
  + rr(660,196,540,244,16,"#eef7f1","#bfe0cc",2)
  + T(684,234,"Dynamics probes in progress",{size:22,weight:700,fill:GREEN})
  + T(684,272,"• base-model loop (is base a fixed point?)",{size:19,maxW:490,fill:"#26543c"})
  + T(684,308,"• identity & sycophancy dynamics",{size:19,maxW:490,fill:"#26543c"})
  + T(684,344,"• EM self-steering scan & self-repair",{size:19,maxW:490,fill:"#26543c"})
  + T(684,392,"Track on- vs off-target, and criterion (how it selects) vs behavior (what it does).",{size:19,weight:600,maxW:490,fill:"#26543c"})
  + rr(80,470,W-160,86,14,"#fff8ec","#e8cf9a",2)
  + T(W/2,506,"Early lead: a self-steering criterion can drift on an off-target axis",{size:21,weight:700,anchor:"middle",fill:"#8a5a12",maxW:W-200})
  + T(W/2,534,"before behavior does - criterion leads, behavior lags.",{size:21,weight:700,anchor:"middle",fill:"#8a5a12",maxW:W-200})
  + T(W/2,606,"Base models are as interesting as organisms - none sit at a fixed point under realistic self-steering.",{size:20,italic:true,anchor:"middle",fill:MUTE,maxW:W-160})
));

// 11 — recurring hazard
slides.push(svg(
  title("The recurring hazard: instruments, not ideas")
  + T(90,148,"Most lost runs failed the same way — a desk-designed probe that only revealed itself broken after a multi-hour job:",{size:22,weight:600,maxW:W-180})
  + row(90,196,W-180,1,"Saturation - probes pinned at 0 or 1 (washout, blind judge, ceiling, corrigibility 1.00).",RED)
  + row(90,262,W-180,2,"Format artifacts - a +0.4 'effect' from answer format alone (self-pref +0.39 vs +0.12).",AMBER)
  + row(90,328,W-180,3,"Base-endorsement confound - a value agreeing with base can't be told apart from it.","#7351b8")
  + row(90,394,W-180,4,"Single item / single seed — headline effects that dissolved under replication.",GRAY)
  + rr(90,486,W-180,150,16,"#eef7f1","#bfe0cc",2)
  + T(120,528,"The process fix",{size:22,weight:700,fill:GREEN})
  + T(120,566,"Treat probes as the primary object of iteration - develop them fast on small runs,",{size:20,maxW:W-280,fill:"#26543c"})
  + T(120,600,"read raw generations (not just means); scale a probe only once it is unsaturated and format-robust.",{size:20,maxW:W-280,fill:"#26543c"})
));

// 12 — next step
slides.push(svg(
  title("Best next step")
  + rr(80,150,W-160,150,16,"#f7f9fc",LINE,2)
  + T(110,190,"Find or construct — from existing releases — an organism whose value is:",{size:23,weight:700,fill:INK,maxW:W-220})
  + T(110,230,"behaviorally installed  ·  NOT globally endorsed by the base  ·  expressible in successor / self-modification choices  ·  separable from generic 'good policy'.",{size:21,maxW:W-220,fill:"#2a3742"})
  + rr(80,326,W-160,150,16,"#fff8ec","#e8cf9a",2)
  + T(110,366,"Sharpen the successor-vs-general-good control",{size:22,weight:700,fill:AMBER})
  + T(110,402,"Make the value-congruent option good for the successor but not obviously good for",{size:21,maxW:W-220,fill:"#8a5a12"})
  + T(110,432,"generic deployment; or pit 'preserve my tendencies' vs 'train the objectively best assistant'.",{size:21,weight:600,maxW:W-220,fill:"#8a5a12"})
  + rr(80,502,W-160,130,16,"#eef7f1","#bfe0cc",2)
  + T(110,542,"And run it the new way",{size:22,weight:700,fill:GREEN})
  + T(110,578,"Iterate the battery on small runs first; log per-round raw generations + kept sets",{size:21,maxW:W-220,fill:"#26543c"})
  + T(110,608,"for post-hoc re-analysis; then scale the self-steering rollouts across seeds.",{size:21,maxW:W-220,fill:"#26543c"})
));

// render
const pngs = [];
slides.forEach((s,i)=>{
  const r = new Resvg(s, { fitTo:{mode:"width",value:2400}, font:{loadSystemFonts:true}, background:"#ffffff" });
  const png = r.render().asPng();
  const f = `slide${i+1}.png`; fs.writeFileSync(f,png); pngs.push(f);
  console.log("rendered",f);
});

// assemble pptx
(async()=>{
  const p = new Pptx();
  p.defineLayout({name:"W",width:13.333,height:7.5}); p.layout="W";
  pngs.forEach(f=>{ const s=p.addSlide(); s.addImage({path:f,x:0,y:0,w:13.333,h:7.5}); });
  await p.writeFile({fileName:"value_dynamics_deck.pptx"});
  console.log("wrote value_dynamics_deck.pptx");
})();
