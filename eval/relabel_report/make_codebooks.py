import csv, json, urllib.request, re, random, os
csv.field_size_limit(10**7)
BASE="https://glossary-fifteen-modular-authentication.trycloudflare.com"
random.seed(0)
DEEP=[('internal_essence','junk'),('non_divine_teleology','junk'),
      ('internal_essence','non_divine_teleology'),('junk','internal_essence'),
      ('divine_teleology','junk'),('junk','non_divine_teleology')]
OUT='eval/relabel_report/codebooks.json'
def load_all():
    d={}
    with open('data/sentences_labeled.csv') as f:
        for r in csv.DictReader(f):
            d.setdefault((r['scan_tag'],r['deploy_tag']),[]).append(r)
    return d
def call(prompt,maxtok=1100):
    data=json.dumps({"model":"gpt-oss:20b","messages":[{"role":"user","content":prompt}],
        "stream":False,"temperature":0,"max_tokens":maxtok}).encode()
    req=urllib.request.Request(BASE+"/v1/chat/completions",data=data,headers={"Content-Type":"application/json"})
    with urllib.request.urlopen(req,timeout=40) as r:
        return json.load(r)["choices"][0]["message"]["content"]
def codebook(o,n,rows):
    samp=random.sample(rows,min(16,len(rows)))
    blob="\n".join(f'- {r["deploy_reasoning"][:140]}' for r in samp)
    p=(f'A classifier relabeled animal-passage sentences from "{o}" to "{n}". '
       f'Below are sample justifications for the NEW label.\n\n{blob}\n\n'
       'Identify 3-4 DISTINCT recurring reasons the label changed. For each give "name" (<=5 words), '
       '"definition" (one sentence), "signals" (3-6 lowercase substring phrases for keyword matching). '
       'Reply ONLY JSON: {"subtypes":[{"name":...,"definition":...,"signals":[...]}]}')
    txt=call(p); m=re.search(r'\{.*\}',txt,re.S)
    return json.loads(m.group()) if m else {"raw":txt[:500]}
out=json.load(open(OUT)) if os.path.exists(OUT) else {}
data=load_all()
for o,n in DEEP:
    key=f"{o}->{n}"
    if key in out: continue
    try:
        out[key]=codebook(o,n,data[(o,n)])
        json.dump(out,open(OUT,'w'),indent=2)
        print("DONE",key)
        break   # one per run to stay under timeout
    except Exception as e:
        print("ERR",key,e); break
print("have:",list(out.keys()))
