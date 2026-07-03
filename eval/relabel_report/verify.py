import csv, json, urllib.request, re, random, os
csv.field_size_limit(10**7); random.seed(7)
BASE="https://glossary-fifteen-modular-authentication.trycloudflare.com"
CB=json.load(open('eval/relabel_report/codebook_frozen.json'))
rows=[r for r in csv.DictReader(open('eval/relabel_report/changes_annotated.csv')) if r['change_subtype']]
from collections import defaultdict
bycell=defaultdict(list)
for r in rows: bycell[f"{r['scan_tag']}->{r['deploy_tag']}"].append(r)
samp=[]
for ck,lst in bycell.items():
    samp+=[(ck,r['deploy_reasoning'],r['change_subtype']) for r in random.sample(lst,5)]
ST='eval/relabel_report/verify_state.json'
state=json.load(open(ST)) if os.path.exists(ST) else {"done":0,"agree":0,"total":0,"mism":[]}
def call(p,maxtok=1600):
    data=json.dumps({"model":"gpt-oss:20b","messages":[{"role":"user","content":p}],"stream":False,"temperature":0,"max_tokens":maxtok}).encode()
    req=urllib.request.Request(BASE+"/v1/chat/completions",data=data,headers={"Content-Type":"application/json"})
    with urllib.request.urlopen(req,timeout=40) as r: return json.load(r)["choices"][0]["message"]["content"]
start=state["done"]; chunk=samp[start:start+4]
if chunk:
    items=[f'[{i}] CELL {ck}\nREASONING:{rsn[:180]}\nOPTIONS:{[s["name"] for s in CB[ck]]}' for i,(ck,rsn,_) in enumerate(chunk)]
    p=('For each item pick the single best sub-type NAME from its OPTIONS given REASONING. Reply ONLY JSON [{"i":0,"name":"..."}].\n\n'+"\n\n".join(items))
    res=json.loads(re.search(r'\[.*\]',call(p),re.S).group())
    for d in res:
        i=d['i']; ck,rsn,assigned=chunk[i]; state["total"]+=1
        if d['name']==assigned: state["agree"]+=1
        else: state["mism"].append([ck,assigned,d['name']])
    state["done"]=start+len(chunk)
    json.dump(state,open(ST,'w'))
print(f"done {state['done']}/{len(samp)}  agree {state['agree']}/{state['total']}")
