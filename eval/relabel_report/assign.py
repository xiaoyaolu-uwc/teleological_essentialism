import csv, json, re
csv.field_size_limit(10**7)
CB=json.load(open('eval/relabel_report/codebook_frozen.json'))
DEEP=set(k for k in CB if not k.startswith('_'))
def subtype_for(cell_key, reason):
    r=reason.lower()
    for st in CB[cell_key]:
        for p in st['patterns']:
            if p=="" or re.search(p, r):
                return st['name']
    return CB[cell_key][-1]['name']
rows=[]
with open('data/sentences_labeled.csv') as f:
    rd=csv.DictReader(f); fields=rd.fieldnames
    for r in rd: rows.append(r)
from collections import Counter, defaultdict
cellcount=defaultdict(Counter)
for r in rows:
    ck=f"{r['scan_tag']}->{r['deploy_tag']}"
    if ck in DEEP:
        st=subtype_for(ck, r['deploy_reasoning'])
        r['change_subtype']=st
        cellcount[ck][st]+=1
    else:
        r['change_subtype']=''
# write annotated csv
with open('eval/relabel_report/changes_annotated.csv','w',newline='') as f:
    w=csv.DictWriter(f, fieldnames=fields+['change_subtype']); w.writeheader()
    for r in rows: w.writerow(r)
# report distribution; flag general-bucket share
print("SUB-TYPE DISTRIBUTION (deep cells)")
for ck in CB:
    if ck.startswith('_'): continue
    tot=sum(cellcount[ck].values()); 
    print(f"\n== {ck}  (n={tot}) ==")
    gen=CB[ck][-1]['name']
    for st in CB[ck]:
        c=cellcount[ck][st['name']]
        flag=" <-general/catchall" if st['name']==gen else ""
        print(f"  {c:>5} ({100*c/tot:>4.0f}%)  {st['name']}{flag}")
