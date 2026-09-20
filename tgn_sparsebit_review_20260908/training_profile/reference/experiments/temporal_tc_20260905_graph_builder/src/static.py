"""Owned-function SASS normalization, resource and spill gate."""
import hashlib,json,re,sys
from pathlib import Path
p=Path(sys.argv[1]);res=(p/'resources.txt').read_text();sass=(p/'sass.txt').read_text();build=(p/'build.stderr').read_text()
regs=list(map(int,re.findall(r'REG:(\d+)',res)))
local=list(map(int,re.findall(r'LOCAL:(\d+)',res)))
shared=list(map(int,re.findall(r'SHARED:(\d+)',res)))
assert len(regs)==4 and max(regs)<=64,(regs,res)
assert local and max(local)==0
assert max(shared or [0])<=128
assert not re.search(r'\b(?:LDL|STL|HMMA|IMMA|BMMA|DMMA|QMMA|MMA|WGMMA|TCGEN\w*)\b',sass)
frames=re.findall(r'(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads',build)
assert len(frames)==4 and all(set(f)=={'0'} for f in frames),frames
functions={};name=None
for line in sass.splitlines():
    m=re.search(r'Function\s*:\s*(\S+)',line)
    if m:name=m[1];functions[name]=[]
    elif name:
        m=re.match(r'\s*/\*[0-9a-fA-F]+\*/\s*(.*?)\s*;\s*/\*',line)
        if m:functions[name].append(m[1].strip()+';')
assert len(functions)==4 and all(functions.values())
ledger=[]
for name,lines in functions.items():
    text='\n'.join(lines)+'\n';(p/(name+'.normalized.sass')).write_text(text)
    ledger.append(dict(function=name,instructions=len(lines),normalized_sha256=hashlib.sha256(text.encode()).hexdigest()))
out=dict(all_pass=True,registers=regs,max_registers=max(regs),local=local,shared=shared,frames=frames,
         no_tensor_mma=True,no_local_load_store=True,normalization='v1: remove PC/encoding/header, preserve instruction predicates and operands',
         binary_sha256=hashlib.sha256((p/'graph_builder.so').read_bytes()).hexdigest(),functions=ledger)
(p/'static.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
