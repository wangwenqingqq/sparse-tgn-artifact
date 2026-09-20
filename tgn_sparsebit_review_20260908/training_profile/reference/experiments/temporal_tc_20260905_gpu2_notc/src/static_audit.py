"""Static owned-kernel evidence, not a throughput result."""
import hashlib,json,re,sys
from pathlib import Path
p=Path(sys.argv[1]);res=(p/'resources.txt').read_text();sass=(p/'sass.txt').read_text()
assert not re.search(r'\b(?:HMMA|IMMA|BMMA|DMMA|QMMA|MMA|WGMMA|TCGEN\w*)\b',sass)
regs=list(map(int,re.findall(r'REG:(\d+)',res)));local=list(map(int,re.findall(r'LOCAL:(\d+)',res)))
assert regs and max(regs)<=64,(regs,res)
assert local and max(local)==0,(local,res)
assert not re.search(r'\b(?:LDL|STL)\b',sass)
report=dict(pass_all=True,registers=regs,max_registers=max(regs),local_bytes=local,no_tensor_mma=True,no_local_load_store=True,
            max_dynamic_shared_bytes=8*2048+4*(2*64+18),source_scope='owned .so only, no vendor selected-function claim',
            binary_sha256=hashlib.sha256((p/'native_gpu.so').read_bytes()).hexdigest(),sass_sha256=hashlib.sha256(sass.encode()).hexdigest())
print(json.dumps(report,indent=2))
