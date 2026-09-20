"""Create-only upload and sequential native replays."""
import hashlib,json
import transport as t
files=sorted(p for p in t.EXP.rglob('*') if p.is_file() and p.suffix in ('.py','.md') and '__pycache__' not in p.parts)
m=json.loads((t.ROOT/'experiments/temporal_tc_20260907_cfree_forward_repair/deterministic_manifest.json').read_text())
m.update({str(p.relative_to(t.ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
mp=t.EXP/'source_manifest.json'
with mp.open('x') as f:f.write(json.dumps(m,indent=2)+'\n')
t.execute('upload1',f'cd {t.REMOTE} && test ! -e {t.REL} && mkdir {t.REL} && tar -xf - && mkdir {t.REL}/tmp {t.REL}/output',t.bundle(files+[mp]))
for process in range(2):
    t.execute(f'replay{process}',t.PREFIX+f'{t.PY} -u {t.REL}/src/replay.py {process}');t.fetch(f'replay_payload{process}')
