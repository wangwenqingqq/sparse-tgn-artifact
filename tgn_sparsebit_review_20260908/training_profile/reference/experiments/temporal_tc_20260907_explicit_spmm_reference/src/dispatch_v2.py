"""Repair only the library identity filter; preserve the failed attempt."""
import hashlib,json
import transport as t
files=[t.EXP/'src/explicit_v2.py',t.EXP/'src/replay_v2.py',t.EXP/'src/dispatch_v2.py',t.EXP/'IDENTITY_AMENDMENT.md']
m=json.loads((t.EXP/'source_manifest.json').read_text());m.update({str(p.relative_to(t.ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
mp=t.EXP/'source_manifest_v2.json'
with mp.open('x') as f:f.write(json.dumps(m,indent=2)+'\n')
t.execute('upload2',f'cd {t.REMOTE} && test ! -e {t.REL}/source_manifest_v2.json && tar -xf -',t.bundle(files+[mp]))
for process in range(2):
    t.execute(f'replay_v2_{process}',t.PREFIX+f'{t.PY} -u {t.REL}/src/replay_v2.py {process}');t.fetch(f'replay_payload{process}')
