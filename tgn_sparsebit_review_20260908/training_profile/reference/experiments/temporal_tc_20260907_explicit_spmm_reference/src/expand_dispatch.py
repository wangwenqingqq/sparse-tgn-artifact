"""Expand only after independent witness gate; fetch failures before propagating."""
import hashlib,json
import transport as t
s=json.loads((t.EXP/'output/replay_analysis1/summary.json').read_text());assert s['expansion_gate_pass']
files=[t.EXP/'src/expand.py',t.EXP/'src/expand_dispatch.py',t.EXP/'analysis/verify_replay.py',t.EXP/'EXPANSION_ADMISSION.md',t.EXP/'output/replay_analysis1/summary.json']
m=json.loads((t.EXP/'source_manifest_v2.json').read_text());m.update({str(p.relative_to(t.ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files});mp=t.EXP/'expansion_manifest.json'
with mp.open('x') as f:f.write(json.dumps(m,indent=2)+'\n')
t.execute('expansion_upload1',f'cd {t.REMOTE} && test ! -e {t.REL}/expansion_manifest.json && tar -xf -',t.bundle(files+[mp]))
try:t.execute('expansion1',t.PREFIX+f'{t.PY} -u {t.REL}/src/expand.py correct')
finally:t.fetch('expansion_payload1')
for kind in ['memcheck','initcheck','racecheck','synccheck']:
    try:t.execute(kind+'1',t.PREFIX+f'compute-sanitizer --tool {kind} --error-exitcode 87 {t.PY} -u {t.REL}/src/expand.py {kind}')
    finally:t.fetch(kind+'_payload1')
