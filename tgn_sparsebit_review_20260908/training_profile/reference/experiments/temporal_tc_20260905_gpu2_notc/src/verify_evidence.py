"""Read-only content/provenance gate; does not initialize CUDA or rerun timings."""
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];PROJECT=ROOT.parents[1]
UUID='GPU-16f27f5a-dfcd-48e0-bb39-bebbe4009245'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
assert (ROOT/'CONTRACT.md').read_bytes()==(ROOT/'logs/contract_frozen.md').read_bytes()
input_rows=(ROOT/'logs/input_manifest.sha256').read_text().splitlines()
for line in input_rows:
    h,p=line.split('  ',1);assert sha(PROJECT/p)==h,p
phases=['source_verify','build','cpu_build','correct','static','memcheck','racecheck','synccheck','initcheck','stress','pair_check']+['bench'+str(i) for i in range(5)]
provenance_checks=0;raw_hashes=0
for run,manifest in [('gpu1','remote_manifest.sha256'),('gpu2','remote_manifest2.sha256')]:
    p=ROOT/'output'/run
    assert (ROOT/'output'/f'{run}.launcher.exit').read_text().strip()=='0'
    assert 'COMPLETE ' in (p/'status.txt').read_text()
    lines=(ROOT/'output'/manifest).read_text().splitlines();assert len(lines)==176
    for line in lines:
        h,name=line.split('  ',1);assert sha(ROOT/name)==h,name;raw_hashes+=1
    for phase in phases:
        assert (p/f'{phase}.exit').read_text().strip()=='0',phase
        for side in ['before','after']:
            assert UUID not in (p/f'{phase}.{side}.apps.csv').read_text(),(phase,side)
    for phase in ['correct','memcheck','racecheck','synccheck','initcheck','stress','pair_check']+['bench'+str(i) for i in range(5)]:
        d=read(p/phase/'provenance.json')
        assert d['cuda_visible_devices']==UUID and d['binary_sha256']==sha(p/'native_gpu.so') and d['reference_binary_sha256']==sha(p/'reference_cpu.so')
        assert d['contract_sha256']==sha(ROOT/'CONTRACT.md')
        if run=='gpu2':assert d['amendment_sha256']==sha(ROOT/'COMPARATOR_REPAIR.md')
        for name,h in d['files'].items():assert sha(ROOT/name)==h,name;provenance_checks+=1
        ref=ROOT.parent/'temporal_tc_20260905_native_notc_control'
        for name,h in d['reference_files'].items():assert sha(ref/name)==h,name;provenance_checks+=1
        author=ROOT.parent/'temporal_tc_20260905_cooccurrence_gate0/sources/TNCN'
        for name,h in d['source_hashes'].items():assert sha(author/name)==h,name;provenance_checks+=1
        assert len(d['fixtures'])==36
        for name,h in d['fixtures'].items():assert sha(ROOT.parent/'temporal_tc_20260905_tncn_mode2_gate0/output/census1'/name)==h,name
        op=read(p/phase/'opening.json');assert len(op['apps'])==1 and str(op['pid']) in op['apps'][0] and UUID in op['apps'][0]
    c=read(p/'correct/summary.json');assert c['all_pass'] and (c['cases'],c['coefficient_checks'],c['decoder_comparisons'])==(173,346,1528)
    assert read(p/'pair_check/summary.json')['comparisons']==90 and read(p/'pair_check/summary.json')['all_pass']
    for tool in ['memcheck','racecheck','synccheck','initcheck']:
        s=(p/f'{tool}.stdout').read_text()
        assert 'PASS 24' in s
        assert ('0 hazards displayed (0 errors, 0 warnings)' in s) if tool=='racecheck' else ('ERROR SUMMARY: 0 errors' in s)
        assert read(p/tool/'summary.json')['all_pass']
    assert read(p/'stress/summary.json')['cases']==200 and read(p/'stress/summary.json')['all_pass']
    st=read(p/'static.stdout');assert st['pass_all'] and st['max_registers']==43 and st['no_tensor_mma']
    assert st['sass_sha256']==sha(p/'sass.txt') and st['binary_sha256']==sha(p/'native_gpu.so')
    raw=[]
    for i in range(5):
        s=read(p/f'bench{i}/samples.json');assert len(s)==900 and sum(not r['warmup'] for r in s)==630
        assert all(r['eligible'] and r['wall_us']>0 and r['event_us']>0 and r['process']==i for r in s);raw+=s
    assert len(raw)==4500
assert (ROOT/'output/gpu1/sass.txt').read_bytes()==(ROOT/'output/gpu2/sass.txt').read_bytes()
assert (ROOT/'output/gpu1/resources.txt').read_bytes()==(ROOT/'output/gpu2/resources.txt').read_bytes()
latest=read(ROOT/'output/analysis2/summary.json');assert latest['retained']==3150 and latest['warmups']==1350
for a in latest['aggregate'][:2]:assert a['ordinary_prototype_gate'] and a['per_shape_gate_pass']==18 and a['process_wins']==5
lib=next(a for a in latest['aggregate'] if a['comparison']=='native-vs-library');assert lib['paired_geomean_ratio']<1 and len(lib['regressing_shape_means'])==16
closing=(ROOT/'logs/closing_live.txt').read_text();assert 'GPU2_LOCK_RELEASED' in closing and 'CLOSING_VERIFIED' in closing
report=dict(pass_all=True,input_manifest_files=len(input_rows),downloaded_file_hashes=raw_hashes,execution_source_reference_hash_checks=provenance_checks,
            campaigns_retained=2,latest_decoder_comparisons=1528,latest_pair_comparisons=90,latest_sanitizer_cases_per_tool=24,latest_stress_cases=200,
            latest_retained_samples=3150,latest_warmups=1350,historical_retained_samples=3150,owned_device_sass_unchanged=True,
            no_joint_TC_result=True,no_task_quality_result=True,gpu2_lock_released=True)
print(json.dumps(report,indent=2))
