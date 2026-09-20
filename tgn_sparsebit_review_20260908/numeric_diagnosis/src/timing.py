import argparse
import gc
import json
from pathlib import Path
import random
import time
import types
import diagnose as d
p, torch = d.p, d.torch


def install_oracle(tr, coefficients):
    pred = tr.model['pred']
    def get_cn(self, x, edges, q, mode, decay=False, time_info=None):
        assert mode == 2 and not decay
        cs = self._oracle_step[self._oracle_call]
        self._oracle_call += 1
        return torch.cat([p.predmod.spmm_add(c, x) for c in cs], dim=-1)
    pred.get_cn_emb = types.MethodType(get_cn, pred)
    return coefficients


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--output', required=True, type=Path)
    # Supervisor-compatible arguments are intentionally unused for this script.
    ap.add_argument('--task'); ap.add_argument('--selected'); args = ap.parse_args()
    args.output.mkdir(exist_ok=False)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = False
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    p.ref.guard(initial=True); torch.zeros(1, device='cuda'); p.ref.guard()
    p.emit('opening', script_sha256=p.sha(__file__), gpu=torch.cuda.get_device_name(), paired_rounds=9)
    rng = random.Random(20260908)
    for name in ['wikipedia', 'college']:
        for bs in [32, 200]:
            for start in ([128, 256, 512] if bs == 32 else [20, 40, 80]):
                label = f'{name}_b{bs}_s{start}'
                ck = torch.load(d.ART/(label+'_checkpoint.pt'), weights_only=False)
                expected = torch.load(d.ART/(label+'_source_replay_checks.pt'), weights_only=False)
                tr = p.Trainer(name, bs); d.install_compatible(tr); tr.restore(ck)
                captured = []
                original_mm = p.predmod.spmm_add
                def intercept(c, x):
                    captured.append(c)
                    return original_mm(c, x)
                p.predmod.spmm_add = intercept
                try:
                    for j in range(8):
                        actual = d.step(tr, start+j, 'compatible', check=True)['check']
                        cmp = p.compare(actual, expected[j]); assert cmp['pass_all'], (label, j, cmp)
                finally:
                    p.predmod.spmm_add = original_mm
                assert len(captured) == 8*14
                cs = [[captured[14*j:14*j+7], captured[14*j+7:14*j+14]] for j in range(8)]
                install_oracle(tr, cs); tr.restore(ck)
                rows, checks = [], []
                for j in range(8):
                    tr.model['pred']._oracle_step, tr.model['pred']._oracle_call = cs[j], 0
                    r = d.step(tr, start+j, 'source', check=True)['check']
                    rows.append(r); cmp = p.compare(r, expected[j]); checks.append(cmp)
                    assert cmp['pass_all'], (label, j, cmp)
                torch.save(rows, args.output/(label+'_oracle_checks.pt'))
                p.emit('oracle_check', case=label, checks=checks)
                variants = ['source', 'keeper', 'compatible', 'compatible_free_relations']
                for rnd in range(9):
                    order = variants.copy(); rng.shuffle(order)
                    for mode in order:
                        if mode == 'compatible':
                            d.install_compatible(tr)
                        elif mode == 'compatible_free_relations':
                            install_oracle(tr, cs)
                        elif hasattr(tr.model['pred'], 'get_cn_emb') and 'get_cn_emb' in tr.model['pred'].__dict__:
                            del tr.model['pred'].get_cn_emb
                        tr.restore(ck); gc.collect(); torch.cuda.synchronize()
                        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        a.record(); begin = time.perf_counter_ns()
                        for j in range(8):
                            if mode == 'compatible_free_relations':
                                tr.model['pred']._oracle_step, tr.model['pred']._oracle_call = cs[j], 0
                            d.step(tr, start+j, 'source' if mode == 'compatible_free_relations' else mode)
                        b.record(); torch.cuda.synchronize()
                        p.emit('timing', case=label, variant=mode, round=rnd, order=order, steps=8,
                               timeline_ms=a.elapsed_time(b), wall_ms=(time.perf_counter_ns()-begin)/1e6)
                del tr, captured, cs, ck, expected, rows
                gc.collect(); p.ref.guard()
    p.emit('complete', apps=p.ref.guard())


if __name__ == '__main__': main()
