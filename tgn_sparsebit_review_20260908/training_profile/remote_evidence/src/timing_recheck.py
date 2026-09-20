"""Timing-only replication of full_v5 checkpoints; prior failures stay frozen."""
import argparse
import gc
import json
import random
import time
from pathlib import Path
import torch
import profile_training as p

parser = argparse.ArgumentParser()
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--dataset', default='both')
parser.add_argument('--batch', default='both')
args = parser.parse_args()
args.output.mkdir(exist_ok=False)
base = p.EXP/'output/full_v5/artifacts'
torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)
torch.utils.deterministic.fill_uninitialized_memory = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
p.ref.guard(initial=True)
torch.zeros(1, device='cuda')
p.ref.guard()
order_rng = random.Random(20260908)
p.emit('opening', mode='timing_recheck', rounds=9, gc_collect_before_each=True,
       gc_enabled_during_timing=gc.isenabled(), original='full_v5')
for dataset in ['wikipedia', 'college']:
    for batch in [32, 200]:
        tr = p.Trainer(dataset, batch)
        for start in ([128, 256, 512] if batch == 32 else [20, 40, 80]):
            case = f'{dataset}_b{batch}_s{start}'
            checkpoint = torch.load(base/(case+'_checkpoint.pt'), weights_only=True)
            old = torch.load(base/(case+'_keeper_checks.pt'), map_location='cpu', weights_only=True)
            tr.restore(checkpoint)
            captures, checks = [], []
            for j, idx in enumerate(range(start, start+8)):
                r = tr.step(idx, capture=True, check=True)
                captures.append(r['cache'])
                checks.append(p.compare(r['check'], old[j]))
            p.emit('checkpoint_replay', case=case, pass_all=all(c['pass_all'] for c in checks), checks=checks)
            assert all(c['pass_all'] for c in checks), 'recheck changed canonical trajectory'
            for rnd in range(9):
                order = order_rng.sample(['keeper', 'free_graph', 'free_relations'], 3)
                for variant in order:
                    tr.restore(checkpoint)
                    gc.collect()  # remove checkpoint construction debt outside timing
                    torch.cuda.synchronize()
                    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    a.record()
                    begin = time.perf_counter_ns()
                    losses = []
                    for j, idx in enumerate(range(start, start+8)):
                        losses.append(tr.step(idx, variant, captures[j])['loss'])
                    b.record()
                    torch.cuda.synchronize()
                    p.emit('timing', case=case, variant=variant, round=rnd, order=order, steps=8,
                           wall_ms=(time.perf_counter_ns()-begin)/1e6, timeline_ms=a.elapsed_time(b),
                           loss_first=losses[0], loss_last=losses[-1])
            p.ref.guard()
            del captures, checkpoint, old
        del tr
        torch.cuda.empty_cache()
p.emit('complete', apps=p.ref.guard())
