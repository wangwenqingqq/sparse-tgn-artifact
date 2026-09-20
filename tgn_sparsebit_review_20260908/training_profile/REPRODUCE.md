# Reproduction and evidence

Local deliverable: this directory. Remote experiment:
`pro6000-8:/home/data/wangxuran/tncn_training_profile_20260908`.
Original project: `/home/data/wangxuran/factor_tgn_sptc_20260902/project`.
Existing original source files and native libraries were not edited.

The runtime is `/home/data/wangxuran/isaacsim6/env/bin/python` (Python 3.12,
Torch 2.11.0+cu130). Extra PyG dependencies were installed into this experiment's
`deps` directory with `pip --target --no-deps`; no existing environment package
was replaced. Exact versions and package provenance are in the install logs,
run opening records, and `remote_manifest.json`. Installation followed the
[official PyG wheel compatibility table](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)
and [Torch 2.11/CUDA13 wheel index](https://data.pyg.org/whl/torch-2.11.0%2Bcu130.html).

The executed full profiling script is preserved in `remote_evidence/src`.
Current `src` additionally contains local analysis and later audit scripts.

CPU source/checkpoint/integer-formula checks, executed before full GPU profiling:

```bash
cd /home/data/wangxuran/factor_tgn_sptc_20260902/project
env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH=/home/data/wangxuran/tncn_training_profile_20260908/deps \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /home/data/wangxuran/isaacsim6/env/bin/python \
  /home/data/wangxuran/tncn_training_profile_20260908/src/cpu_check.py
```

GPU runs must use fresh output names. The supervisor acquires the existing
`/home/data/wangxuran/.locks/tgn_sptc_gpu2.lock`, requires the target UUID to be
idle, exports the GPU/threads/determinism environment, and records before/after
GPU processes. The child checks for other processes on the GPU during work.
These commands show the executed protocol with new output names:

```bash
python3 /home/data/wangxuran/tncn_training_profile_20260908/src/supervise.py \
  --run new_smoke --smoke --dataset both --batch both
python3 /home/data/wangxuran/tncn_training_profile_20260908/src/supervise.py \
  --run new_full --dataset both --batch both
```

`timing_recheck.py` intentionally reloads archived `output/full_v5/artifacts`
checkpoints. It rechecks 96 keeper steps, then runs nine paired timing rounds:

```bash
python3 /home/data/wangxuran/tncn_training_profile_20260908/src/recheck_supervise.py \
  --run new_timing_recheck
```

The separate raw audit runs on CPU and uses Torch only to deserialize the
trusted, experiment-generated tensor files; comparisons use NumPy, including
FP64 tolerance arithmetic. Its output must not exist beforehand:

```bash
env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /home/data/wangxuran/isaacsim6/env/bin/python \
  /home/data/wangxuran/tncn_training_profile_20260908/raw_audit.py \
  --input /home/data/wangxuran/tncn_training_profile_20260908/output/full_v5/artifacts \
  --output /home/data/wangxuran/tncn_training_profile_20260908/output/full_v5/new_raw_audit.json
```

Raw checkpoints and per-step full model/optimizer/gradient/memory tensors occupy
3,974,810,576 bytes in `output/full_v5/artifacts` on the remote host. They remain
there and have not been copied into this local report bundle. The full raw-audit
JSON lists hashes and sizes for all 60 per-path raw archives; `remote_manifest.json`
also indexes checkpoints, other runs, source snapshots, and dependencies.
The local `remote_evidence` directory includes stdout JSONL, stderr, GPU snapshots,
source versions, analysis, and audit results. A local verification record checks
downloaded file hashes against the remote manifest.

Local analysis:

```bash
python src/analyze.py --input remote_evidence/output/full_v5/stdout.jsonl --output new_analysis
python src/analyze_recheck.py
python src/plot_phases.py
```

The latter two write derived files in `analysis`; raw evidence stays immutable.
Primary tables use paired wall-clock ratios per window. CUDA event timings are
kept separately. Do not pool the first and second timing protocols into one
confidence interval. GPU clock/host-load variability is not eliminated by the
within-session bootstrap.

Attempts retained:

| Attempt | Outcome |
|---|---|
| smoke_v1 | Idle check refused launch because another job occupied GPU2 |
| smoke_v3 | Integration initialization failure: CPU message-store tensors had not been reset after model transfer |
| smoke_v4 | Reset fixed; default nondeterministic execution changed neighbor caches and subsequent training states |
| smoke_v5 | Four configurations passed after the declared deterministic amendment |
| full_v5 | All 12 windows measured; source equivalence passed 86/96 steps, keeper and both oracles each passed 96/96 |
| raw_audit | Independently reproduced all 384 comparison outcomes; 374 pass, 10 fail |
| timing_recheck_v5 | All archived keeper states reproduced; nine paired timing rounds per window completed |

No new Tensor Core kernel was developed, no training quality score was measured,
and no failed comparison was removed or assigned a looser tolerance.
