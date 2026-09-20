Reduce the paid format-conversion overhead of the source-compatible TNCN path.

Keep all prior experiment files read only. Reuse the 12 frozen checkpoints and
96 archived source checks: Wikipedia/CollegeMsg, B32/B200, mode2, no decay,
D100, actual memory/GNN/full Adam. Admission requires byte-identical checked
fields to the archived source in every step; also report the existing
atol=rtol=2e-4 comparator without changing it. This is conditional equivalence
from the same checkpoints, not a full-epoch quality result.

Candidates use the exact existing native integer producer and original 14
torch_sparse SpMM calls per step, with original positive/negative endpoint
autograd organization. No native kernel or model-precision changes.

1. direct: concatenate CSR channel segments and row pointers directly, without
   an O(B*N) dense coefficient tensor or dense nonzero scan. Read all channel
   boundaries in one batched host transfer per prediction call.
2. fused_bounds: same CSR conversion, but read the eight channel boundaries at
   the producer's existing NNZ/allocation synchronization, avoiding the extra
   boundary read. Count/pack native kernels and exact allocation stay paid.

Check constructed row pointers, columns and values against the previous dense
compatibility conversion on all 96 real states (two predictions each). Save
actual CSR tensors and full training-check tensors for independent CPU replay.
Preserve all failures. Only candidates passing every window are performance
eligible. Measure both eligible candidates against old dense compatibility and
old keeper in nine randomized paired rounds per window, eight complete steps
per interval. Restore checkpoint/RNG and collect GC outside timing; all work,
including allocation, host transfers, conversion and backward, stays inside.
Old keeper remains an unqualified diagnostic comparator in two windows.

Use identical determinism/environment settings and the GPU3 shared lock
(/home/data/wangxuran/.locks/tgn_sptc_gpu3.lock), with live idle and process
checks. GPU2 is occupied by another workload; do not use or disturb it. New
GPU3 results are not pooled with previous GPU2 timings.
