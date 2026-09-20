# Real training-step profile, 2026-09-08

Purpose: measure remaining full-step opportunity before another Tensor Core kernel.
New experiment only; archived sources/binaries remain immutable.

Workload: pinned TNCN mode 2, two-hop LastNeighborLoader k=10, memory/time/embedding
dimension 100, hidden decoder 256, IdentityMessage, LastAggregator,
GraphAttentionEmbedding with source dropout 0.1, Adam lr=1e-4, FP32 without TF32,
source-default no CN decay. Real chronological 20,000-event prefixes of JODIE
Wikipedia (172 real message features) and CollegeMsg (featureless: one constant
zero message channel). Existing Wikipedia20k archive's ID mapping is retained;
this is not the official TGB train/validation/test benchmark.

Batch sizes: 32 (previous bounded decoder regime) and 200 (source default).
Train through history with actual memory, embedding, gradients, and Adam.
Three predeclared windows start at event offsets approximately 4096,8192,16384:
B32 steps 128,256,512; B200 steps 20,40,80; eight consecutive steps per window.
No fixed random embedding input. Seed 20260908. No validation/test quality claim.

Keeper: existing paid graph builder, list producer, PyTorch CSR SpMM with its
autograd. Share graph between positive/negative targets; at B200 use query chunks
of at most 64, retaining source separate positive/negative MLP calls. Require
n<=2048 for every native call; report failures rather than silently omit shapes.
This composition at B200 needs fresh coefficient/output/dX checks.

At each window, snapshot parameters, optimizer, memory/message stores, sampler,
and RNG. An untimed keeper pass captures graph and integer coefficient plans.
Replay source and keeper and verify outputs/loss/parameters/memory/Adam against
same starting state. Use atol=rtol=2e-4; retain failures, do not widen tolerances.
Check exact integer coefficients against native torch_sparse source.

Then measure paired keeper / free_graph / free_relations, three rotated rounds,
all executing complete consecutive training steps from the same checkpoint.
Oracles use exactly captured graphs or C for each same-batch state, retaining
memory, sampling, feature aggregation/backward/update, and all other costs.
They are impossible free-preparation bounds, not deployable speedups. Timed runs
do not compare GPU tensors or save raw arrays; matching is checked separately.
Real setup/data upload/checkpoint restore excluded; batch slicing, negative
sampling, graph/relation allocations and host sync, loss extraction included.

Separate instrumented replays record per-phase host time and CUDA event
timeline. Event durations include host launch gaps and are not pure kernel time.
Instrumented total is compared with uninstrumented timing. Q and weighted S are
fused inside keeper producer; do not invent separate timing from whole-kernel
cost. Inspect/diagnose repeated work separately. Use actual oracle full-step
measurements as the main decision evidence.

GPU2 existing flock + idle check and process isolation checks. Other GPUs may
be busy; retain snapshots and restrict claims to this shared-host session.
