This is an idealized opportunity measurement for message-cache reads on the qualified batched-store direct-CSR TNCN path. It is not a deployable implementation or a mathematical hard upper bound.

New remote root on `pro6000-8`: `/home/data/wangxuran/tncn_cache_read_oracle_20260909`. The actual device is NVIDIA RTX PRO 6000 Blackwell Server Edition, GPU3 UUID `GPU-a149f5af-55ab-ce33-8d3d-371a7ae61dd2`. Batch sizes are 32 and 200; these are not GPU model names. Keep the established GPU3 flock/idle/process guards, original sources and dependencies, PyTorch 2.11.0+cu130, FP32, deterministic algorithms, TF32 off, CUBLAS `:4096:8`, one CPU thread and uninitialized-memory filling off. All prior projects and results remain read only.

`src/read_oracle.py` copies the literal `_compute_msg` method through AST. The source prefix ends after four `torch.cat` calls. Capture inserts only an observer after this prefix, saving source/destination node IDs, timestamps and raw message features, with selected n_id and direction for audit. All four cached fields must have `requires_grad=False` and contiguous storage. There are four calls per step: source/destination for prediction Memory read and for state advance.

The oracle substitutes resident recorded fields for that prefix. The source tail remains literal: calculate time differences from current `last_update`, apply the learned time encoder, gather current `memory`, and compose messages. Aggregation, GRU, state writeback, real batched message-store writes, neighborhood, source decoder, backward, Adam and detach all still execute. No learned output or gradient is cached. No no-grad state-update variant is used.

On a fresh Trainer, `prepare(tr, 'batch_store', tape)` selects the paid baseline; `prepare(tr, 'free_cache_reads', tape)` selects the oracle. Each step goes through `step(tr, index, start, tape_if_oracle)` to select and count the four records. Qualification verifies actual n_id/direction; timing omits tensor equality checks but pays cursor/reference bookkeeping and checks call counts. Both arms retain the exact same tape tensors for each window. Preloading and recording are excluded from time, so this comparison cannot be presented as an implementable acceleration. Reused buffers and changed allocation/cache behavior also prevent a formal hard-bound claim.

Run from the remote root with unused output names:

```bash
python3 src/supervise.py --run qualify_new --stage qualify
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 /home/data/wangxuran/isaacsim6/env/bin/python src/verify_raw.py --run qualify_new
python3 src/supervise.py --run timing_new --stage timing --qualification /home/data/wangxuran/tncn_cache_read_oracle_20260909/output/qualify_new/artifacts/qualification.json
python3 src/supervise.py --run profiles_new --stage profiles --qualification /home/data/wangxuran/tncn_cache_read_oracle_20260909/output/qualify_new/artifacts/qualification.json
```

Qualification runs capture, plain paid baseline and oracle from the same 12 checkpoints, eight continuous steps each. Every result compares 119 fields byte for byte with the preceding direct archive and the original 107 fields with source. Raw checks and captured records are saved. Independent CPU/NumPy verification reconstructs each recorded prefix from the checkpoint's per-node stores at the first step, or preceding archived logical stores for subsequent steps, indexing selected nodes in order and concatenating all four fields. It also recomputes every training comparison, matches the log and hashes all tensor files. The audit completes before timing; all gates are required.

Timing uses nine randomized paired rounds per window, eight full steps per interval: 216 intervals, 1728 steps. Warmup, checkpoint/RNG restore, GC and tape load are outside timing. Detailed event phases run separately and verify 115 persistent fields at each window end. Prefix call counts show the removed work; tape statistics disclose the precomputation data volume and retained storage.

Completed run names are `qualify_v2`, `timing_v1`, and `profiles_v1`. `qualify_v1` aborted on a module-name import collision before training; its logs and exact original scripts remain. Only the module import name and provenance path were corrected; see [ATTEMPTS.md](ATTEMPTS.md). No numerical failure was replaced or tolerance relaxed.

Copy non-PT run outputs into corresponding local `evidence/` directories, then regenerate summaries and plots:

```bash
python src/analyze.py
python src/plot.py
```

The analysis uses the fixed decision threshold from [CONTRACT.md](CONTRACT.md): geometric mean of the 12 window paired-ratio medians must reach 1.10 before pursuing a complex cache-read implementation for an overall 1.10x target. It reports every window and descriptive 20,000-sample bootstrap intervals, without selecting successful shapes. Aggregate bootstrap and leave-one-window-out sensitivity are additional post-run descriptive checks; they do not change the predeclared point-estimate gate. A positive oracle result only admits further investigation and does not establish a realizable speedup.

The report is [读取上限实验.md](读取上限实验.md). Detailed timings, phases and tape statistics are in `analysis/summary.json`. Forty-eight qualification tensor archives remain remotely under `output/qualify_v2/artifacts/`, with hashes and sizes in the CPU audit and remote manifest. Previous manifests, original runtime-source checks, transfer verification, closure and the local `MANIFEST.sha256` preserve evidence.

These are eight-step windows from keeper-generated historical checkpoints. The prior batched-store view-retention limitation remains; no full-epoch quality, long-run memory bound, original-TGN entry point or Tensor Core claim is made.
