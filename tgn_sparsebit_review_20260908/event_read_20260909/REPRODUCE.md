This experiment measures a real event-ID message-store prototype against the previously qualified batched-store direct-CSR baseline. It uses no read oracle at runtime. The previous oracle module supplies only AST helpers for retaining the literal source method tail.

Local directory: `tgn_sparsebit_review_20260908/event_read_20260909`. Remote directory on `pro6000-8`: `/home/data/wangxuran/tncn_event_read_20260909`. Device: GPU3, NVIDIA RTX PRO 6000 Blackwell Server Edition, UUID `GPU-a149f5af-55ab-ce33-8d3d-371a7ae61dd2`. The workloads use `batch_size=32` and `batch_size=200`.

`src/event_store.py` changes the store representation to per-node NumPy event-ID slices. Reads pay `n_id.tolist()`, concatenation of selected event-ID arrays, H2D transfer and four `torch.index_select` calls on existing input fields. Reverse-direction reads swap the two ID fields. Writes pay the original GPU `src.sort()`, permutation transfer to CPU, grouping by sorted source IDs from the input mirror, and replacement of the affected index slices. This retains the exact ordering of the source GPU sort, including ties. The timer includes passing the current batch's event offset. All arithmetic after these constant fields remains the literal source tail.

The adapter assumes the frozen immutable input table and sequential training batches of this harness. It is not a general drop-in replacement for arbitrary generated messages, train/eval transitions or arbitrary checkpoint APIs. Historical checkpoint import matches all four raw fields to events preceding the checkpoint. The CPU input mirror and checkpoint import are measured and reported outside steady-step time. Their wall-clock sections can include waiting for previously queued initialization work. All per-step indexing, transfer, reads and writes are inside the timer. No dynamically selected read is prepared ahead of time.

Use a fresh Trainer and call `prepare(tr, variant, start)` before restoring the checkpoint, with variant `batch_store` or `event_index`. Call `step(tr, index)` for each full step. Diagnostics use `expanded_check`, which reconstructs the candidate's logical store from event IDs on CPU; actual GPU read prefixes are separately recorded during qualification and independently checked. These diagnostic reconstructions and captures are absent from timing.

Run with new output names from the remote root:

```bash
python3 src/supervise.py --run qualify_new --stage qualify
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 /home/data/wangxuran/isaacsim6/env/bin/python src/verify_raw.py --run qualify_new
python3 src/supervise.py --run timing_new --stage timing --qualification /home/data/wangxuran/tncn_event_read_20260909/output/qualify_new/artifacts/qualification.json
python3 src/supervise.py --run profiles_new --stage profiles --qualification /home/data/wangxuran/tncn_event_read_20260909/output/qualify_new/artifacts/qualification.json
```

The supervisor enforces GPU3 exclusivity and idle/process guards. Environment: existing PyTorch 2.11.0+cu130 and dependencies, deterministic FP32, TF32 disabled, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, one CPU thread, uninitialized-memory fill disabled. Prior source files and libraries remain read only.

Qualification compares both arms over 12 windows × 8 steps against all 119 expanded fields of the prior direct archive and the original 107 source fields. Archive 24 complete check files and 12 candidate-prefix files. The independent CPU audit recomputes every byte comparison, reconstructs all 384 prefixes from checkpoint/preceding direct logical stores, and separately gathers their fields from original dataset rows at the recorded event IDs. It asserts every ID precedes the currently executing batch. Timing requires all these checks to pass.

The timer uses nine randomized paired rounds per window, eight complete steps per interval, 216 intervals and 1728 steps. Warmup and checkpoint/RNG restore remain outside the timer. Separate event profiles verify 115 persistent fields at the end of all 24 eight-step replays. Primary ratios and descriptive bootstrap methods are fixed in [CONTRACT.md](CONTRACT.md). All windows and both batch sizes are reported; earlier gains are not multiplied into these ratios.

Copy non-PT outputs from remote `output/` to local `evidence/`, then run:

```bash
python src/analyze.py
python src/plot.py
```

Completed run names and findings are recorded in [验证结果.md](验证结果.md). Previous manifests, source hashes at each GPU opening, independent audits, transfer hashes and local/remote closure manifests preserve evidence. Raw PT files stay remotely with their hashes.

Candidate storage statistics distinguish logical raw-message bytes from active CPU index storage. The immutable GPU input table is shared with existing training inputs. Additional CPU input mirrors are reported separately. Frozen checkpoint seed indices are harness restore data and can retain their initial backing arrays; full Python/allocator/process memory and a measured full-epoch memory bound are outside the statistics. No full-epoch training-quality, original-TGN or Tensor Core result is claimed.
