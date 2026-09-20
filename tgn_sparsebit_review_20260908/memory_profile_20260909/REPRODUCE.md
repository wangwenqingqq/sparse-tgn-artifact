This experiment splits Memory and backward costs on the source-qualified direct-CSR TNCN harness. All prior projects and evidence are read only. The experiment directory on `pro6000-8` is `/home/data/wangxuran/tncn_memory_profile_20260909`.

`src/instrument.py` imports the preceding direct-CSR bridge and the original full training harness. `prepare(tr, 'batch_store', 'none')` installs the tested batched message-store probe; call `i.b.step(tr, index, 'direct')` to run the normal whole training step and reset the shared per-step graph. `prepare(tr, 'direct', 'none')` uses the original store method on a fresh Trainer. Do not reuse a previously patched candidate Trainer as a baseline without restoring its original method. The runner uses separate Trainer objects per variant.

The probe keeps sorting, node and message order, and all logical cache values. It gathers the four fields once per direction and stores split views in the per-node dictionaries. These fixtures have constant raw features with `requires_grad=False`. Views retain their entire parent batch; `expanded_check` reports logical bytes and deduplicated `untyped_storage().nbytes()`, which excludes Python overhead and allocator reservation. The eight-step result is not a long-run storage bound.

The supervisor uses GPU3, UUID `GPU-a149f5af-55ab-ce33-8d3d-371a7ae61dd2`, the existing `/home/data/wangxuran/.locks/tgn_sptc_gpu3.lock`, an idle precheck and the existing live process guard. Interpreter: `/home/data/wangxuran/isaacsim6/env/bin/python`; dependencies: `/home/data/wangxuran/tncn_training_profile_20260908/deps`. PyTorch 2.11.0+cu130, FP32, deterministic algorithms on, uninitialized-memory filling off, TF32 off, CUBLAS workspace `:4096:8`, one CPU thread per math library. The supervisor records GPU/process inventories before and after every run and hashes the actual runner, instrumentation and imported harness sources.

Run from the new remote experiment root, always using unused output names:

```bash
python3 src/supervise.py --run qualify_new --stage qualify
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 /home/data/wangxuran/isaacsim6/env/bin/python src/verify_raw.py --run qualify_new
python3 src/supervise.py --run timing_new --stage timing --qualification /home/data/wangxuran/tncn_memory_profile_20260909/output/qualify_new/artifacts/qualification.json
python3 src/supervise.py --run profiles_new --stage profiles --qualification /home/data/wangxuran/tncn_memory_profile_20260909/output/qualify_new/artifacts/qualification.json
```

Qualification restores the same 12 frozen checkpoints (two datasets × two batch sizes × three history positions), then executes eight consecutive full steps per variant. It saves 24 raw tensor archives. Baseline and candidate retain the prior 107 checked fields, add ten logically packed message-store fields and two RNG fields. The independent CPU/NumPy audit compares dtype and actual tensor bytes including signed zero, and compares each implementation's 107 fields with archived source output. It runs with CUDA hidden and finishes before timing. Any failure remains in the output and prevents timing eligibility.

Timing includes nine randomized paired rounds per window, eight full steps per interval, 216 intervals and 1728 timed steps. Both variants have an eight-step warmup; checkpoint/RNG restore and GC occur outside each interval. CUDA event and wall times are saved. No detailed Memory instrumentation or tensor-state checks occur within these intervals. Normal Python null-context overhead remains common, and the candidate retains its small wrapper/context overhead.

Profiles are separate from timing. `events` mode records nested CPU/CUDA-event phases for eight steps in every window, and verifies the final persistent state (115 fields). `trace` mode records the first step of the four middle windows: B32 step256 and B200 step40, for both datasets and both variants. The source step runs with `check=False`, preserving its normal autograd graph without `z.retain_grad`; post-trace state checks occur outside the profiler. Restore is followed by CUDA synchronization before profiling. Statistics are cleared after warmup. The eight trace state checks are also byte identical.

`src/analyze_traces.py` uses unique FunctionEvent object UIDs for parentage; external event IDs alone are not unique because CUDA runtime records reuse CPU-op IDs. GPU activity association uses Chrome `cpu_op`/`user_annotation` IDs, with a Chrome CPU containment-tree fallback for nested operations pruned by FunctionEvent parsing. Shared-event owners agree across both trees. Forward sequence number and thread associate autograd nodes; six reused-sequence ambiguities in each trace are resolved against actual `fwdbwd` flow endpoints. Nineteen leaf/unsequenced nodes are kept separate. Unknown native activities are not forced into any owner. Only CUDA kernel/memcpy/memset activities enter durations; GPU annotations are excluded. The ROI runs from the first training phase to the end of detach/loss-read. No profiler gap or static occupancy estimate is treated as production hardware utilization.

`pilot_v1` preserves the initial one-step exploration, including its raw tensor checks, Chrome traces and the exact `src/pilot_v1.py` serializer. Its warmup statistics and external-ID parentage were inadequate for final attribution. The final `pilot.py` serializer adds unique UIDs, and full runs fix pre-trace synchronization, statistics reset and post-trace checks. Pilot output is not pooled into final counts or timing. `analyze_traces_v1.py` preserves the initial conservative offline parser before the Chrome-tree/flow refinement; final analyses use `analyze_traces.py` only. Raw GPU traces were not rerun for that parser refinement.

Completed GPU runs: `qualify_v1`, `timing_v1`, `profiles_v1`, each exit code zero. Copy their non-PT output to corresponding local `evidence/` directories, then regenerate tables and figures locally:

```bash
python src/analyze.py
python src/analyze_traces.py
python src/findings.py
python src/plot.py
```

The main deliverable is [性能瓶颈.md](性能瓶颈.md); [analysis/summary.json](analysis/summary.json) holds paired timing and all storage observations, [analysis/findings.json](analysis/findings.json) holds grouped phases and backward work, and [analysis/trace_summary.json](analysis/trace_summary.json) records coverage. The 24 qualification PT archives remain on the remote host; sizes and hashes are in `remote_manifest.json` and the CPU audit. All eight final Chrome traces and serialized events are available locally. `previous_profile.json`, `previous_diagnosis.json` and `previous_csr.json` pin prior remote evidence. `src/close_remote.py` verifies these manifests and the original runtime sources before creating the current remote manifest. Local transfer verification, closure checks and `MANIFEST.sha256` make the deliverable independently inspectable.

These are conditional eight-step windows from keeper-generated historical checkpoints. There is no full-epoch accuracy, original-TGN entry-point or Tensor Core result.
