This bounded experiment suppresses autograd recording only while TGNMemory advances its persistent state. The unchanged `_update_memory` is called inside `torch.no_grad()`; it still recomputes messages, aggregation, GRU and timestamps, and performs both writes. The prediction-time Memory read, GNN, source decoder, backward, optimizer, message-store update and end-of-step detach remain unchanged.

Remote host: `pro6000-8`. New root: `/home/data/wangxuran/tncn_state_nograd_20260909`. Prior experiment roots and original project are read only. `src/candidate.py` imports the preceding Memory implementation and direct-CSR bridge. On a fresh Trainer, `prepare(tr, variant)` selects `direct`, `direct_nograd`, `batch_store` or `batch_nograd`; call `i.b.step(tr, index, 'direct')` for each real training step. Do not repurpose a candidate-patched Trainer as a baseline by merely changing an argument. The runner creates separate objects per variant.

GPU3 is selected by UUID `GPU-a149f5af-55ab-ce33-8d3d-371a7ae61dd2`, with the existing GPU3 flock, live idle precheck and process guard. Interpreter `/home/data/wangxuran/isaacsim6/env/bin/python`, dependencies `/home/data/wangxuran/tncn_training_profile_20260908/deps`, PyTorch 2.11.0+cu130. Keep the established FP32, deterministic-algorithm, TF32-off, CUBLAS `:4096:8`, uninitialized-fill-off and one-thread settings. The supervisor records inventories, timestamps, exit status and actual code hashes.

Run in the remote experiment directory with unused output names:

```bash
python3 src/supervise.py --run qualify_new --stage qualify
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 /home/data/wangxuran/isaacsim6/env/bin/python src/verify_raw.py --run qualify_new
python3 src/supervise.py --run timing_new --stage timing --qualification /home/data/wangxuran/tncn_state_nograd_20260909/output/qualify_new/artifacts/qualification.json
python3 src/supervise.py --run profiles_new --stage profiles --qualification /home/data/wangxuran/tncn_state_nograd_20260909/output/qualify_new/artifacts/qualification.json
```

Qualification restores the same 12 frozen historical checkpoints and executes eight continuous steps per variant. Each step saves 119 fields: the preceding 107 checked model/optimizer/output fields, ten logically packed message-store fields and CPU/CUDA RNG. It compares actual bytes and dtype with the preceding direct baseline, and the original 107 fields with the source archive. Independent CPU/NumPy verification reopens every tensor archive with CUDA hidden, recomputes comparisons and matches the logged results. Failures remain in the output and cannot enter performance-eligible pairs.

Timing performs nine randomized paired rounds per window, all four eligible variants, eight complete steps per interval: 432 intervals and 3456 steps. Warmup, checkpoint/RNG restore and GC occur outside timing. Actual allocation, synchronization, integer generation, CSR consumption and all training work remain paid. Direct/direct_nograd and batch_store/batch_nograd isolate no-grad. Direct/batch_nograd measures the combination in this same session, not a multiplication of historical speedups.

Detailed CUDA-event replays are separate from timing and verify 115 persistent fields at the final state. In an additional diagnostic step, the observer inspects `memory.requires_grad`, its `grad_fn`, and the nodes reachable through `next_functions` immediately after `_update_memory`. The observer is never installed in timing or phase measurements. It uses no hooks, and each diagnostic step is checked against the complete 119-field archive. Its outer grad mode remains enabled; graph recording is disabled only inside the candidate's bound call. Graph removal does not imply that state recomputation arithmetic disappeared.

Completed output names are `qualify_v1`, `timing_v1` and `profiles_v1`. Their logs, independent audit and inventories are mirrored under local `evidence/`. The 48 large qualification tensor archives remain in the remote `output/qualify_v1/artifacts/`, with byte sizes and hashes in the independent audit and `remote_manifest.json`. Regenerate local tables and the figure with:

```bash
python src/analyze.py
python src/plot.py
```

`analysis/summary.json` contains every paired ratio, interval, phase and mechanism observation. The intervals use 20,000 bootstrap samples of the nine paired ratios per window and are descriptive within one session. The primary report is [验证结果.md](验证结果.md). Previous manifests, transfer verification, closure checks and `MANIFEST.sha256` preserve provenance without rewriting earlier evidence.

Batch variants inherit the preceding split-view message-cache retention behavior. No-grad does not fix or worsen that storage layout. The qualification is for eight-step windows from keeper-generated historical checkpoints; there is no full-epoch quality, long-run cache bound, original-TGN entry point or Tensor Core claim.
