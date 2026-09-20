The direct CSR bridge replaces only coefficient-format conversion in the
source-compatible TNCN training harness. Existing native count/pack kernels,
model precision, fourteen source SpMM calls and positive/negative autograd
organization are retained. No original project or prior experiment is edited.

Remote host: `pro6000-8`. Experiment root:
`/home/data/wangxuran/tncn_compatible_csr_20260908`. The supervisor uses GPU3,
UUID `GPU-a149f5af-55ab-ce33-8d3d-371a7ae61dd2`, with the existing GPU3 flock,
live idle checks and target-GPU process guards. GPU2 was occupied. Measurements
compare every arm on GPU3 and are not pooled with the preceding GPU2 session.

Run from that new remote experiment directory, choosing unused output names:

```bash
python3 src/supervise.py --run qualify_new --stage qualify
python3 src/supervise.py --run timing_new --stage timing --qualification /home/data/wangxuran/tncn_compatible_csr_20260908/output/qualify_new/artifacts/qualification.json
```

Python and packages are unchanged from the previous experiments: interpreter
`/home/data/wangxuran/isaacsim6/env/bin/python`, prior training profile's `deps`,
PyTorch 2.11.0+cu130 and torch_sparse 0.6.18+pt211cu130. The supervisor sets the
original deterministic, TF32-off and CUBLAS workspace settings. Each run restores
the same frozen checkpoints and RNG states. The 12 windows contain eight
continuous full training steps each; there is no full-epoch quality claim.

`bridge.install(tr, 'direct')`, followed by
`bridge.step(tr, index, 'direct')`, is the tested integration point for
the prior Trainer. `bridge.step` resets the per-step shared graph. The `direct`
variant retains the original producer's NNZ scalar read and adds a batched
boundary read. `fused_bounds` replaces that scalar read with eight offsets per
query chunk. Both create seven canonical CSR consumers without a dense C.
The implementation is bounded to mode2 without time decay, n <= 2048 and native
query chunks <= 64. Validation here covers the original B32/B200, D100 fixtures.

Qualification checks all four variants, including the old keeper as a diagnostic
comparator. Both new candidates additionally save input/output CSR data and
compare row pointers, columns and values with the previous dense conversion.
Full-state acceptance is byte equality with all archived source fields; the
old 2e-4 absolute/relative tolerance is also reported without alteration.

Run independent CPU verification after qualification, with CUDA hidden and one
CPU thread per math library:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 /home/data/wangxuran/isaacsim6/env/bin/python src/verify_raw.py --run qualify_new
```

It loads actual tensor archives on CPU, recomputes byte comparisons and
independently reconstructs dense integer coefficients from the input CSR chunks
to check the output CSR. It also validates pointer bounds, strictly sorted
columns and nonzero values. The CPU audit completed before performance timing.

Timing uses nine random paired rounds per window. All allocation, native
production, host synchronization, format conversion and full training updates
are paid. Restore and garbage collection occur outside each interval. Separately
instrumented replays divide graph building, integer production, conversion and
SpMM. These phase times include CPU submission gaps. Direct CSR defers row-index
generation to SpMM, so compare conversion and aggregation together as well as
the whole training step.

Archived runs are `qualify_v1` and `timing_v1`. Local `evidence/` holds their
logs, checks and GPU inventories. The 72 large raw tensor archives remain on
the remote host under `output/qualify_v1/artifacts/`; hashes and byte counts are
in `remote_manifest.json` and the CPU audit. `src/analyze.py` reconstructs the
paired summaries from local logs. `src/plot.py` produces the standalone figure.

`previous_profile.json` and `previous_diagnosis.json` preserve the earlier remote
manifests. `src/close_remote.py` verifies their listed files and runtime sources
remain unchanged. `transfer_verification.json` records local evidence hash
checks; `MANIFEST.sha256` covers the local deliverable excluding itself and
Python cache files. No prior manifest is rewritten.

Both variants were tested explicitly. After timing, the recommended default was
changed to `direct`; no computational branch changed. `src/bridge_v1.py`
preserves the exact source hash logged by the GPU runs before that default-only
change. The boundary-fusion variant remains available as an unsuccessful
additional speed optimization.
