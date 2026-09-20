This experiment diagnoses and repairs the numerical-trajectory gate in the
preceding real-training profile. It does not modify the old project, old
checkpoints, libraries or dependency installation.

- Host: `pro6000-8`; GPU2 UUID `GPU-16f27f5a-dfcd-48e0-bb39-bebbe4009245`.
- Experiment: `/home/data/wangxuran/tncn_numeric_diagnosis_20260908`.
- Prior experiment: `/home/data/wangxuran/tncn_training_profile_20260908`.
- Python: `/home/data/wangxuran/isaacsim6/env/bin/python`.
- Dependencies: prior experiment's `deps`; imported by the immutable prior
  `src/profile_training.py`. Runtime uses PyTorch 2.11.0+cu130,
  torch_sparse 0.6.18+pt211cu130 and the pinned source recorded in the prior
  experiment. The new implementation adds no native kernels.

The supervisors acquire the existing GPU2 lock, refuse a busy GPU, set
`CUBLAS_WORKSPACE_CONFIG=:4096:8` and deterministic algorithms, and release the
GPU after the child exits. Use unused run names: output directories refuse
overwriting. Run from the new experiment directory on the remote host:

```bash
python3 src/supervise.py --run witnesses_new --task witnesses
python3 src/supervise.py --run qualify_new --task qualify --selected compatible
python3 src/timing_supervise.py --run timing_new
```

The archived runs are `witnesses_v1`, `qualify_v1` and `timing_v1`. The first run
tests source, compatibility, two causal controls and time64; it selects the
first archived time-frequency difference mechanically. Four identical-input
decoder probes use the first and last states of the two witness windows.
The second tests only the accepted compatibility candidate on all 12 frozen
windows. The time64 candidate failed its witness gate and was not qualified.

The timing script measures all four arms in nine randomly ordered paired
rounds per window. It first captures the compatible sparse coefficients,
checks the free-relations oracle against all archived source states, and saves
the actual check tensors. `--task` and `--selected` arguments emitted by its
shared supervisor are unused; the four measured arms are fixed in `timing.py`.
All construction and conversion are paid in the ordinary compatibility arm.

CPU audit scripts initialize no CUDA context. They read the actual `.pt`
archives with `map_location='cpu'`, compare all saved fields using NumPy, and
independently recompute the FP64 decoder forward/backward. Set
`CUDA_VISIBLE_DEVICES=''`, `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1` and
`MKL_NUM_THREADS=1`, then invoke:

```bash
/home/data/wangxuran/isaacsim6/env/bin/python src/audit.py --output output/audit.json
/home/data/wangxuran/isaacsim6/env/bin/python src/audit_timing.py
```

The audit currently names the archived run directories literally. Change those
input paths when auditing new runs. `audit_v1.py` and `audit_v1.json` preserve
the first numeric-equality audit; the final `audit.py` additionally checks dtype
and element bytes, including signed zeros, before calling a field bitwise equal.
This strengthens the exactness check without changing the training tolerance.

Run `src/plot_drift.py` and `src/analyze_timing.py` locally after downloading the
JSONL evidence. They produce the standalone PNG/SVG and timing tables in
`analysis/`. Bootstrap intervals describe the nine paired ratios within this
one session; results are not pooled with the earlier profile.

Full raw tensors stay on the remote host under `output/*/artifacts/`. Local
`evidence/` includes logs, CPU audits and before/after GPU inventories, but not
the large raw tensor archives. `remote_manifest.json` records their byte sizes
and hashes. `src/close_remote.py` also verifies all previously archived files
and runtime-source hashes against the copied `prior_manifest.json`.

`transfer_verification.json` records local download checks. `MANIFEST.sha256`
covers the local report, code, figures and evidence, excluding itself and Python
cache files. Neither element counts nor repeated training steps should be read
as independent model-quality trials.
