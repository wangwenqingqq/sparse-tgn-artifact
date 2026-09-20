This experiment uses the existing FlashTGN artifact as the project baseline. It contains no TNCN training imports. Read 验证与性能结果.md, CONTRACT.md and all AMENDMENT_*.md before interpreting the measurements.

Remote experiment: pro6000-8:/home/data/wangxuran/flash_tgn_rebaseline_20260909
Interpreter: /home/data/wangxuran/isaacsim6/env/bin/python
Source artifact: /home/data/wangxuran/factor_tgn_sptc_20260902/vendor/flash-tgn-artifact
Data root: /home/data/wangxuran/factor_tgn_sptc_20260902/project/experiments/factor_tgn_20260904_anchored_time_quality_wikipedia_full/data_root

The original Python environment, FlashTGN artifact, extension and data were not edited. Real scikit-learn 1.7.2 and its missing joblib/threadpoolctl dependencies were installed only into this experiment's deps/ directory; deps_install.log records this. prepare.py verifies source/data/dependency contents and writes input_manifest.json before GPU work. The manifest has no verified Git revision because the supplied artifact lacks .git metadata. Preserve and check content hashes, including the actual loaded extension.

The exact commands, start/end timestamps, GPU UUID, environment policy and exit status of every GPU invocation are in evidence/<run>/run.json and the supervisor source. GPU supervisors reject an existing output directory, non-idle GPU or foreign GPU process. A successful exit never implies passing numeric qualification. The original fullrun.py performance gate remains closed.

Typical commands, executed on pro6000-8 from any directory (use new output names to avoid overwriting evidence):

```bash
python3 /home/data/wangxuran/flash_tgn_rebaseline_20260909/src/supervise.py --run qualify_repeat --stage qualify
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 /home/data/wangxuran/isaacsim6/env/bin/python /home/data/wangxuran/flash_tgn_rebaseline_20260909/src/verify_raw.py --run qualify_repeat
python3 /home/data/wangxuran/flash_tgn_rebaseline_20260909/src/supervise_followup.py --run diagnostic_repeat --stage diagnostic_timing --rep 4 --qualification /home/data/wangxuran/flash_tgn_rebaseline_20260909/output/qualify_v1/artifacts/qualification.json
```

supervise.py retains the original GPU3 qualification environment. supervise_followup.py and supervise_trace.py use GPU0 under AMENDMENT_2. followup.py explicitly checks that the original qualification and CPU audit failed before the diagnostic timing exception; it never reports this as qualification success. The mailbox probe intentionally reads the preserved qualify_v1 first-step archives. The repaired replay uses the same original gate in isolation. Source arithmetic is copied only for the two mailbox assignment replacements in mailbox_repair.py, with AST checks; all original source files stay unchanged.

The complete retained campaign is:

- qualify_v1: 8 warmup + 4 replay steps for source A, source B, and event scopes on each of 8 configurations. verify_raw.py independently audits all raw states and sampled temporal queries on CPU.
- mailbox_cpu.py: sequential CPU mailbox reconstruction from the common archived first-step memory plus original input events.
- mailbox_probe_v1: 16 identical-input writes × original / writer-local deterministic / explicit last-position selection, all eight cases. verify_probe.py independently audits raw outputs and recorded pass/fail counts.
- qualify_repaired_v1: same full replay matrix with only explicit last-position mailbox semantics. verify_raw.py retains the original 2e-4 + 2e-4*abs(reference) gate.
- diagnostic_timing_v1/v2/v3: unmodified FlashTGN, three fresh processes, four full training epochs per case, random case order. Each case's three epoch-1 observations are retained as cold/warmup evidence; remaining nine are summarized. No runs or shapes are selected away.
- diagnostic_profiles_v1: separate full two-epoch event profiles, all eight cases. The reported phase distribution uses epoch 2; both are retained. Source dispatch counters in raw epoch records are cumulative; analysis/summary.json also computes per-epoch deltas.
- kernel_trace_v1: one original source training epoch per case, with one middle-epoch step traced after profiler warmup. Exact active step indices are in trace_complete logs. The active step synchronizes after its mailbox write solely to bound the trace. These records are not added to the timing sample.

Local analysis:

```bash
python src/analyze.py
python src/analyze_traces.py
python src/kernel_findings.py
python src/plot.py
python src/report.py
```

Paths resolve relative to each script's experiment root. analyze.py reads downloaded metadata under evidence/; analyze_traces.py reads complete Chrome traces plus CPU UID records and records correlation/ownership coverage. No new GPU training is needed to regenerate tables/figures. src/verify_training.py runs on the remote machine with CUDA hidden, verifies all 32 complete final state archives for finiteness and checks logged field counts; it does not test accuracy or quality parity.

Local evidence/ contains every run's non-tensor logs and kernel traces. Large .pt replay/final-state archives are retained remotely, hashed in the CPU audits and remote_manifest.json. Transfer verification and local MANIFEST.sha256 identify exactly which artifacts are present locally. Followup original sources/ contains the Python/CUDA sources for review; input_manifest.json also pins the extension and all data/dependencies. Prior sealed local experiments were checked independently in prior_local_verification.json.
