This is a bounded FlashTGN fixed-input experiment, not an end-to-end training speed or quality result. Read CONTRACT.md and AMENDMENT_1/2.md before interpreting any number. Original full-training replay gates from flash_tgn_rebaseline_20260909 remain failed.

Remote root: pro6000-8:/home/data/wangxuran/flash_wkv_probe_20260911
Python: /home/data/wangxuran/isaacsim6/env/bin/python
Frozen source/data/dependency manifest: /home/data/wangxuran/flash_tgn_rebaseline_20260909/input_manifest.json
Source environment and isolated dependencies from the previous FlashTGN campaign are reused read-only. common.py imports its frozen core factory and changes only the new experiment's train-only output directory. Original FlashTGN sources, compiled extension and data are never edited.

Exact invocation, timestamp, UUID and process outcome are in evidence/<run>/run.json. Supervisors reject output reuse, an occupied lock or a foreign GPU process. Do not rerun an existing run name; choose a fresh name. capture.py fixes 12 attention inputs and 12 real fused-backward input sets at two positions in four source configurations. Capture takes four complete source epochs solely to obtain real intermediate states and upstream loss gradients; serialization time is not a training benchmark.

Sequence retained:

1. capture_v1, GPU0: unmodified FlashTGN at K50, batch_size200/2000, one/two layers, steps8 and floor(batches/2)+1. Input archives include full real tensors, all rows, original weights and actual upstream loss gradient.
2. qualify_v1, GPU0: initial valid-only candidate. Source A/B and captured forward are exact; six changed dense cases fail because empty queries use projected padding values. No timing of this failed candidate is performed. verify.py independently audits all outputs/gradients on CPU.
3. noise_v1 is a preflight-only lock rejection. noise_v2, GPU0, executes the same noise code and frozen inputs. Every captured CUDA backward invocation repeats16 times. verify_run.py --stage noise --run noise_v2 checks all full raw outputs independently and hashes them.
4. diagnose_empty.py locates initial output failures in the source's empty-query rows; source dense softmax semantics require retaining these projections. AMENDMENT_1 freezes that correction without changing the numerical tolerance or attention kernels.
5. qualify_keep_empty_v1 is a guard failure before CUDA initialization because an unrelated GPU0 process appeared after the idle snapshot. qualify_keep_empty_v2 uses GPU6 and the identical corrected candidate arithmetic. Independent CPU audit uses verify_run.py --stage qualify --run qualify_keep_empty_v2.
6. timing_keep_empty_v1, GPU6, requires the corrected qualification and CPU audit to pass. Nine random paired rounds, five full local forward+backward calls per interval, all 12 inputs, with four warm calls per arm. CUDA interval and synchronized wall times are both retained. Source path counters and count-only extension wrappers remain paid; no detailed event scopes or hooks are inside timed calls.

Examples (run on remote host, pick a fresh run name; captured-input paths remain pinned to capture_v1):

```bash
python3 /home/data/wangxuran/flash_wkv_probe_20260911/src/supervise_gpu6.py --run timing_repeat --stage timing --script probe_keep_empty.py
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 /home/data/wangxuran/isaacsim6/env/bin/python /home/data/wangxuran/flash_wkv_probe_20260911/src/verify_run.py --stage qualify --run qualify_keep_empty_v2
```

Local regeneration from transferred metadata:

```bash
python src/analyze.py
python src/plot.py
python src/report.py
```

Source-dense and source-packed cohorts are defined by the frozen SOURCE policy before timing, not selected by a favorable speed outcome. Ratios are medians of paired ratios, so dividing independent displayed arm medians need not produce the same value. Bootstrap intervals are descriptive for these fixed input sets in this session, not estimates of full-epoch or general training speed. No local ratio is multiplied by previous profile fractions.

Large .pt input and full-result archives remain remotely available. capture.json records every captured input hash; each independent CPU audit records all compared raw result hashes. Final remote_manifest.json verifies all files, and transfer_verification.json defines which non-tensor files are local. MANIFEST.sha256 seals local artifacts. ATTEMPTS.md and amendments retain every failure rather than replacing it with a later passing result.
