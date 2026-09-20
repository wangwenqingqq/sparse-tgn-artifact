# Reproduction

Remote root: `pro6000-8:/home/data/wangxuran/tncn_flow_replay_20260909`.
Interpreter: `/home/data/wangxuran/isaacsim6/env/bin/python`.
Dependencies: the unchanged `/home/data/wangxuran/tncn_training_profile_20260908/deps`.
The existing project and earlier experiments are imported read-only. No new package installation was needed.

The protocol is fixed in [CONTRACT.md](CONTRACT.md). The primary code is [src/experiment.py](src/experiment.py). It imports the previously source-qualified direct-CSR decoder and batched message writes. All teacher and candidate historical replay executes in evaluation mode under `torch.no_grad`; dropout is off. Student fitting is ordinary autograd training of a separate network. Teacher weights remain frozen during sample generation, candidate evaluation and timing.

Use unused run and artifact-directory names:

```bash
python3 src/supervise.py --run new_teacher --stage teacher --gpu 4 --artifact-dir new_experiment
python3 src/supervise.py --run new_pipeline --stage pipeline --gpu 4 --artifact-dir new_experiment
```

`pipeline` holds the existing GPU lock across samples, fitting, evaluation, CPU audit and timing. It requires an idle GPU before launch. The child checks the target UUID for competing processes during work. The CPU audit hides CUDA. A nonzero exit stops dependent stages. Each run records command arguments, code hashes, GPU/process inventories and logs. Pipeline source snapshots and the `start` event identify the actual implementation. Timings compare all methods on the same GPU4 session; teacher preparation ran separately on GPU2 and is not pooled into replay speed ratios.

Completed/attempted run names are preserved in `evidence/runs`. `teacher_v1` generated the new five-epoch teacher checkpoints. `samples_v1` did not launch because GPU2's lock was occupied. `pipeline_v1` switched to the idle GPU4 and performed the planned candidate experiment.

The temporal data boundaries are fixed at events 12000, 16000 and 20000. Teacher training uses the first 12000 events for five epochs. Epoch selection uses validation AP on the next 4000. As in the existing training harness, validation immediately follows the training history with the standard train-to-eval memory flush. After selecting and freezing weights, all distillation targets and test trajectories rebuild memory from zero with the frozen teacher. Thus the targets do not mix teacher weights from different optimization steps.

Both sources retain the earlier dataset convention: integer timestamps, Wikipedia's real 172-dimensional messages, CollegeMsg's single zero message channel, and uniform negatives from the dataset destination-ID range. These are 20000-event archive experiments, not official full-dataset benchmark scores. One teacher seed was used; the three repeated seeds apply to students.

Teacher inference updates memory in batches of 32 events. Blocks of 128/512 events have 4/16 such updates. Every candidate retains all original per-batch cache writes and neighbor insertions in the same order. Predicted memory is only substituted at block boundaries. Timestamps are derived from observed events. `coarse` computes one actual teacher GRU update using the block's final exact message cache and the initial memory, then writes exact timestamps. `copy` retains initial memory. Neither control is bitwise equivalent to the teacher's floating-point state.

The MLP and paired conditional FM use identical parameter counts and the same sampled training rows. The FM auxiliary inputs are an interpolated state residual and interpolation time; the MLP receives zeros in these positions. Initial memory is part of conditioning. Four-bin event averaging and initial peer memory are lossy summaries: this screen evaluates that encoder, not every possible conditional flow architecture. One/two/four Euler evaluations all use the FM checkpoint selected by two-evaluation validation loss. There is no reflow, extra distillation, endpoint clipping or test-based model selection.

The exact test-start checkpoint at event 16000 is shared by candidates. Their subsequent blocks use their own predicted memory. Each block is followed by a read-only query on the next 32 events. Query outputs do not change memory or the neighbor cache. The next block includes those events as observed history; no query uses its own event or later events. All 31/7 full blocks are retained, respectively. This leaves 32/416 events outside the replay interval. Per-case query sets differ between block sizes and must not be pooled as an accuracy comparison.

The main timing is **historical replay only**. It includes slicing, message-cache operations, neighbor insertion, student feature encoding and normalization, every model evaluation and solver update, memory/timestamp writes, allocations and synchronization. It excludes query GNN/decoder calls, backward, Adam, checkpoint restoration, loading weights, pretraining and CPU verification. Seven randomized paired rounds execute 420 complete suffix replays. Each timed final memory is checked against that method's archived closed-loop output, outside the timer. There is no teacher-target lookup inside candidate timing.

Raw `.pt` artifacts (~1.2 GB for this experiment) remain under the remote `artifacts/main` directory. Small JSON/NPZ artifacts are copied to `evidence/artifacts`. Their remote hashes and sizes are listed in `remote_manifest.json`. Recompute the CPU audit using a new output filename:

```bash
env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /home/data/wangxuran/isaacsim6/env/bin/python src/audit.py \
  --input artifacts/main --output new_cpu_audit.json
env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /home/data/wangxuran/isaacsim6/env/bin/python src/export_queries.py --input artifacts/main
```

The CPU audit independently computes AP using score thresholds, AUC using average ranks, BCE using a stable scalar expression, selected memory errors and raw discrete-state bytes. It also checks split boundaries, the initial-memory feature and normalization statistics against training rows only. All query predictions and 210 selected raw state snapshots are audited; intermediate block metadata has recorded GPU byte comparisons. This is numerical evidence, not a formal proof of the source or causality.

Local analysis and standalone plots:

```bash
python3 src/analyze.py --input evidence/artifacts --output analysis
python3 src/plot.py --input evidence/artifacts --analysis analysis
```

The nominal practical screen is >=1.10x replay speed, with AP and AUC each losing at most one percentage point on boundary queries. `analysis/summary.json` reports every seed, seven-round paired bootstrap timing intervals and paired block-bootstrap quality intervals. Temporal blocks are correlated and only one test suffix and GPU timing session are present. The intervals are descriptive, and a point-estimate pass is not a deployment-quality guarantee.
