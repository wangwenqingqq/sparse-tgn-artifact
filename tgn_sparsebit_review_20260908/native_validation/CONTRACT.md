Native consumer-layout validation, frozen before compilation and execution.

Objective: verify the proposed B1 output column schedule, actual sparse operand
and metadata ABI, and its benefit against equally optimized controls. This is
a bounded GPU-resident operator experiment, not a TNCN training experiment.

Operation: Q[r,z] = any_k(Pleft[r,k] AND Pright[z,k]); C = Q * W; Y = C X.
Pleft has 16 rows, Pright 32 rows, and both are ordinary row-major bitmaps.
W contains signed, exactly representable small weights; X is FP16. CPU reference
uses the literal bit intersections and double accumulation of actual input
values. C packing is lossless, with at most two 2:4 planes. Accumulation is FP32.
This weighted predicate consumer models a handoff; W is an explicit input,
not a claimed free TNCN S computation. No neural-feature quantization claim.

Controls:
- natural_sparse: contiguous B1 panels, Boolean compression before shuffle,
  dynamic sparse encoding and metadata, sparse MMA.
- aligned_sparse: consumer-directed B1 panels, local Boolean result ownership,
  otherwise identical coefficient, metadata, and sparse consumption.
- natural_dense: B1 with naturally compatible ordinary dense MMA consumption.
- simt_sparse: direct word intersection with legal first-positive short-circuit,
  same dynamic sparse consumer.
- aligned_split: same aligned producer but writes C, then a second kernel reads
  and encodes it. It is an explicit materialization attribution control.

Measured resident scope includes canonical bitmap loads/column selection, Q,
W loads, coefficient construction, sparse selection, metadata gather, X loads,
MMA, output stores and launches. A second timing scope additionally re-packs
byte-valued bitmap inputs on GPU on every invocation. Allocation and H2D are
outside both scopes and cannot be inferred to be free in a training system.

Correctness: exact Q; finite Y within abs error 1e-5 (dyadic small input tests
are expected exact). Include all 16 quartet masks, dense/empty/single witness,
K boundaries/tails, signed/zero weights, X identity and feature tails.
All variants must pass before timing. Keep all failed attempts.
Run Compute Sanitizer memcheck/racecheck/initcheck/synccheck on bounded tests.

Benchmark: predeclared K={128,512}, D={8,32,100}, tile count={32,256,2048},
two topology classes (exactly two support positions per quartet; dense support).
Report every cell and paired randomized-order rounds, not a best-case subset.
No specific speedup is required to accept correctness; no performance benefit
may be claimed solely from instruction or byte counts.

Target selection: use a currently idle, lockable GPU; recheck UUID/processes
before launch and preserve other workloads. Prefer native BMMA when reachable.
On Blackwell, qualify actual lowering and label B1-to-IMMA execution separately.
Compile sm80/sm89/sm120a for static ISA evidence; only run on the selected device.

Native backward and real sampled TNCN data are outside this small experiment.
The experiment does not establish an end-to-end joint mechanism or novelty.
