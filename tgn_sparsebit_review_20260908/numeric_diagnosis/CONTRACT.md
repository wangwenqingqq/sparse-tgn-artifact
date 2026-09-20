# Diagnosis of real-training numerical divergence

Prior evidence is immutable: training_profile/output/full_v5 on the remote host.
Primary witnesses: Wikipedia and CollegeMsg, B200, steps 20–27. No tolerance
change; existing atol=rtol=2e-4 applies to all full-state comparisons.

Questions, in order:

1. At identical actual embedding, query, integer C and decoder weights, where
   do the source and keeper first differ? Compare aggregation, logits, BCE,
   dX and decoder parameter gradients against an FP64 dense mathematical
   reference lifted from the same FP32 inputs. Save intermediate raw tensors.
   Check common-upstream VJPs as well as each path's own BCE backward, so
   forward rounding and backward reduction can be distinguished.
2. From the first differing time-encoding weight update, is that parameter
   difference alone sufficient to change the next embedding? Use artificial
   parameter transplantation and time-weight anchoring to the keeper trajectory
   as causal controls. These are diagnostics, not proposed deployable methods.
3. Before changing model precision, test an arithmetic-compatibility candidate:
   retain the keeper's paid integer coefficient producer but reconstruct the
   seven torch_sparse consumers and original positive/negative endpoint-product
   autograd graph. A dense coefficient intermediate is permitted for diagnosis;
   speed is not assumed. If it passes, use it as a source-compatible control.
4. Test a bounded numerical-stability candidate: time-encoding affine/cos and
   its learnable parameters/Adam moments in FP64, casting encoding outputs back
   to FP32. All other training stays FP32, with actual memory and embeddings.
   This changes numerical precision and must be compared as a separate mode;
   it is not a claim of bitwise equivalence to the original FP32 trajectory.
   If promising, test all twelve already-frozen windows, not only the failures.

Use the prior checkpoint, actual chronological data, source modules, GPU
libraries and dependency environment. Use deterministic algorithms and prior
CUBLAS workspace setting. All repeated runs restore the same checkpoint/RNG.
Record observed differences, including failures. Do not infer final task quality
from eight-step windows. No new data split, loss, sampling rule, fixed learned
frequency, or time rescaling is introduced as a repair.

Use a new experiment directory, existing GPU2 lock, live idle/process checks.
No native kernel changes. The source snapshot and previous profiling archives
are read only. FP64 is a reference of higher precision, not exact real arithmetic.
