# Measured cost of the qualified compatibility path

The compatibility path reproduced every checked field exactly in all 96 frozen
steps. The time64 candidate failed 9 of 16 witness steps, so it is rejected at
the witness gate and will not be promoted by changing tolerance or selecting
other windows.

Measure the actual compatibility implementation, including dense coefficient
conversion, against the old source and keeper. Add an oracle which caches only
the compatible sparse coefficients, still recomputing all seven aggregations,
embeddings, memory, gradients and Adam updates. This oracle removes graph build,
integer relation production and conversion; it is a diagnostic upper bound,
not a proposed implementation. Verify all 96 oracle states against archived
source states before timing each window. Use nine paired, randomized rounds of
eight complete steps, restored from the frozen checkpoint each time. Collect
garbage outside each timed interval. All four arms use the same protocol.

The old keeper still fails the original source-trajectory gate in two windows;
its timing remains a diagnostic comparator. A compatibility-path slowdown is a
cost of this implementation, not evidence that exact arithmetic requires that
cost. No native kernel modification or memory optimization is part of this
diagnostic experiment.
