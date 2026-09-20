# Deterministic trajectory amendment, before full timing

GPU smoke v4 is retained as failed. Repeating keeper from the same checkpoint
at Wikipedia B200 changed neighbor_eid at step 4; step 5 then changed embeddings,
logits, and gradients. CollegeMsg B200 eventually changed query/edge structure.
CollegeMsg B32 also had a gradient discrepancy. No tolerance is widened.

The source LastNeighborLoader.insert performs indexed assignments to dense_id;
when a node appears more than size=10 times, these indices repeat. This is a
source-level explanation for the observed replay instability, not a diagnosis
of the relation producer. PyTorch documents deterministic index_put when
torch.use_deterministic_algorithms(True) is enabled:
https://docs.pytorch.org/docs/2.11/generated/torch.use_deterministic_algorithms.html

Full v5 profiling will enable deterministic algorithms for every comparison arm
and all training history; set CUBLAS_WORKSPACE_CONFIG=:4096:8. Keep ordinary
uninitialized allocation behavior (fill_uninitialized_memory=False); unused
capacity is not logical input. Record flags with output and rerun all checks.
Third-party operations may still have nondeterminism; the checks remain the
acceptance criterion. Full default-nondeterministic throughput is a different
denominator and is not claimed by this amended experiment. All prior failures
remain visible; this setting does not retroactively qualify them.
