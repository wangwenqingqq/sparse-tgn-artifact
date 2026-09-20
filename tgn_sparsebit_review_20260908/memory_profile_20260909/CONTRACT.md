Identify the real training bottleneck on the qualified direct-CSR TNCN path.

Use the same twelve frozen checkpoints and chronological training windows,
source modules, FP32 model, original consumer/autograd order and deterministic
settings. Prior experiments and original project remain read only. Use the
shared GPU3 lock, idle checks and per-run process isolation. New timings are
not pooled with the prior GPU3 or GPU2 sessions.

Measure whole steps without detailed instrumentation, and separately measure
memory lookup/concatenation, time encoding, last-message aggregation, GRU,
state writes, and message-store updates. Collect CPU/CUDA profiler traces on
the first step of the four middle windows (B32 step256, B200 step40). Correlate
backward nodes with their forward creation scopes through sequence number and
forward thread; report coverage, ambiguities and unassigned work. Do not add
backward hooks that change the autograd graph. Do not equate traced GPU gaps
with a hardware-utilization metric or treat profiler timings as production time.

A concrete causal probe is permitted: replace per-node gathers in
_update_msg_store with one ordered gather per field followed by tensor views
and the same per-node dictionary assignment. Keep sorting, per-node message
order and values. This reduces GPU submissions while preserving the logical
cache. It can retain larger parent allocations; measure actual logical and
unique-storage bytes and disclose the bounded-window limitation.

Before treating the probe as performance-eligible, require all 96 steps to
match the direct baseline byte for byte, including the previously checked
fields, logically packed message stores and CPU/CUDA RNG states. Also check
the baseline against the archived source fields. Retain old tolerances and all
failures. Timing: nine randomized paired rounds per window, eight complete
steps per interval, restore and GC outside timing. CPU audit precedes timing.

The purpose is bottleneck attribution and a bounded causal measurement. There
is no full-epoch, downstream-quality, original-TGN or Tensor Core claim.
