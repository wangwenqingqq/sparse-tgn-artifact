# Timing replication, after full_v5

Full_v5 completed all 12 windows. Its 96 keeper self-replay checks and each
96 free_graph/free_relations checks passed the frozen tolerance. Source-vs-keeper
failed on 10 of 96 steps, across Wikipedia/CollegeMsg B200 window s20; these
failures are retained and must not be replaced by the replication.

All free-relations median ratios were 1.029–1.070, but individual ratios reached
1.551, and one case ranged 0.975–1.126. This warrants a bounded timing repeat.
Reload exactly the full_v5 checkpoints (no new history, seeds, data, or models).
First verify the reproduced keeper outputs/state against archived keeper raw
arrays. Then collect nine randomly ordered paired rounds per window and variant.
Collect Python garbage after each checkpoint restore and before timing to remove
checkpoint-construction debt; leave GC enabled during normal training. This is
a measurement-protocol change, reported separately from the original timing.
Keep both timing sets, original correctness flags, and all raw observations.
