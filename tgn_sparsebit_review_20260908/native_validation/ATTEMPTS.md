V1 compiled for sm80 and sm120a, and its 30-case native smoke passed on GPU2.
The compiler reported a 64-byte local stack for all sparse consumers. Inspection
identified runtime indexing of four candidate values in encode4. Before timing,
V2 replaced this with unrolled register selections for every sparse arm. V2's
sm120a compilation reports zero stack and zero spill loads/stores. V1 source,
binary, compiler output and smoke results remain available; no V1 timing is used.

The first local rsync lacked its parent output directory and failed; a following
verification command also failed on that absent directory. The directory was
created and the unchanged remote raw files were transferred again. These were
local transfer/setup errors, with no modification to the GPU evidence.

Configured a800-server1 and a8003 SSH connections timed out. RTX PRO 6000 is the
runtime target; sm80 and sm89 compilation is static evidence only. No process on
other GPUs is stopped or changed by this experiment.
