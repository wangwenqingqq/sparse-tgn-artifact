Build with CUDA 13.1, C++17:

```bash
nvcc -O3 -std=c++17 -lineinfo -Xptxas=-v -arch=sm_120a layout_probe.cu -o probe
```

Choose an available GPU and retain the same external lock/idle checks as
run_guarded.py. The archived experiment used GPU2 UUID recorded in run.json.
Do not reuse an existing raw output directory.

```bash
mkdir NEW_RAW_DIRECTORY
CUDA_VISIBLE_DEVICES=YOUR_GPU_UUID LAYOUT_ARCHIVE_DIR=NEW_RAW_DIRECTORY ./probe check
CUDA_VISIBLE_DEVICES=YOUR_GPU_UUID compute-sanitizer --tool memcheck --error-exitcode 91 ./probe small
CUDA_VISIBLE_DEVICES=YOUR_GPU_UUID compute-sanitizer --tool racecheck --error-exitcode 91 ./probe small
CUDA_VISIBLE_DEVICES=YOUR_GPU_UUID compute-sanitizer --tool initcheck --error-exitcode 91 ./probe small
CUDA_VISIBLE_DEVICES=YOUR_GPU_UUID compute-sanitizer --tool synccheck --error-exitcode 91 ./probe small
CUDA_VISIBLE_DEVICES=YOUR_GPU_UUID ./probe bench > NEW_BENCHMARK.csv
```

The host checker reconstructs all literal intersections and weighted outputs.
Only run timing after all correctness and sanitizer commands pass.

Independent CPU replay (NumPy; no CUDA import):

```bash
OPENBLAS_NUM_THREADS=1 python3 verify_raw.py output/full_v2/correctness_raw --output NEW_VERIFY.json
python3 analyze_benchmark.py output/full_v2/benchmark.stdout --output NEW_SUMMARY.json
```

The binary raw format is little endian: six uint32 fields (magic, tile count,
logical K, padded word count, feature dimension, variant), followed by packed
uint32 P[tiles,48,words], float32 W[tiles,16,32], float16 X[tiles,32,D], GPU
uint8 Q[tiles,16,32], GPU float32 Y[tiles,16,D]. No reference Q/Y is embedded;
verify_raw.py computes those independently.

The archived main executable, source and contract hashes are recorded in
output/full_v2/run.json. V1 compiler artifacts and smoke are historical; primary
timings use compile/probe_v2_sm120a. Static sm80/sm89 compilation is not runtime
evidence on those architectures. Input generation/H2D/allocation, TNCN weighted
S construction, native backward and training quality are outside these timings.
