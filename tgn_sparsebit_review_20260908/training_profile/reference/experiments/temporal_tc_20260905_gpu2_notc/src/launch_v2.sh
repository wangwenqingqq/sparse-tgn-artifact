#!/usr/bin/env bash
# User-authorized GPU2 only. No packages, resets, clocks, or other workloads touched.
set -euo pipefail
cd /home/data/wangxuran/factor_tgn_sptc_20260902/project/experiments/temporal_tc_20260905_gpu2_notc
RUN_ID="${1:?fresh run ID required}";mkdir -p output;mkdir "output/$RUN_ID";RUN="$PWD/output/$RUN_ID"
exec 9>/home/data/wangxuran/.locks/tgn_sptc_gpu2.lock
flock -n 9 || { echo 'project GPU2 lock busy'; exit 73; }
export CUDA_VISIBLE_DEVICES=GPU-16f27f5a-dfcd-48e0-bb39-bebbe4009245
export PATH=/usr/local/cuda/bin:$PATH
export CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export GPU_NT_LIB="$RUN/native_gpu.so" NT_LIB="$RUN/reference_cpu.so"
PY=/home/data/wangxuran/isaacsim6/env/bin/python
echo "PID=$$ $(date -u +%FT%TZ)" > "$RUN/launcher.pid"
snapshot() {
  local tag="$1"
  { date -u; hostname; id; who; df -h "$PWD"; nvidia-smi --query-gpu=index,uuid,clocks.current.sm,clocks.current.memory,power.limit,pstate,temperature.gpu,memory.used,utilization.gpu --format=csv; } > "$RUN/$tag.host.txt"
  nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory --format=csv > "$RUN/$tag.apps.csv"
  if grep -q "$CUDA_VISIBLE_DEVICES" "$RUN/$tag.apps.csv"; then echo 'Selected GPU occupied, abort without touching it';exit 74;fi
}
phase() {
  local tag="$1";shift;snapshot "$tag.before"
  echo "START $tag $(date -u +%FT%TZ)" | tee -a "$RUN/status.txt"
  set +e
  "$@" > "$RUN/$tag.stdout" 2> "$RUN/$tag.stderr"
  local rc=$?;set -e;echo "$rc" > "$RUN/$tag.exit"
  echo "END $tag rc=$rc $(date -u +%FT%TZ)" | tee -a "$RUN/status.txt"
  snapshot "$tag.after";test "$rc" -eq 0
}
snapshot opening
test "$(df -Pk "$PWD" | tail -1 | awk '{print $4}')" -ge 5242880
{ nvcc --version; compute-sanitizer --version; g++ --version; "$PY" -V; } > "$RUN/toolchain.txt"
phase source_verify "$PY" ../temporal_tc_20260905_native_notc_control/src/verify_source.py
phase build nvcc -std=c++17 -O3 --fmad=false -lineinfo -arch=sm_120 -Xcompiler=-fPIC -shared -Xptxas=-v src/kernels.cu -o "$GPU_NT_LIB"
phase cpu_build g++ -std=c++17 -O3 -ffp-contract=off -fPIC -shared ../temporal_tc_20260905_native_notc_control/src/native.cpp -o "$NT_LIB"
sha256sum "$GPU_NT_LIB" "$NT_LIB" > "$RUN/binary.sha256"
phase correct "$PY" src/run_v2.py --mode correct --output "output/$RUN_ID/correct"
cuobjdump --dump-resource-usage "$GPU_NT_LIB" > "$RUN/resources.txt"
cuobjdump --dump-sass "$GPU_NT_LIB" > "$RUN/sass.txt"
phase static "$PY" src/static_audit.py "$RUN"
for tool in memcheck racecheck synccheck initcheck; do
  phase "$tool" compute-sanitizer --tool "$tool" --error-exitcode 99 "$PY" src/run_v2.py --mode sanitizer --output "output/$RUN_ID/$tool"
done
phase stress "$PY" src/run_v2.py --mode stress --output "output/$RUN_ID/stress"
phase pair_check "$PY" src/run_v2.py --mode pair-check --output "output/$RUN_ID/pair_check"
for i in 0 1 2 3 4; do
  phase "bench$i" "$PY" src/run_v2.py --mode bench --process "$i" --output "output/$RUN_ID/bench$i"
done
snapshot closing;echo "COMPLETE $(date -u +%FT%TZ)" | tee -a "$RUN/status.txt"
