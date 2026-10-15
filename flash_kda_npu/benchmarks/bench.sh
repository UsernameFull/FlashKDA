set -e
cd "$(dirname "$0")/../.."

# Ascend hosts need the CANN runtime on LD_LIBRARY_PATH before torch can
# even import: torch_npu auto-loads as a device backend and its _C ext fails
# to resolve the runtime symbols otherwise.
if [ -f /usr/local/Ascend/ascend-toolkit/latest/set_env.sh ]; then
    source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
fi

pip install -e . --no-build-isolation -q
python flash_kda_npu/benchmarks/bench_fwd.py "$@"
