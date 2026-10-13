#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON_BIN:-python3}"
PYTHON_PREFIX="${PYTHON_PREFIX:-$(dirname "$(dirname "$(command -v "$PYTHON_BIN")")")}"
TORCH_PREFIX="${TORCH_PREFIX:-$(find "$PYTHON_PREFIX/lib" -maxdepth 5 -type d -path "*/site-packages/torch" | head -n 1)}"
ASCEND_HOME="${ASCEND_HOME_PATH:-/usr/local/Ascend/cann-9.1.0}"
ASCEND_ARCH="$(uname -m)-linux"
CXX="${CXX:-g++}"

PYTHON_INCLUDE="$(find "$PYTHON_PREFIX/include" -maxdepth 1 -type d -name "python3.*" | sort | tail -n 1)"
if [[ -z "$TORCH_PREFIX" || -z "$PYTHON_INCLUDE" ]]; then
    echo "cannot locate Python or Torch headers (PYTHON_PREFIX=$PYTHON_PREFIX)" >&2
    exit 1
fi

ROOT="$(cd ../.. && pwd)"
BUILD_DIR="$ROOT/flash_kda_npu/lib"
OUTPUT="$BUILD_DIR/flash_kda_npu_launcher.so"
SOURCE="$PWD/flash_kda.cpp"
HEADER="$PWD/fwd.h"
mkdir -p "$BUILD_DIR"

if [[ -f "$OUTPUT" && "$OUTPUT" -nt "$SOURCE" && "$OUTPUT" -nt "$HEADER" ]]; then
    echo "flash_kda_npu_launcher.so is up to date: $OUTPUT"
    exit 0
fi

echo "Building extension module flash_kda_npu_launcher..."
"$CXX" \
    -MMD -MF "$BUILD_DIR/flash_kda.o.d" \
    -DTORCH_EXTENSION_NAME=flash_kda_npu_launcher \
    -DTORCH_API_INCLUDE_EXTENSION_H \
    -DPYBIND11_COMPILER_TYPE='"_gcc"' \
    -DPYBIND11_STDLIB='"_libstdcpp"' \
    -DPYBIND11_BUILD_ABI='"_cxxabi1016"' \
    -D_GLIBCXX_USE_CXX11_ABI=1 \
    -isystem "$TORCH_PREFIX/include" \
    -isystem "$TORCH_PREFIX/include/torch/csrc/api/include" \
    -isystem "$PYTHON_INCLUDE" \
    -I "$ASCEND_HOME/$ASCEND_ARCH/include" \
    -fPIC -std=c++17 -O3 \
    -c "$SOURCE" -o "$BUILD_DIR/flash_kda.o"

echo "Linking $OUTPUT"
"$CXX" "$BUILD_DIR/flash_kda.o" \
    -shared \
    -L "$ASCEND_HOME/$ASCEND_ARCH/lib64" -lascendcl -lacl_rtc \
    -L "$TORCH_PREFIX/lib" -lc10 -ltorch_cpu -ltorch -ltorch_python \
    -o "$OUTPUT"

echo "Build successful: $OUTPUT"
