#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_DIR="${SCRIPT_DIR}/INSTALL-SCOREP"
LIBTORCH_DIR="$(realpath "${SCRIPT_DIR}/../libtorch")"

echo "Building AIX against libtorch at: ${LIBTORCH_DIR}"
echo "Libtorch version: $(cat "${LIBTORCH_DIR}/build-version" 2>/dev/null || echo 'unknown')"

rm -f "${INSTALL_DIR}/lib/libAIxeleratorService.so"

cmake -S "${SCRIPT_DIR}" -B "${SCRIPT_DIR}/build" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="${INSTALL_DIR}" \
    -DWITH_TORCH=ON \
    -DLIBTORCH_DIR="${LIBTORCH_DIR}" \
    -DUSE_PYTHON_TORCH_CMAKE_PREFIX=OFF \
    -DAIX_SKIP_VENV_CREATION=ON \
    -DBUILD_TESTS=OFF

cmake --build "${SCRIPT_DIR}/build" --target AIxeleratorServiceLib -j "$(nproc)"

# Copy the AIX .so to INSTALL-SCOREP
cp "${SCRIPT_DIR}/build/src/libAIxeleratorService.so" "${INSTALL_DIR}/lib/"

# Also copy the matching torch libraries (no NCCL in standalone libtorch)
cp "${LIBTORCH_DIR}"/lib/libtorch.so \
   "${LIBTORCH_DIR}"/lib/libtorch_cpu.so \
   "${LIBTORCH_DIR}"/lib/libtorch_cuda.so \
   "${LIBTORCH_DIR}"/lib/libc10.so \
   "${LIBTORCH_DIR}"/lib/libc10_cuda.so \
   "${INSTALL_DIR}/lib/" 2>/dev/null || true

echo "Done. Library at ${INSTALL_DIR}/lib/libAIxeleratorService.so"