#!/usr/bin/env bash
#SBATCH --partition=devel
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --job-name=dl-libtorch
#SBATCH --output=dl-libtorch.%j.out
#SBATCH --error=dl-libtorch.%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ABS_SCRIPT="${SCRIPT_DIR}/$(basename "${BASH_SOURCE[0]}")"

if [ -z "${SLURM_JOB_ID:-}" ]; then
    echo "Submitting to Slurm..."
    exec sbatch "${ABS_SCRIPT}"
fi

LIBTORCH_DIR="$(realpath "${SCRIPT_DIR}/../../libtorch")"
LIBTORCH_VERSION="2.4.0"
LIBTORCH_ARCH="cu124"
TMPDIR="${TMPDIR:-/tmp}"

echo "Downloading libtorch ${LIBTORCH_VERSION}+${LIBTORCH_ARCH}..."

# Remove old libtorch
rm -rf "${LIBTORCH_DIR}"

tmp_dir="$(mktemp -d "${TMPDIR}/libtorch.XXXXXX")"
libtorch_zip="libtorch-cxx11-abi-shared-with-deps-${LIBTORCH_VERSION}%2B${LIBTORCH_ARCH}.zip"
libtorch_url="https://download.pytorch.org/libtorch/${LIBTORCH_ARCH}/${libtorch_zip}"

echo "URL: ${libtorch_url}"
cd "${tmp_dir}"
wget -O libtorch.zip "${libtorch_url}" || curl -L -o libtorch.zip "${libtorch_url}"
unzip -q libtorch.zip

if [ -d "${tmp_dir}/libtorch" ]; then
    mv "${tmp_dir}/libtorch" "${LIBTORCH_DIR}"
    echo "libtorch ${LIBTORCH_VERSION} installed to ${LIBTORCH_DIR}"
    cat "${LIBTORCH_DIR}/build-version" 2>/dev/null || true
    echo "Done."
else
    echo "ERROR: libtorch directory not found after extraction"
    exit 1
fi

rm -rf "${tmp_dir}"
