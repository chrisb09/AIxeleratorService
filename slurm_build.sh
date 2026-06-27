#!/usr/bin/env bash
#SBATCH --partition=devel
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --time=01:00:00
#SBATCH --job-name=build-aix
#SBATCH --output=build-aix.%j.out
#SBATCH --error=build-aix.%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ABS_SCRIPT="${SCRIPT_DIR}/$(basename "${BASH_SOURCE[0]}")"

if [ -z "${SLURM_JOB_ID:-}" ]; then
    echo "Not inside a Slurm job. Re-executing via srun on devel partition with 96 cores..."
    exec srun --partition=devel --cpus-per-task=96 --time=01:00:00 "${ABS_SCRIPT}" "$@"
fi

echo "=== Slurm Build Job Started ==="
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "CPUs allocated: ${SLURM_CPUS_ON_NODE:-96}"

cd "${SCRIPT_DIR}"
echo "Running build.sh in ${SCRIPT_DIR}..."
./build.sh "$@"

echo "=== Slurm Build Job Completed Successfully ==="