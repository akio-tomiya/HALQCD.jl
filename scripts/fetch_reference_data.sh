#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
destination="$repo_root/reference"
mkdir -p "$destination"

clone_or_update() {
    local url="$1"
    local name="$2"
    if [[ -d "$destination/$name/.git" ]]; then
        git -C "$destination/$name" pull --ff-only
    else
        git clone --depth 1 "$url" "$destination/$name"
    fi
}

clone_or_update https://github.com/tkmiriya/hal_pot_code.git hal_pot_code
clone_or_update https://github.com/tkmiriya/systematics_of_hal_qcd_pot.git systematics_of_hal_qcd_pot
clone_or_update https://github.com/tkmiriya/nn_int_at_mpi_510MeV.git nn_int_at_mpi_510MeV

git -C "$destination/hal_pot_code" rev-parse HEAD
git -C "$destination/systematics_of_hal_qcd_pot" rev-parse HEAD
git -C "$destination/nn_int_at_mpi_510MeV" rev-parse HEAD
