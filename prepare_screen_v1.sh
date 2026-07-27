#!/usr/bin/env bash
set -eu

root="$(cd "$(dirname "$0")" && pwd)"
python="$root/.venv/bin/python"
seqbench="$root/.venv/bin/seqbench"
source="${1:-$root/data/causal-v2}"
output="${2:-$root/data/screen-v1}"

mkdir -p "$output"

if [ ! -f "$source/proofwriter-credit.csv" ]; then
  "$python" "$root/tools/materialize_credit.py" \
    "$source/proofwriter-full.csv" "$source/proofwriter-credit.csv"
fi
if [ ! -f "$source/mrcr-full.csv" ]; then
  "$python" "$root/tools/build_csv.py" "$source/mrcr-full.csv" \
    --datasets mrcr --configs 2needle --splits test --mrcr-needles 2needle
fi
if [ ! -f "$source/clutrr-rare-exceptions.csv" ]; then
  "$python" "$root/tools/materialize_rare_exceptions.py" \
    "$source/clutrr-full.csv" "$source/clutrr-rare-exceptions.csv" --seed 42
fi
if [ ! -f "$source/clutrr-correction.csv" ]; then
  "$python" "$root/tools/materialize_correction.py" \
    "$source/clutrr-full.csv" "$source/clutrr-correction.csv"
fi

"$python" "$root/tools/materialize_screen.py" \
  "$root/specs/screens/screen_v1.yaml" "$output/tasks.csv" \
  "$source/babi-full.csv" \
  "$source/babi-oracle.csv" \
  "$source/babilong-full.csv" \
  "$source/mrcr-full.csv" \
  "$source/clutrr-full.csv" \
  "$source/clutrr-nonce.csv" \
  "$source/clutrr-nonce-counterfactual.csv" \
  "$source/clutrr-noise.csv" \
  "$source/clutrr-rare-exceptions.csv" \
  "$source/clutrr-correction.csv" \
  "$source/proofwriter-full.csv" \
  "$source/proofwriter-credit.csv" \
  "$source/recogs-full.csv" \
  "$source/slog-full.csv"

"$seqbench" validate-csv "$output/tasks.csv"
