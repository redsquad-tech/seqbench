#!/usr/bin/env bash
set -eu

root="$(cd "$(dirname "$0")" && pwd)"
python="$root/.venv/bin/python"
seqbench="$root/.venv/bin/seqbench"
output="${1:-$root/data/causal-v1}"

mkdir -p "$output"

"$python" "$root/tools/build_csv.py" "$output/babi-full.csv" \
  --datasets babi --configs en-qa2 --splits train,test
"$python" "$root/tools/build_csv.py" "$output/babi-oracle.csv" \
  --datasets babi --configs en-qa2 --splits train,test --variant oracle
"$python" "$root/tools/build_csv.py" "$output/babilong-full.csv" \
  --datasets babilong --configs 0k,1k,16k

"$python" "$root/tools/build_csv.py" "$output/clutrr-full.csv" \
  --datasets clutrr --configs gen_train23_test2to10 --splits train,test
"$python" "$root/tools/materialize_nonce.py" \
  "$output/clutrr-full.csv" "$output/clutrr-nonce.csv" --seed 42
"$python" "$root/tools/materialize_counterfactual.py" \
  "$output/clutrr-full.csv" "$output/clutrr-counterfactual.csv"
"$python" "$root/tools/materialize_noise.py" \
  "$output/clutrr-full.csv" "$output/clutrr-noise.csv" --seed 42

"$python" "$root/tools/build_csv.py" "$output/proofwriter-full.csv" \
  --datasets proofwriter --configs depth-5 --splits train,test
"$python" "$root/tools/materialize_supervision.py" \
  "$output/proofwriter-full.csv" "$output/proofwriter-supervision.csv"

"$python" "$root/tools/build_csv.py" "$output/recogs-full.csv" \
  --datasets recogs --configs recogs_v2 --splits train,test,generalization
"$python" "$root/tools/build_csv.py" "$output/slog-full.csv" \
  --datasets slog --configs cogs_LF --splits train,test,generalization

"$seqbench" validate-csv "$output"/*.csv
