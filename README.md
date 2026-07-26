# seqbench

`seqbench` is an executable, model-independent behavioral benchmark. It sees
an opaque checkpoint and calls only two process operations:

```text
learn(model, examples) -> new model
infer(model, requests) -> predictions
```

It does not import the tested algorithm and does not require retrieval traces,
attention maps, class IDs, caches, or any other model internals.

## Install

```bash
python -m venv .venv
.venv/bin/pip install -e '.[dev,data]'
```

## Prepare CSV files

No dataset is stored in this repository. The base builder is deliberately a
plain streaming script:

```bash
.venv/bin/python tools/build_csv.py data/full.csv
```

It downloads missing pinned upstream sources and writes only the original
`full` rows. There is no preflight, output checksum, SQLite index, temporary
dataset, or row-count manifest.

Materialize fixed probes explicitly:

```bash
.venv/bin/python tools/build_csv.py data/oracle.csv --variant oracle
.venv/bin/python tools/materialize_nonce.py \
  data/full.csv data/nonce.csv --seed 42
.venv/bin/python tools/materialize_counterfactual.py \
  data/full.csv data/counterfactual.csv
.venv/bin/python tools/materialize_noise.py \
  data/full.csv data/noise.csv --seed 42
.venv/bin/python tools/materialize_supervision.py \
  data/full.csv data/supervision.csv
```

An interrupted direct build leaves an incomplete output file. Delete it and
run the command again.

Validate any collection of task files:

```bash
.venv/bin/seqbench validate-csv data/*.csv
```

## Algorithm contract

An algorithm YAML supplies an argv command, manifest, and initial checkpoint:

```yaml
command: ["./algorithm"]
manifest: manifest.json
initial_model: model
```

The runner invokes:

```bash
algorithm learn \
  --model-in MODEL_IN \
  --examples EXAMPLES.jsonl \
  --model-out MODEL_OUT \
  --budget BUDGET.json \
  --seed 42

algorithm infer \
  --model MODEL \
  --requests REQUESTS.jsonl \
  --output PREDICTIONS.jsonl \
  --budget BUDGET.json
```

`score` responses contain the natural logarithm of normalized conditional
probability. `log_probability: null` means exact probability zero.

The static manifest is:

```json
{
  "name": "algorithm-name",
  "version": "1",
  "capabilities": {
    "learn": true,
    "generate": true,
    "score": true
  },
  "external_pretraining": false
}
```

## Run

```bash
.venv/bin/seqbench run specs/runs/full_v1.yaml \
  --algorithm /path/to/algorithm.yaml \
  --tasks data/full.csv \
  --tasks data/oracle.csv \
  --tasks data/nonce.csv \
  --tasks data/counterfactual.csv \
  --tasks data/noise.csv \
  --tasks data/supervision.csv \
  --output runs/my-algorithm
```

The runner reads the CSV files once, routes rows through declarative YAML
selectors, enforces budgets, checks that inference did not mutate checkpoints,
and writes:

```text
runs/my-algorithm/
    manifest.json
    predictions.parquet
    metrics.json
    properties.json
    diagnostics.json
    resources.json
    report.md
    report.html
    curves/
```

The first release intentionally reports `UNCALIBRATED` instead of inventing
pass/fail thresholds. Once weak and strong reference profiles exist:

```bash
.venv/bin/seqbench calibrate \
  --weak runs/random \
  --weak runs/memorizer \
  --strong runs/reference \
  --output specs/calibration/v1.json
```

## Tests

```bash
.venv/bin/pytest
.venv/bin/ruff check .
```

The golden test executes CSV → learn → infer → metrics → properties →
Parquet/Markdown/HTML with a black-box fixture adapter.

