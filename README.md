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
  data/nonce.csv data/nonce-counterfactual.csv
.venv/bin/python tools/materialize_noise.py \
  data/full.csv data/noise.csv --seed 42
.venv/bin/python tools/materialize_supervision.py \
  data/full.csv data/supervision.csv
```

For the causal v2 suite the same steps are collected in one direct command:

```bash
./prepare_causal_v2.sh
```

The v2 proof variant appends a training-only proof to the input while retaining
the same final target and one row per source example. It is therefore reported
as privileged supervision, not as evidence about final-only credit assignment.

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

Deterministic algorithms may additionally declare `"seed_invariant": true`.
Comparison then evaluates them once and broadcasts that fixed prediction vector
across the learned models' seed axis without pretending that repeated runs are
independent observations.

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

The corrected causal protocol is immutable and versioned separately:

```bash
.venv/bin/seqbench run specs/runs/causal_v2.yaml ...
```

It performs stable hash sampling after control/stress pairing, records the
selected IDs, and uses an independent `sampling_seed`.

## Fast hypothesis screen

Prepare the fixed compact cohort once:

```bash
./prepare_screen_v1.sh
```

Run an algorithm normally, then compare one or more completed seed runs with a
saved exact-kNN run:

```bash
.venv/bin/seqbench screen specs/screens/screen_v1.yaml \
  --candidate runs/candidate/seed-0 \
  --baseline runs/exact-knn \
  --output runs/candidate/screen
```

The screen covers prediction, retrieval, position/length transfer, binding,
multi-hop reasoning, composition, auxiliary proof supervision, noise,
one-example correction, and checkpoint efficiency. A single stochastic seed
can return `PROMOTE` or `INCONCLUSIVE`; `DROP` requires three candidate seeds.
`PROMOTE` means that the paired 95% interval still permits an improvement over
exact kNN on at least one metric; it is deliberately a permissive discovery
gate, not evidence that the improvement is already established.

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
    scaling.json
    selection.json
    resolved_specs.json
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
