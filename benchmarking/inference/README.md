# Inference Benchmarking

This directory contains the OpenAI-compatible serving benchmark runner used for
AgentPerfBench trace-replay and synthetic-replay experiments.

## Layout

- `src/benchmark/` contains the runner, client, and metric aggregation.
- `src/workloads/` contains profile definitions, real-trace loaders, and
  distributional synthetic replay.
- `configs/sweep.yaml` is the hardware/model/profile sweep manifest.
- `scripts/compile_sweep.py` compiles `configs/sweep.yaml` into
  `scripts/bench_jobs.txt`.
- `scripts/run_mse_validation.sh` runs paired real-trace and distributional MSE
  validation on one vLLM instance.
- `data/distributions/` contains compact distributional profile artifacts.

Raw agent trajectory JSONL files and generated benchmark outputs are not
committed. They are ignored by `.gitignore` and are published through the
Hugging Face dataset artifact described in
`../../docs/artifact-submission/scope.md`.

## Local Smoke Test

From the repository root:

```bash
PYTHONPATH=benchmarking/inference python -m pytest -q benchmarking/tests
```

## Single Benchmark

Start a vLLM or other OpenAI-compatible server, then run:

```bash
cd benchmarking/inference
PYTHONPATH=. python -m src.benchmark.runner \
  --url http://localhost:8000/v1/chat/completions \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --backend vllm \
  --profile chat-singleturn \
  --concurrency 10 \
  --num-requests 100 \
  --api-key test \
  --output results/chat-singleturn_c10.json
```

The paper-facing default is `--arrival steady`, which this runner implements as
closed-loop max-in-flight concurrency. The Poisson and ramp arrival generators
schedule open-loop arrival times but still pass through the same concurrency
semaphore; they are retained for comparison and saturation probes, not as the
main methodology.

For validated APC-aware synthetic replay, use the `*-synth` profile names such
as `swebench-multiturn-synth`, `terminalbench-multiturn-synth`, and
`osworld-multiturn-synth`. The multi-turn `*-synth` profiles set the
paper-facing generator options in the profile itself: code-like filler,
3.8 chars/token calibration, and a 1024-token shared APC prefix.

## Sweep Compilation

```bash
cd benchmarking/inference
python scripts/compile_sweep.py --scope synthetic --dry-run
python scripts/compile_sweep.py --scope synthetic
HOST_FILTER=target_a PY=/path/to/vllm/python \
  bash scripts/bench_orchestrator.sh --jobs scripts/bench_jobs.txt --dry-run
```

`configs/sweep.yaml` keeps the paper hardware grid but does not encode
machine-local model directories. Set the relevant model-root variable before
compiling if your models are not under `/models`, for example:

```bash
APB_TARGET_A_MODEL_ROOT=/models/datacenter \
APB_TARGET_D_MODEL_ROOT=/models/target_d \
python scripts/compile_sweep.py --scope synthetic
```

The generated `scripts/bench_jobs.txt` is an operator artifact and should not be
committed. `bench_orchestrator.sh` consumes that file locally; use
`HOST_FILTER` to select one logical host row set on a given machine. The target
aliases are public hardware-row labels, not machine hostnames.

## MSE Validation

The paired validation script launches vLLM, runs a distributional profile and
its real-trace counterpart on the same server, then tears the server down.

```bash
cd benchmarking/inference
PREFIX_AWARE_SYNTHETIC=on \
SYNTHETIC_STYLE=code \
TARGET_CHARS_PER_TOKEN=3.8 \
SESSIONS=40 \
bash scripts/run_mse_validation.sh \
  /models/Llama-3.1-8B-Instruct 1 Llama-3.1-8B vllm \
  swebench 5 results/mse_validation \
  python 0.85 32768
```

Use `PREFIX_CACHING=off` for the APC-off ablation.
