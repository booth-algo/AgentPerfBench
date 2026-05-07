# AgentPerfBench

AgentPerfBench is the benchmark and profiling artifact repository for the
agentic serving paper. This repo keeps the tooling needed to reproduce the
benchmark/profiling data. Generated result dumps are published separately and
kept out of version control.

## Contents

- `benchmarking/inference/` — OpenAI-compatible serving benchmark runner,
  workload profiles, trace replay, distributional synthetic replay, and sweep
  scripts.
- `benchmarking/inference/data/distributions/` — compact distributional
  workload artifacts used by the synthetic profiles.
- `benchmarking/tests/` — unit tests for profile registry, distributional
  workloads, real-trace loading, and cache metadata.
- `profiling/ncu/` — NVIDIA Nsight Compute and roofline profiling tools used
  for per-layer/kernel profiling.
- `docs/` — submission scope notes and artifact pointers.

Raw trajectories and full benchmark outputs are intentionally not committed
here. They are published in the Hugging Face dataset artifact; see
`docs/artifact-submission/scope.md`.

## Quickstart

These commands exercise the repository without a GPU or model server:

```bash
git clone https://github.com/booth-algo/AgentPerfBench.git
cd AgentPerfBench
python -m venv .venv
. .venv/bin/activate
pip install -e .[dev]
PYTHONPATH=benchmarking/inference python -m pytest -q benchmarking/tests
PYTHONPATH=benchmarking/inference python -m src.benchmark.runner --list-profiles
python benchmarking/inference/scripts/compile_sweep.py --scope synthetic --dry-run
```

For full benchmark data collection against an OpenAI-compatible server, install
runtime extras:

```bash
pip install -e .[benchmark]
python -c "import src.benchmark.runner"
```

For NCU/roofline plotting, install profiling extras:

```bash
pip install -e .[profiling]
```

## Run A Benchmark

Start an OpenAI-compatible server such as vLLM, then run:

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

Synthetic profiles validated for the paper use the `*-synth` names, for
example `swebench-multiturn-synth`. The multi-turn synthetic profiles carry the
APC-aware replay settings directly: code-like filler, 3.8 chars/token
calibration, and a 1024-token shared prefix. `chat-singleturn-synth` is a
synthetic-scope ShareGPT baseline.

## Compile And Run A Sweep

```bash
cd benchmarking/inference
python scripts/compile_sweep.py --scope synthetic
HOST_FILTER=target_a PY=/path/to/vllm/python \
  bash scripts/bench_orchestrator.sh --jobs scripts/bench_jobs.txt
```

The public sweep manifest uses model-root placeholders rather than
machine-local paths. Set `APB_TARGET_A_MODEL_ROOT`, `APB_TARGET_B_MODEL_ROOT`,
`APB_TARGET_C_MODEL_ROOT`, or `APB_TARGET_D_MODEL_ROOT` before compiling if your
model cache is not mounted at `/models`. The logical target names are stable
operator aliases for the public hardware families in
`benchmarking/inference/configs/sweep.yaml`.

Use `--dry-run` on `bench_orchestrator.sh` to inspect the generated launches
without starting model servers.

## Run Tests

```bash
PYTHONPATH=benchmarking/inference python -m pytest -q benchmarking/tests
```

## Profiling

NCU profiling entrypoints live in `profiling/ncu/`. See
`profiling/ncu/README.md` for the commands used for layer/kernel profiling.
