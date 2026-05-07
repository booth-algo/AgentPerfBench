# Artifact Submission Scope

This repository is the code artifact for AgentPerfBench benchmarking and
profiling. It contains the components needed to reproduce the paper's benchmark
and profiling methodology.

## Included

- OpenAI-compatible inference benchmark runner.
- Real-trace and distributional synthetic workload profile code.
- APC-aware synthetic profile support and MSE validation scripts.
- Sweep manifest, job compiler, and local generated-job orchestrator for
  benchmark collection.
- Unit tests for workload/profile behavior and cache metadata.
- Nsight Compute and roofline profiling utilities.
- Compact distribution JSON artifacts required by synthetic replay.

## Not Included

- Generated benchmark result JSONs.
- Raw SWE-bench, TerminalBench, and OSWorld trajectory JSONL files.
- Cloud sync credentials, private artifact mirrors, and machine-local logs.

## Data Pointers

The public dataset artifact is published on Hugging Face:

```text
https://huggingface.co/datasets/agent-perf-bench/AgentPerfBench
```

The dataset repository contains the published benchmark tables and the
paper-facing MSE validation subset: paired H100/Llama-3.1-8B real-trace and
synthetic-replay runs for SWE-bench and TerminalBench, including the
no-shared-prefix, APC-off, morphology, and shared-prefix ablations.
