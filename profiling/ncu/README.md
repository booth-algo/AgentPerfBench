# NCU And Roofline Profiling

This directory contains the analytical roofline helpers and optional Nsight
Compute capture utilities used for the paper's profiling artifacts.

## Requirements

- NVIDIA GPU with CUDA installed.
- `ncu` available at `/usr/local/cuda/bin/ncu`, or update `NCU_PATH` in
  `roofline_config.py`.
- Python dependencies from `pip install -e .[profiling]`.
- Local model paths under `APB_MODELS_DIR` (default: `/models`) matching
  `MODEL_REGISTRY` in `roofline_config.py`, or update that registry for your
  host.

## Analytical Per-Layer OI

This mode does not require NCU and is useful for checking roofline assumptions.
It writes per-layer analytical JSON files to the output directory.

```bash
python -m profiling.ncu.profile_all_layers \
  --model /models/Llama-3.1-8B-Instruct \
  --batch-size 80 \
  --seq-len 512 \
  --phase prefill \
  --mode analytical \
  --output-dir profiling/results/roofline/raw
```

## NCU Kernel Profiling

`run_ncu.py` captures raw Nsight Compute reports for the model keys in
`MODEL_REGISTRY`. Use `--dry-run` first to inspect the command without launching
GPU work.

```bash
python -m profiling.ncu.run_ncu \
  --model Llama-3.1-8B \
  --batch-sizes 1,16,64 \
  --seq-len 512 \
  --phases prefill,decode \
  --layers 2 \
  --device cuda:0 \
  --output-dir profiling/results/roofline/raw \
  --dry-run
```

Without `--dry-run`, the command writes `.ncu-rep` files. Add `--export-csv` to
also export raw CSV artifacts. These `.ncu-rep` and CSV files are archival raw
captures; the checked-in `parse_ncu.py` path below does not parse them.

## Torch-Profiler JSON Parsing And Plotting

`parse_ncu.py` expects Torch-profiler-style JSON files with a `kernel_summary`
field. It skips non-JSON inputs, so do not point it at `.ncu-rep` or CSV output
from `run_ncu.py`.

```bash
python -m profiling.ncu.parse_ncu \
  --input 'profiling/results/roofline/raw/*.json' \
  --output-dir profiling/results/roofline/parsed

python -m profiling.ncu.plot_roofline \
  --input 'profiling/results/roofline/parsed/*.json' \
  --output-dir profiling/results/roofline/figures
```

For paper-specific per-layer figures, use the specialized scripts in this
directory, such as `parse_per_layer.py` and `plot_per_layer_roofline.py`, with
their `--help` output as the source of truth for required inputs.

Generated profiler outputs are intentionally ignored by git.
