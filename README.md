# Marimo EE Lab

This repository contains Marimo notebooks for electrical engineering and electronics lab work.
It is intended for interactive exploration, simulation, and visualization of EE/ECE concepts.
## What This Repository Is For

- Control systems notebooks
- Circuit and signal-processing experiments
- Numerical simulation and visualization
- Lightweight lab-style reports in notebook form

## Current Content

- [control-system/PID_controller.py](control-system/PID_controller.py): a control-systems notebook for P, PI, and PID analysis

## Using the Notebooks

Open any notebook with Marimo from the repository root:

```bash
marimo edit path/to/notebook.py
```

To run a notebook without the editor:

```bash
marimo run path/to/notebook.py
```

## Requirements

Typical notebooks in this repo may use:

- Python 3.10+
- marimo
- numpy
- scipy
- matplotlib

Install the common dependencies with:

```bash
pip install marimo numpy scipy matplotlib
```

## Project Structure

```text
.
├── README.md
└── control-system/
    └── PID_controller.py
```

## Notes

- Individual notebooks may add their own domain-specific dependencies.
- Some notebooks are designed for headless rendering and may use stable plotting backends.
