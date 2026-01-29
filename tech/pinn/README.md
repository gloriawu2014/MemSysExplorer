# PINN-based RRAM Characterization and NVSim Configuration

This module utilizes a Physics-Informed Neural Network (PINN/[PIGen](https://ieeexplore.ieee.org/document/11240964)) to model the dynamic behavior of RRAM devices. It simulates the resistance switching characteristics of different materials (HfO2, Al2O3, TiO2) under various operating conditions and automatically generates compatible cell and configuration files for [NVSim](https://github.com/SEAL-UCSB/NVSim).

## Overview

Traditional device modeling can be computationally expensive or rely on simplified analytical models. This module uses a trained PINN to accurately predict the time-evolution of the conductive filament gap and the resulting current/resistance in RRAM devices.

The workflow consists of:
1.  **Input Specification**: Define target voltage parameters (Set/Reset voltages, pulse widths) via a sample cell file.
2.  **PINN Simulation**: The `RRAMCycleEvaluator` uses the neural network to simulate read/write cycles and extract key metrics like On/Off resistance at different voltages.
3.  **NVSim Integration**: `generate_nvsim_config.py` converts these characteristics into `.cell` and `.cfg` files, ready for circuit-level simulation in NVSim.

## Directory Structure

- `checkpoints/`: Contains the pre-trained PINN model weights (`checkpoint.pth`).
- `data/`: Contains reference data (`rram.mat`) used for normalization/scaling.
- `generate_nvsim_config.py`: The main entry point for the generation workflow.
- `rram_cycle_evaluator.py`: The core engine that runs the PINN inference.

## Usage

The primary interface is `generate_nvsim_config.py`. It can be run from the project root or the `pinn` directory.

### Basic Usage

Generate NVSim configuration files based on the default sample cell:

```bash
# From project root
python tech/pinn/generate_nvsim_config.py
```

### Advanced Usage

Run NVSim simulation immediately after generation, specify materials, and use a custom sample cell:

```bash
python tech/pinn/generate_nvsim_config.py \
  --run-nvsim \
  --materials HfO2 TiO2 \
  --sample-cell sample_RRAM.cell \
  --template-config tech/ArrayCharacterization/sample_configs/sample_RRAM_32nm.cfg
```

### Arguments

- `--model-path`: Path to the `.pth` checkpoint (default: `tech/pinn/checkpoints/checkpoint.pth`).
- `--data-path`: Path to the `.mat` data file (default: `tech/pinn/data/rram.mat`).
- `--tech-dir`: Path to the `tech` directory (default: `tech`).
- `--sample-cell`: Name of the sample cell file in `tech/ArrayCharacterization/sample_cells/` to use as a template for voltage parameters (default: `sample_RRAM.cell`).
- `--template-config`: Path to the NVSim configuration template (default: `tech/ArrayCharacterization/sample_configs/sample_RRAM_32nm.cfg`).
- `--materials`: List of materials to evaluate (choices: `HfO2`, `Al2O3`, `TiO2`).
- `--run-nvsim`: If set, automatically runs the NVSim binary with the generated configurations.

## Dependencies

This module requires the following Python packages:
- `torch`
- `numpy`
- `scipy`
- `sklearn` (scikit-learn)

## How It Works

1.  **Gap Evolution**: The `RRAM_PINN` model predicts the evolution of the conductive filament gap ($g$) as a function of time ($t$), time-step ($dt$), voltage ($V$), and material properties.
2.  **Current Prediction**: The `MLP_Current` model predicts the device current ($I$) based on the gap, voltage, and initial state.
3.  **Cycle Simulation**: `RRAMCycleEvaluator` strings these predictions together to simulate a full operation cycle (Read -> Set -> Read -> Reset -> Read) to characterize the device's resistance states.

## References

1. Dong, Xiangyu, et al. "Nvsim: A circuit-level performance, energy, and area model for emerging nonvolatile memory." IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems 31.7 (2012): 994-1007.
2. Zhang, Zihan and Donato, Marco. "PIGen: Accelerating ReRAM Co-Design via Generative Physics-Informed Modeling." 2025 IEEE/ACM International Conference On Computer Aided Design (ICCAD). IEEE, 2025.
