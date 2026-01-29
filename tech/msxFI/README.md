# msxFI - Memory Fault Injection Framework

Welcome to msxFI, a simple interface for quantifying the impact of memory faults and failure modes on application accuracy.

An evolution of the [Ares](https://ieeexplore.ieee.org/document/8465834) and [nvmFI](https://ieeexplore.ieee.org/document/9773239) frameworks, msxFI provides an extended PyTorch interface to model and inject memory faults. It supports fault models for both NVMs (with configurable MLC) and DRAM (based on operating conditions).

## Features

- **NVM Fault Injection**: Simulate faults in RRAM and other non-volatile memories with configurable MLC programming.
- **DRAM Fault Modeling**: Inject faults based on operating conditions and technology parameters for 1T/3T DRAM/eDRAM.
- **PyTorch Integration**: Seamlessly inject faults into neural network trainable parameters.
- **Quantization Support**: Supports a wide range of data formats, including IEEE floating-point (`float16`, `bfloat16`, `float32`, `float64`) and various fixed-point types (`signed`, `unsigned`, `int`, `afloat`).
- **Simplified Interface**: User-friendly command-line interface with clear memory model naming and automatic parameter handling.
- **Flexible Model Loading**: Supports custom model architectures via dynamic module importing.

## Dependencies

PyTorch, numpy, scipy, and (optionally, for graph analytics tasks) snapPY.

## Core Concepts

msxFI operates through a sequential pipeline to simulate memory faults. Understanding this flow is key to configuring your experiments correctly.

```mermaid
graph LR
    A["Input Data<br>(Matrix / Weights)"] --> B{"Quantize &<br/>Convert to Bits"};
    B --> C("NVM Path<br>--rep_conf");
    B --> D("DRAM Path<br>--refresh_t");
    C --> E{"Decode Bits"};
    D --> E;
    E --> F["Faulty Data<br>(Matrix / .pth File)"];

    style A fill:#f9f,stroke:#333,stroke-width:2px
    style F fill:#f9f,stroke:#333,stroke-width:2px
```

The process consists of two main stages:
1.  **Data Representation**: The input floating-point data is first quantized into a lower-precision format (e.g., fixed-point or float16). This quantized data is then converted into a binary bitstream. For NVM models, these bits are packed into multi-level cells based on your `--rep_conf`. For DRAM models, they are treated as a simple sequence of bits.
2.  **Fault Injection**: Faults are injected into this bit-level representation.
    -   **NVM Models** (e.g., `rram_mlc`): Use pre-calibrated error maps based on physical device characteristics. The `--rep_conf` parameter is crucial here.
    -   **DRAM Models** (e.g., `dram1t`): Generate faults based on a calculated bit-flip probability, which is determined by the `--refresh_t` and technology parameters.

## Usage Guide

This guide provides practical examples for common use cases. All parameters are detailed in the "Parameter Reference" section below.

### Use Case 1: NVM Matrix Fault Injection

This mode injects faults directly into a data matrix using an NVM model like RRAM.

**Command:**
```bash
python run_msxfi.py \
  --mode rram_mlc \
  --q_type afloat \
  --int_bits 2 \
  --frac_bits 4 \
  --rep_conf 8 8
```

**Key Parameters for this use case:**
- `--mode`: Specifies the memory model (`rram_mlc`, `fefet_200d`, etc.).
- `--q_type`, `--int_bits`, `--frac_bits`: Define the data quantization format.
- `--rep_conf`: Defines how bits are packed into NVM cells. This is critical for NVM models. See the "Configuration Deep Dive" for validation rules.

### Use Case 2: DRAM Matrix Fault Injection

This mode simulates DRAM faults in a data matrix based on operating conditions.

**Command:**
```bash
python run_msxfi.py \
  --mode dram1t \
  --refresh_t 80 \
  --vth_sigma 50 \
  --q_type signed \
  --int_bits 3 \
  --frac_bits 5
```

**Key Parameters for this use case:**
- `--mode`: Specifies a DRAM model (`dram1t`, `dram3t`, `dram333t`).
- `--refresh_t`: **Required for DRAM.** Refresh time in microseconds (e.g., `80`). This is a crucial parameter for DRAM fault modeling.
- `--vth_sigma`: Standard deviation of threshold voltage (Vth) in mV.
- `--vdd`: **Optional.** Custom vdd in volts.
- `--vpp`: **Optional.** Custom vpp in volts.
- Note: `--rep_conf` is not used for DRAM models.

### Use Case 2b: Finding DRAM Parameters for Target Fault Rate

This mode performs a parameter sweep to find DRAM configurations (refresh time and vpp) that achieve a target fault rate. This is useful when you want to explore the design space or find operating points for a specific reliability target.

**Command:**
```bash
python run_msxfi.py \
  --mode dram333t \
  --target_fr 0.01 \
```

**Output Example:**
```
Top configurations (showing up to 3 closest matches per Vpp):
Total configurations found: 32, displaying: 12

Vpp (V)   Refresh (us)   Fault Rate (%)    Error (%)
-----------------------------------------------------------------
1.25      10.0           0.010405          0.000405

1.35      129.0          0.010271          0.000271

1.40      462.0          0.009893          0.000107
1.40      463.0          0.010127          0.000127
1.40      461.0          0.009663          0.000337
```

**Key Parameters for this use case:**
- `--target_fr`: **Required for sweep mode.** Target fault rate in percentage (e.g., `0.01` for 0.01%).
- `--mode`: Must be a DRAM model (`dram1t`, `dram3t`, `dram333t`).
- The sweep searches refresh times from 1us to 64ms and vpp voltages from 0.8V to 2.0V.
- Results show up to 3 closest matches per vpp value, sorted by error from target.
- When using `--target_fr`, other fault injection modes are disabled (matrix/DNN FI).

### Use Case 3: DNN Fault Injection

Enable DNN fault injection using the `--eval_dnn` flag. This injects faults into the weights of a pre-trained model. The same parameters for matrix injection are used to control the fault characteristics.

**Command (DRAM example):**
```bash
python run_msxfi.py \
  --mode dram1t \
  --eval_dnn \
  --model /path/to/your/model.pth \
  --model_def /path/to/your/model.py \
  --model_class YourModelClass \
  --refresh_t 100 \
  --vth_sigma 50 \
  --q_type afloat \
  --int_bits 2 \
  --frac_bits 4
```

**Key Parameters for this use case:**
- `--eval_dnn`: Activates DNN fault injection mode.
- `--model`: Path to your pre-trained PyTorch model file (`.pth`).
- `--model_def`: Path to the Python script containing your model's class definition.
- `--model_class`: The name of your model's class within the definition file.

## Parameter Reference

Below is a summary of all command-line parameters for `run_msxfi.py`:

| Parameter         | Description                                                                                                | Default Value | Applicability       |
|-------------------|------------------------------------------------------------------------------------------------------------|---------------|---------------------|
| `--mode`          | Specifies the memory model (e.g., `rram_mlc`, `fefet_100d`, `dram1t`). See `fi_config.py` for all options. | (Required)    | All Modes           |
| `--q_type`        | Quantization type. **IEEE**: `float16`, `bfloat16`, `float32`, `float64`. **Fixed-point**: `signed`, `unsigned`, `afloat`, `int`. | `afloat`      | All Modes           |
| `--int_bits`      | Defines integer bits for quantization.                                                                       | 2             | All Modes           |
| `--frac_bits`     | Defines fractional bits for quantization. For `afloat`, these are exponent bits.                           | 4             | All Modes           |
| `--seed`          | Seed for random number generation for reproducibility.                                                       | random        | All Modes           |
| `--matrix_size`   | Size of the test matrix in matrix fault injection mode.                                                      | 1000          | Matrix FI           |
| `--eval_dnn`      | Enables DNN fault injection mode.                                                                            | N/A (flag)    | DNN FI              |
| `--model`         | Path to the pre-trained DNN model (`.pth` file).                                                               | N/A           | DNN FI              |
| `--model_def`     | Path to the Python file with the model's class definition.                                                   | N/A           | DNN FI              |
| `--refresh_t`     | Refresh time in microseconds (required for DRAM models).                                                     | N/A           | DRAM models         |
| `--vth_sigma`     | Standard deviation of threshold voltage (Vth) in mV.                                                       | 50            | DRAM models         |
| `--vdd`           | Custom vdd in volts for DRAM models. If not provided, uses default vdd from pickle file.        | N/A           | DRAM models         |
| `--vpp`           | Custom vpp in volts for DRAM models.                                                       | N/A           | DRAM models         |
| `--target_fr`     | Target fault rate in percentage for parameter sweep mode (e.g., `0.01` for 0.01%). Enables automatic search for matching refresh_t and vpp configurations. | N/A           | DRAM models         |
| `--rep_conf`      | Rep conf for MLC encoding. Space-separated integers (e.g., `2 2 4`).                         | `[8, 8]`      | NVM models          |
| `--model_class`   | Name of the model class in the model definition file.                                            | N/A           | DNN FI              |

## Configuration Deep Dive

### Hardcoded Parameters (`fi_config.py`)

Certain global parameters that affect the underlying fault models, especially for DRAM, are configured directly within `msxFI/fi_config.py`. You will need to edit this file to change them.

Key parameters in `fi_config.py` include:
- `temperature`: Operating temperature in Kelvin (e.g., `300`). Affects DRAM fault rates.
- `feature_size`: Technology node in nm (e.g., `16`). Used for selecting appropriate DRAM parameters.
- `SS`: Subthreshold Swing in mV/dec (e.g., `70`). Affects DRAM fault rate calculations.

### NVM Configuration Validation (`--rep_conf`)

msxFI automatically validates your configuration to prevent common errors, primarily focusing on the interaction between data width (`--q_type`) and NVM cell capacity (`--rep_conf`). This validation applies to NVM models only (e.g., `rram_mlc`, `fefet_*`).

#### Step 1: Determine Data Bit Width (`--q_type`)

First, `msxFI` determines the total number of bits for your chosen data type.

| `q_type`   | Total Bits                | Description                                        |
| :--------- | :------------------------ | :------------------------------------------------- |
| `float16`  | 16                        | IEEE 754 half precision                            |
| `bfloat16` | 16                        | Brain Floating Point                               |
| `float32`  | 32                        | IEEE 754 single precision                          |
| `float64`  | 64                        | IEEE 754 double precision                          |
| `signed`   | `int_bits` + `frac_bits`  | Signed fixed-point                                 |
| `unsigned` | `int_bits` + `frac_bits`  | Unsigned fixed-point                               |
| `int`      | `int_bits` + `frac_bits`  | Two's complement integer                           |
| `afloat`   | `int_bits` + `frac_bits`  | [AdaptivFloat](https://arxiv.org/abs/1909.13271) format |

#### Step 2: Calculate NVM Cell Capacity (`--rep_conf`)

Next, it calculates the total bit capacity of the NVM cells defined by `--rep_conf`.

- Each value in `--rep_conf` must be a **power of 2** (e.g., 2, 4, 8), representing the number of levels in a cell.
- The total capacity is the **sum of bits** from all cells: `sum(log2(value) for each value in rep_conf)`.

#### Step 3: Validate Capacity vs. Data Width

Finally, the two values are compared:

- **Error**: `Cell Capacity > Data Width`
  - This is an invalid configuration. You cannot store more bits than the data type provides.
- **Error**: `Cell Capacity < Data Width`
  - This is an invalid configuration. All data bits must be mapped to a cell.
- **Valid**: `Cell Capacity == Data Width`
  - The configuration is valid with no precision loss.

#### Configuration Examples

**Valid Configurations:**
```bash
# Float16 (16 bits) with exact 16-bit rep_conf capacity
# 16 cells, each storing log2(2)=1 bit. Total = 16 * 1 = 16 bits.
python run_msxfi.py --mode rram_mlc --q_type float16 --rep_conf 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2

# Fixed-point (2 int + 4 frac = 6 bits) with matching 6-bit capacity
# 3 cells, each storing log2(4)=2 bits. Total = 3 * 2 = 6 bits.
python run_msxfi.py --mode rram_mlc --q_type signed --int_bits 2 --frac_bits 4 --rep_conf 4 4 4
```

**Invalid Configurations (will be rejected):**
```bash
# ERROR: rep_conf values must be powers of 2 (3 is invalid)
python run_msxfi.py --mode rram_mlc --q_type float16 --rep_conf 3 3 3

# ERROR: rep_conf capacity (18 bits) > q_type width (16 bits)
python run_msxfi.py --mode rram_mlc --q_type float16 --rep_conf 8 8 8 8 8 8

# ERROR: rep_conf capacity (16 bits) < q_type width (32 bits)
python run_msxfi.py --mode rram_mlc --q_type float32 --rep_conf 4 4 4 4 4 4 4 4
```

## Updating DRAM Parameters

If you modify the underlying technology parameters in the C++ source files, you must regenerate the corresponding pickle files for the changes to take effect in `msxFI`.

**When to Regenerate:**
- After modifying `currentOffNmos` or `vdd` values in `Technology.cpp`.
- After updating cell parameters in `sample_cells/sample_edram1ts/` or `sample_cells/sample_edram3ts/`.

**To regenerate DRAM parameter files:**
```bash
cd msxFI/data_transforms/
python gen_dram_params.py
```
This updates `dram1t_args.p` and `dram3t_args.p` in `msxFI/mem_data/`.

## End-to-End Example: LeNet on MNIST

The `msxFI/example_nn/lenet/` directory provides a complete, runnable example of using `msxFI` to inject faults into a LeNet CNN and evaluate its impact on accuracy. This is the best place to see the framework in action. A similar example for ResNet-18 on CIFAR-10 is available in `msxFI/example_nn/resnet18/`.

The example includes scripts to train the model and to run the fault-injection evaluation. For detailed instructions, please refer to the dedicated README in that directory:

**[Click here for the LeNet Example README](example_nn/lenet/README.md)**

## File Structure

Key directories and files:
```
msxFI/
├── mem_data/           # Contains memory model parameter files (e.g., *.p files)
├── data_transforms/    # Data processing utilities
│   ├── gen_dram_params.py      # Script to regenerate DRAM pickle files
│   ├── data_transform_utils.py # Quantization and data format conversion utilities
│   ├── bitmask_utils.py        # Sparse encoding utilities for bitmask format
│   └── graph_utils.py          # Graph processing utilities (requires snapPY)
├── example_nn/         # Example neural network
│   ├── lenet/          # LeNet CNN implementation and training scripts
│   └── resnet18/       # ResNet-18 CNN implementation and training scripts
├── address_mapping.py  # Physical address mapping for eDRAM memory hierarchy
├── fi_config.py        # Core configuration parameters (temperature, feature_size, etc.)
├── fi_utils.py         # Utilities for fault injection, error map generation
└── fault_injection.py  # Main fault injection logic
run_msxfi.py            # Command-line interface script
```

---

## Output

- **Matrix Mode**: Displays original vs. faulty matrices, highlighting differences.
- **DNN Mode**: Saves fault-injected model weights to the original model's directory. Filenames are generated to include key parameters for easy identification.
  - **NVM Example**: `modelname_rram_mlc_s0_qafloat_i2_f4.pth`
  - **DRAM Example**: `modelname_dram1t_s0_qfloat32_rt80_vpp1.4.pth`

## Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

## Citation

If using msxFI for research, please cite the relevant papers. Contact authors for details.

## Authors

Original [nvmFI](https://ieeexplore.ieee.org/document/9773239) framework by Lillian Pentecost and Marco Donato, 2020.

Afloat and FeFET implementations by Emma Claus, 2023

eDRAM extensions and interface improvements by Zihan Zhang and David Kong, 2025.