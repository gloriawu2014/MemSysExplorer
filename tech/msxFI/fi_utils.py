import numpy as np
import torch
import random
from statistics import NormalDist
import scipy.stats as ss
import pickle
import sys
import os
import time
import cProfile, pstats
import re
import math
from .data_transforms import * 
from . import fi_config
import importlib.util

# DRAM calibration helpers (configurable via fi_config)
_dram_calibration_scale_cache = {}

def get_error_map(max_lvls_cell, refresh_t=None, vth_sigma=0.05, custom_vdd=None, custom_vpp=None):
  """
  Retrieve the correct per-storage-cell error map for the configured NVM settings according to the maximum levels-per-cell used
  OR generate DRAM error map based on physical parameters

  :param max_lvls_cell: Across the storage settings for fault injection experiment, provide the maximum number of levels-per-cell required (max 16 for 4BPC for provided fault models)
  :param refresh_t: Refresh time in seconds for DRAM models
  :param vth_sigma: Standard deviation of Vth in Volts for DRAM fault rate calculation
  :param custom_vdd: Custom vdd in volts for DRAM models (optional)
  :param custom_vpp: Custom vpp in volts for DRAM models (optional)
  """
  if 'dram' in fi_config.mem_model:
    mem_data_path = os.path.dirname(__file__)
    print("Using DRAM model " + fi_config.mem_model)
    dram_params_path = os.path.join(mem_data_path, 'mem_data', fi_config.mem_dict[fi_config.mem_model])
    try:
        with open(dram_params_path, 'rb') as f:
            dram_params_data = pickle.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"DRAM parameter file not found: {dram_params_path}")
    except pickle.UnpicklingError as e:
        raise ValueError(f"Failed to load DRAM parameters from {dram_params_path}: corrupted or incompatible pickle file. {e}")
    available_sizes = sorted(dram_params_data.keys())
    selected_size = None
    for size in reversed(available_sizes):
      if size <= fi_config.feature_size:
        selected_size = size
        break
    if selected_size is None:
      selected_size = available_sizes[0]
    tech_node_data = dram_params_data[selected_size]
    dist_args = (tech_node_data, fi_config.temperature, selected_size)

    fault_prob = fault_rate_gen(dist_args, refresh_t, vth_sigma, custom_vdd, custom_vpp)
    error_map = np.zeros(1, dtype=object)
    error_map[0] = np.zeros((2, 2))
    error_map[0][0, 1] = 0.0
    error_map[0][1, 0] = fault_prob

  else:
    mem_data_path = os.path.dirname(__file__)
    print("Using NVM model "+ fi_config.mem_model)
    nvm_params_path = os.path.join(mem_data_path, 'mem_data', fi_config.mem_dict[fi_config.mem_model])
    try:
        with open(nvm_params_path, 'rb') as f:
            args_lut = pickle.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"NVM parameter file not found: {nvm_params_path}")
    except pickle.UnpicklingError as e:
        raise ValueError(f"Failed to load NVM parameters from {nvm_params_path}: corrupted or incompatible pickle file. {e}")

    emap_entries = int(np.log2(max_lvls_cell))
    error_map = np.zeros(emap_entries, dtype=object)
    if len(args_lut) < emap_entries:
      raise SystemExit("ERROR: model does not support "+str(emap_entries)+"-bit cells")

    for i in range(emap_entries):
      num_levels = int(2**(i+1))
      error_map[i] = np.zeros((num_levels, 2))

      for j in range(num_levels-1):
        th = get_temp_th(args_lut[i], j)
        dist_args = (th, *args_lut[i][j+1])
        error_map[i][j+1, 0] = fault_rate_gen(dist_args, vth_sigma=vth_sigma)
        dist_args = (th, *args_lut[i][j])
        error_map[i][j, 1] = 1. - fault_rate_gen(dist_args, vth_sigma=vth_sigma)

    if fi_config.Debug:
      for i, emap in enumerate(error_map):
        print("Error map for", int(2**(i+1)), "levels")
        print(emap, "\n\n")

  return error_map

 
def fault_rate_gen(dist_args, refresh_time=None, vth_sigma=0.05, custom_vdd=None, custom_vpp=None):
  """
  Randomly generate fault rate per experiment and storage cell config according to fault model

  :param dist_args: arguments describing the distribution of level-to-level faults (programmed level means and sdevs) for RRAM,
                    or tuple of (tech_node_data, temperature, selected_size) for DRAM
  :param refresh_time: refresh time in seconds for DRAM (required for DRAM models)
  :param vth_sigma: standard deviation of Vth in Volts for DRAM fault rate calculation
  :param custom_vdd: custom vdd in volts for DRAM models (optional)
  :param custom_vpp: custom vpp in volts for DRAM models (optional)
  """
  if 'rram' in fi_config.mem_model:
    x = dist_args[0]
    mu = dist_args[1]
    sigma = dist_args[2]
    cdf = NormalDist(mu, sigma).cdf(x)
    return cdf
  elif 'fefet' in fi_config.mem_model:
    cdf = ss.gamma.cdf(*dist_args)
    return cdf
  elif 'dram' in fi_config.mem_model:
    kB = 1.380649e-23
    q = 1.60217663e-19

    if refresh_time is None:
      raise ValueError("refresh_time is required for DRAM models")
    tech_node_data, temperature, selected_size = dist_args
    cap_F = tech_node_data['CellCap']
    vdd = custom_vdd if custom_vdd is not None else tech_node_data['vdd']

    available_temps = sorted(tech_node_data['Ioff'].keys())
    temp_diffs = [abs(temp - temperature) for temp in available_temps]
    closest_temp_idx = temp_diffs.index(min(temp_diffs))
    selected_temp = available_temps[closest_temp_idx]

    median_Ioff = tech_node_data['Ioff'][selected_temp]

    if custom_vpp is not None:
      vpp = custom_vpp
      SS_V_per_dec = fi_config.SS * 1e-3
      delta_v = vpp - vdd
      exponent = -delta_v / SS_V_per_dec
      median_Ioff = median_Ioff * (10**exponent)

    vth_sigma_mv = vth_sigma * 1000.0
    calibration_scale = get_dram_calibration_scale(dist_args, vth_sigma_mv)
    median_Ioff *= calibration_scale

    if vth_sigma is None or vth_sigma <= 0:
      raise ValueError("Invalid Vth sigma provided for DRAM fault rate calculation")

    type_config = get_dram_type_config(fi_config.mem_model)
    sigma_multiple = type_config['sigma_multiple']

    vth = spread_to_sigma(vth_sigma, sigma_multiple)
    Vt = (kB * selected_temp) / q
    n_factor = fi_config.SS * 1e-3 / (Vt * math.log(10))
    sigma_ln_Ioff = vth / (n_factor * Vt)
    ln_mu = np.log(median_Ioff)
    I_critical = (cap_F * vdd / 2) / refresh_time
    z = (np.log(I_critical) - ln_mu) / sigma_ln_Ioff
    cdf = 1.0 - ss.norm.cdf(z)
    cdf = max(0, cdf)

    print(
      "DRAM Params: "
      f"median_Ioff={median_Ioff:.2e}A, "
      f"I_critical={I_critical:.2e}A, "
      f"Bit-flip Rate (1->0): {cdf*100:.5f}%"
    )

    return cdf
  else:
    raise SystemExit("ERROR: model not defined; please update fi_config.py")
  
# Use this when std dev btwn levels are not even
def solveGauss(mu1, sdev1, mu2, sdev2):
  """
  Helper function to compute intersection of two normal distributions; used to calculate probability of level-to-level fault for specific current/voltage distributions

  :param mu1: mean of first distribution
  :param mu2: mean of second distribution
  :param sdev1: standard dev of first distribution
  :param sdev2: standard dev of second distribution
  """
  a = 1./(2*sdev1**2) - 1./(2*sdev2**2)
  b = mu2/(sdev2**2) - mu1/(sdev1**2)
  c = mu1**2/(2*sdev1**2) -  mu2**2/(2*sdev2**2) - np.log(sdev2/sdev1)
  return np.roots([a, b, c])

def get_temp_th(dist_args, lvl):
  """
  Helper function to compute threshold for detecting a mis-read storage cell according to input fault model and stored value

  :param dist_args: arguments describing the distribution of level-to-level faults
  :param lvl: programmed value to specific memory cell (e.g., 0 or 1 for SLC)
  """
  th = None
  if 'rram' in fi_config.mem_model:
    temp_th = solveGauss(dist_args[lvl][0], dist_args[lvl][1], dist_args[lvl+1][0], dist_args[lvl+1][1])
    for temp in temp_th:
      if temp > dist_args[lvl][0] and temp < dist_args[lvl+1][0]:
        th = temp
        break
    if th is None:
      raise ValueError(f"No valid threshold found for RRAM level {lvl}. Check distribution parameters.")
  elif 'fefet' in fi_config.mem_model:
    th = 0.5*(ss.gamma.median(*dist_args[lvl])+ss.gamma.median(*dist_args[lvl+1]))
  else:
    raise SystemExit("ERROR: model not defined; please update fi_config.py")
  return th

def inject_faults(weights, rep_conf=None, error_map=None):
  """
  Perform fault injection on input MLC-packed data values according to storage settings and fault model (NVM)
  or on binary data according to a single fault rate (DRAM).

  :param weights: MLC-packed data values (NVM) or binary data tensor (DRAM).
  :param rep_conf: storage setting dictating bits-per-cell per data value (NVM). Unused for DRAM.
  :param error_map: generated base fault rates according to storage configs and fault model (NVM), or single fault rate (DRAM).
  """
  
  if 'dram' in fi_config.mem_model:
    random_tensor = torch.rand_like(weights, device=weights.device)
    ones_mask = (weights == 1)
    fault_prob = error_map[0][1, 0]
    fault_mask = (random_tensor < fault_prob)
    actual_faults_to_inject = ones_mask & fault_mask
    weights[actual_faults_to_inject] = 0
    total_num_faults = torch.sum(actual_faults_to_inject).item()
    
    if total_num_faults > 0:
      faulty_indices = torch.where(actual_faults_to_inject)
      affected_elements = torch.unique(faulty_indices[0]).numel()
      total_elements = weights.shape[0]
      print(f"Number of generated faults: {total_num_faults}")
      print(f"Number of affected data values: {affected_elements} (out of {total_elements})")
    else:
      print(f"Number of generated faults: 0")

    return weights
  else:
    # perform fault injection
    total_num_faults = 0

    original_device = weights.device
    weights_cpu = weights.cpu().numpy()

    for cell in range(np.size(rep_conf)):
      max_level = rep_conf[cell] - 1

      cell_error_map_index = int(np.log2(rep_conf[cell])) - 1
      cell_errors = error_map[cell_error_map_index]

      # Loop through all possible levels for cell
      for lvl in range(rep_conf[cell]):
        lvl_cell_addresses = np.where(weights_cpu[:, cell] == lvl)[0]
        if len(lvl_cell_addresses) > 0:
          # Get error probabilities for both up and down transitions
          # the probability of min level going down and max level going up is always 0

          prob_faults_down = cell_errors[lvl][0]
          prob_faults_up = cell_errors[lvl][1]

          # Compute total number of errors for lvl
          num_lvl_faults = int((prob_faults_up+prob_faults_down) * lvl_cell_addresses.size)

          if num_lvl_faults > 0:

            faulty_lvls_indexes = np.random.choice(lvl_cell_addresses, int(num_lvl_faults), replace=False)
            # divide the total number of faults according to up/down fault ratio
            total_prob = prob_faults_up + prob_faults_down
            if total_prob > 0:
              faulty_middle = int(faulty_lvls_indexes.size * prob_faults_up / total_prob)
            else:
              faulty_middle = faulty_lvls_indexes.size // 2

            if prob_faults_up > 0:
              weights_cpu[faulty_lvls_indexes[:faulty_middle], cell] += 1
            if prob_faults_down > 0:
              weights_cpu[faulty_lvls_indexes[faulty_middle:], cell] -= 1

            total_num_faults += num_lvl_faults

    weights = torch.from_numpy(weights_cpu).to(original_device)
    
    if total_num_faults > 0:
      print(f"Number of generated faults: {total_num_faults}")
    else:
      print(f"Number of generated faults: 0")

    for cell_idx in range(np.size(rep_conf)):
      cell_max_level = rep_conf[cell_idx] - 1
      if (torch.sum(weights[:, cell_idx] > cell_max_level) != 0) or (torch.sum(weights[:, cell_idx] < 0) != 0):
        print(f"ERROR: fault injection out of bound for cell {cell_idx}")
        sys.exit(1)

    return weights
  
def import_model_class(py_path):
    """Import model class from the specified Python file."""
    spec = importlib.util.spec_from_file_location("model", py_path)
    model_module = importlib.util.module_from_spec(spec)
    sys.modules["model"] = model_module
    spec.loader.exec_module(model_module)
    return model_module
  
def get_q_type_bit_width(q_type, int_bits=0, frac_bits=0):
    """Returns the total bit width for a given q_type."""
    width_map = {
        'float16': 16,
        'bfloat16': 16,
        'float32': 32,
        'float64': 64,
    }
    if q_type in width_map:
        return width_map[q_type]
    elif q_type in ['signed', 'unsigned', 'afloat', 'int']:
        return int_bits + frac_bits
    else:
        return None # Unknown q_type

def validate_config(args, rep_conf):
    """
    Validates memory and quantization configuration.
    Returns True if valid, False otherwise.
    """
    # --- Validation and Defaulting for q_type vs. int_bits/frac_bits ---
    float_types = ['float16', 'bfloat16', 'float32', 'float64']
    fixed_point_types = ['signed', 'unsigned', 'afloat', 'int']
    
    if args.q_type in float_types:
        if args.int_bits is not None or args.frac_bits is not None:
            print(f"Error: --int_bits and --frac_bits are not applicable for q_type '{args.q_type}'.")
            print("Please remove these arguments when using floating-point q_types.")
            return False
        args.int_bits = 0
        args.frac_bits = 0
    elif args.q_type in fixed_point_types:
        if args.int_bits is None:
            args.int_bits = 2  # Default for fixed-point
            print(f"Info: --int_bits not provided for fixed-point type, using default value of {args.int_bits}.")
        if args.frac_bits is None:
            args.frac_bits = 4  # Default for fixed-point
            print(f"Info: --frac_bits not provided for fixed-point type, using default value of {args.frac_bits}.")

    # --- Validation for NVM capacity ---
    if 'dram' in args.mode:
        return True # This validation is for NVM models

    q_type_width = get_q_type_bit_width(args.q_type, args.int_bits, args.frac_bits)
    if q_type_width is None:
        print(f"ERROR: Unsupported q_type '{args.q_type}'.")
        print(f"Supported q_types: float16, bfloat16, float32, float64, signed, unsigned, afloat, int")
        return False

    rep_conf_capacity = 0
    for level in rep_conf:
        if level <= 1 or (level & (level - 1) != 0):
            print(f"ERROR: rep_conf values must be powers of 2 and > 1. Found: {level}")
            return False
        rep_conf_capacity += math.log2(level)
    
    rep_conf_capacity = int(rep_conf_capacity)

    if rep_conf_capacity > q_type_width:
        print(f"\nERROR: rep_conf capacity ({rep_conf_capacity} bits) is greater than q_type width ({q_type_width} bits).")
        print("This configuration is invalid because you cannot store more bits than the data type provides.")
        return False

    if rep_conf_capacity < q_type_width:
        print(f"\nERROR: rep_conf capacity ({rep_conf_capacity} bits) is less than q_type width ({q_type_width} bits).")
        print("This configuration is invalid because all data bits must be mapped to a cell.")
        return False

    if rep_conf_capacity == q_type_width:
        print(f"Configuration valid: rep_conf capacity ({rep_conf_capacity} bits) matches q_type width ({q_type_width} bits).")

    return True
  
def cdf_tail_for_sigma_multiple(sigma_multiple):
  """One-sided Gaussian tail probability beyond the provided sigma multiple."""
  return 1.0 - ss.norm.cdf(sigma_multiple)


def spread_to_sigma(vth_spread, sigma_multiple):
  """
  Convert a ±Nσ Vth spread to the corresponding 1σ standard deviation.

  Args:
      vth_spread: Total spread in Volts (e.g., 0.05V for ±3.5σ)
      sigma_multiple: Number of sigmas (e.g., 3.5)

  Returns:
      1σ value in Volts
  """
  if sigma_multiple <= 0:
    raise ValueError("Sigma multiple must be positive")
  return vth_spread / sigma_multiple


def _get_dram_calibration_config(tech_node_data, vth_sigma_mv=50):
  """Fetch DRAM calibration parameters based on current mem_model."""
  type_config = get_dram_type_config(fi_config.mem_model)

  nominal_refresh_time = type_config['refresh_time'] * 1e-6
  sigma_multiple = type_config['sigma_multiple']
  nominal_fault_rate = cdf_tail_for_sigma_multiple(sigma_multiple)

  nominal_vdd = tech_node_data['vdd']
  vth_spread = vth_sigma_mv / 1000.0
  nominal_vth_sigma = spread_to_sigma(vth_spread, sigma_multiple)

  return {
      'nominal_vdd': nominal_vdd,
      'nominal_refresh_time': nominal_refresh_time,
      'vth_spread': vth_spread,
      'nominal_vth_sigma': nominal_vth_sigma,
      'sigma_multiple': sigma_multiple,
      'nominal_fault_rate': nominal_fault_rate,
  }


def compute_dram_calibration_scale(dist_args, vth_sigma, target_fault_rate,
                                   target_refresh_time, nominal_vdd):
  """Compute the Ioff scaling needed so the nominal condition meets the target fault rate."""
  tech_node_data, temperature, _ = dist_args

  available_temps = sorted(tech_node_data['Ioff'].keys())
  temp_diffs = [abs(temp - temperature) for temp in available_temps]
  closest_temp_idx = temp_diffs.index(min(temp_diffs))
  selected_temp = available_temps[closest_temp_idx]

  median_Ioff = tech_node_data['Ioff'][selected_temp]
  cap_F = tech_node_data['CellCap']

  kB = 1.380649e-23
  q = 1.60217663e-19
  Vt = (kB * selected_temp) / q
  n_factor = fi_config.SS * 1e-3 / (Vt * math.log(10))
  sigma_ln_Ioff = vth_sigma / (n_factor * Vt)

  z_target = ss.norm.isf(target_fault_rate)
  I_critical = (cap_F * nominal_vdd / 2) / target_refresh_time
  ln_mu_target = np.log(I_critical) - z_target * sigma_ln_Ioff

  return math.exp(ln_mu_target - np.log(median_Ioff))


def get_dram_calibration_scale(dist_args, vth_sigma_mv=50):
  """Retrieve (or compute) the cached Ioff calibration scale for the provided tech data."""
  tech_node_data, temperature, selected_size = dist_args
  params = _get_dram_calibration_config(tech_node_data, vth_sigma_mv)
  nominal_vdd = params['nominal_vdd']
  nominal_refresh_time = params['nominal_refresh_time']
  nominal_vth_sigma = params['nominal_vth_sigma']  # This is already converted to 1σ
  vth_spread = params['vth_spread']
  sigma_multiple = params['sigma_multiple']
  nominal_fault_rate = params['nominal_fault_rate']

  cache_key = (
      id(tech_node_data),
      temperature,
      selected_size,
      nominal_vdd,
      nominal_refresh_time,
      vth_spread,
      sigma_multiple,
  )

  if cache_key not in _dram_calibration_scale_cache:
    scale = compute_dram_calibration_scale(
        dist_args,
        nominal_vth_sigma,  # Use converted 1σ value
        nominal_fault_rate,
        nominal_refresh_time,
        nominal_vdd,
    )
    _dram_calibration_scale_cache[cache_key] = scale
  return _dram_calibration_scale_cache[cache_key]

def get_dram_type_config(mem_model):
  """
  Get default configuration for different DRAM types.
  Returns: dict with 'refresh_time' (in microseconds) and 'sigma_multiple'
  """
  configs = {
      'dram333t': {
          'refresh_time': 501.0,  # microseconds
          'sigma_multiple': 3.5
      },
      'dram3t': {
          'refresh_time': 10.0,  # microseconds
          'sigma_multiple': 3.09  # 99.9% bit yield (double-sided)
      },
      'dram1t': {
          'refresh_time': 64000.0,  # 64ms in microseconds
          'sigma_multiple': 3.09  # 99.9% bit yield (double-sided)
      }
  }
  return configs.get(mem_model, configs['dram333t'])


def calculate_fault_rate(mem_model, refresh_t_us, vth_sigma_mv, vdd=None, vpp=None):
  """
  Calculate fault rate for given DRAM parameters.

  Args:
      mem_model: DRAM type (dram333t, dram3t, dram1t)
      refresh_t_us: Refresh time in microseconds
      vth_sigma_mv: Vth sigma in millivolts
      vdd: Supply voltage in volts (optional)
      vpp: Plate voltage in volts (optional)

  Returns:
      Fault rate as a percentage
  """
  # Temporarily set the memory model
  original_model = fi_config.mem_model
  fi_config.mem_model = mem_model

  try:
      refresh_t = refresh_t_us * 1e-6  # Convert to seconds
      vth_sigma = vth_sigma_mv / 1000.0  # Convert to volts

      # Get error map which calculates fault rate (suppress output)
      import contextlib
      with contextlib.redirect_stdout(open(os.devnull, 'w')):
        error_map = get_error_map(None, refresh_t=refresh_t, vth_sigma=vth_sigma,
                                  custom_vdd=vdd, custom_vpp=vpp)

      # Extract fault probability from error map (1->0 transition)
      fault_prob = error_map[0][1, 0]
      return fault_prob * 100  # Convert to percentage
  finally:
      # Restore original model
      fi_config.mem_model = original_model


def sweep_dram_params(mem_model, target_fr_pct, vth_sigma_mv=50):
  """
  Sweep DRAM parameters to find configurations that achieve target fault rate.

  Args:
      mem_model: DRAM type (dram333t, dram3t, dram1t)
      target_fr_pct: Target fault rate in percentage (e.g., 0.1 for 0.1%)
      vth_sigma_mv: Vth sigma in mV (default: 50mV)

  Returns:
      List of tuples (refresh_t_us, vdd, vpp, actual_fr_pct) that meet target
  """
  mem_data_path = os.path.dirname(__file__)
  dram_params_path = os.path.join(mem_data_path, 'mem_data', fi_config.mem_dict[mem_model])
  try:
      with open(dram_params_path, 'rb') as f:
          dram_params_data = pickle.load(f)
  except FileNotFoundError:
      raise FileNotFoundError(f"DRAM parameter file not found: {dram_params_path}")
  except pickle.UnpicklingError as e:
      raise ValueError(f"Failed to load DRAM parameters from {dram_params_path}: {e}")
  available_sizes = sorted(dram_params_data.keys())
  selected_size = None
  for size in reversed(available_sizes):
    if size <= fi_config.feature_size:
      selected_size = size
      break
  if selected_size is None:
    selected_size = available_sizes[0]
  tech_node_data = dram_params_data[selected_size]
  vdd_fixed = tech_node_data['vdd']

  print(f"\n{'='*70}")
  print(f"Sweeping {mem_model.upper()} parameters for target fault rate: {target_fr_pct}%")
  print(f"Sweep ranges: refresh_t=[1us, 64ms], Vpp=[1.2V, 2.0V], Vdd={vdd_fixed}V")
  print(f"{'='*70}\n")

  results = []

  refresh_times = list(range(1, 1001))
  refresh_times.extend(range(1010, 64001, 10))
  vpps = np.arange(0.8, 2.05, 0.05).tolist()

  for rt in refresh_times:
      for vpp in vpps:
          try:
              fr = calculate_fault_rate(mem_model, rt, vth_sigma_mv, vdd=vdd_fixed, vpp=vpp)
              if abs(fr - target_fr_pct) < target_fr_pct * 0.05:
                  results.append((rt, vdd_fixed, vpp, fr))
          except Exception:
              pass

  print(f"\n{'='*70}")
  print(f"Found {len(results)} configuration(s) matching target fault rate (±5%)")
  print(f"{'='*70}\n")

  return results


def filter_top_configs_per_vpp(results, target_fr, top_n=3):
  """
  Filter results to show only top N closest configurations per VPP value.

  Args:
      results: List of tuples (refresh_t, vdd, vpp, fr)
      target_fr: Target fault rate for comparison
      top_n: Number of top configurations to keep per VPP

  Returns:
      Filtered list of tuples
  """
  from collections import defaultdict

  vpp_groups = defaultdict(list)
  for config in results:
    vpp = config[2]
    vpp_groups[vpp].append(config)

  filtered_results = []
  for vpp in sorted(vpp_groups.keys()):
    configs = vpp_groups[vpp]
    configs_sorted = sorted(configs, key=lambda x: abs(x[3] - target_fr))
    filtered_results.extend(configs_sorted[:top_n])

  return filtered_results

