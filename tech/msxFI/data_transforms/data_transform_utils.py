import numpy as np
import torch
import random
import pickle
import sys
import os
import time
import cProfile, pstats
from .. import fi_config

def get_afloat_bias(num_float, n_exp):
    """
    Extract bias term for AdaptivFloat data format https://arxiv.org/abs/1909.13271

    :param num_float: input value as float
    :param n_exp: number of exponential bits for adaptivFloar data format
    """
    # support for AdaptivFloat [cite]
    bias_temp = np.frexp(num_float.max().item())[1] - 1
    bias = bias_temp - (2**n_exp - 1)
    
    return bias

def get_q_afloat(num_float, n_bits, n_exp, bias):
    """
    Conversion to AdaptivFloat data format https://arxiv.org/abs/1909.13271

    :param num_float: input value as float
    :param n_bits: total number of bits per value (mantissa bits = n_bits - n_exp)
    :param n_exp: number of exponential bits for adaptivFloar data format
    :param bias: input bias term
    """
    # support for AdaptivFloat [cite]
    n_mant = n_bits - 1 - n_exp
  
    # 1. store sign value and do the following part as unsigned value
    sign = torch.sign(num_float)
    num_float = torch.abs(num_float)

    # 2. limits the range of output float point
    min_exp = 0 + bias
    max_exp = 2**(n_exp) - 1 + bias 
    min_value = 2.0**min_exp * (1 + 2.0**(-n_mant))
    max_value = (2.0**max_exp) * (2.0 - 2.0**(-n_mant))
    
    ## 2.1. reduce too small values to zero 
    num_float[num_float < min_value] = 0
    ## 2.2. reduce too large values to max value of output format
    num_float[num_float > max_value] = max_value
    
    # 3. get mant, exp (the format is different from IEEE float)
    mant, exp = np.frexp(num_float.cpu())
    mant = 2 * mant
    exp = exp - 1

    ## 4. quantize mantissa
    scale = 2 ** (-n_mant) ## e.g. 2 bit, scale = 0.25
    if fi_config.true_con:
        mant = ((mant/scale).round()) * scale
        
        mask = mant > 2.0
        mant[mask] = mant[mask] / 4
        exp[mask] = exp[mask] + 2
        
        mask = mant == 2.0
        mant[mask] = mant[mask] / 2
        exp[mask] = exp[mask] + 1
        
        mask = mant >= 1.0
        mant[mask] = mant[mask] - 1
    else:
        mant = ((mant/scale).round()) * scale

        mask_gt2 = mant > 2.0
        mant[mask_gt2] = mant[mask_gt2] / 4
        exp[mask_gt2] = exp[mask_gt2] + 2

        mask_eq2 = mant == 2.0
        mant[mask_eq2] = mant[mask_eq2] / 2
        exp[mask_eq2] = exp[mask_eq2] + 1

        mask_ge1 = mant >= 1.0
        mant[mask_ge1] = mant[mask_ge1] - 1

        mask = torch.zeros(mant.size(0), dtype=torch.int64)

    sign = (sign < 0.).float().to(num_float.device)
    mant = torch.abs(mant).to(num_float.device)
    exp = (exp + abs(bias)).to(num_float.device)

    return sign, mant, exp, mask

def get_floating_point_binary(orig_flt, precision, q_type):
    if q_type == 'bfloat16':
        target_dtype = torch.bfloat16
        int_dtype = torch.uint16
        bit_width = 16
    elif precision == 16:
        target_dtype = torch.float16
        int_dtype = torch.uint16
        bit_width = 16
    elif precision == 32:
        target_dtype = torch.float32
        int_dtype = torch.uint32
        bit_width = 32
    elif precision == 64:
        target_dtype = torch.float64
        int_dtype = torch.int64
        bit_width = 64
    else:
        raise ValueError(f"Unsupported precision: {precision}")
    
    conv_tensor = orig_flt.to(target_dtype)
    int_repr = conv_tensor.view(int_dtype)
    int_vals = int_repr.long().to(fi_config.pt_device)
    bit_positions = torch.arange(bit_width - 1, -1, -1, device=fi_config.pt_device, dtype=torch.int64)
    binary_matrix = ((int_vals.unsqueeze(1) >> bit_positions) & 1).float()
    return binary_matrix
  
def binary_float_conversion(binary_tensor, precision, q_type):
    if precision == 16:
        mant_bits = 10
        exp_bits = 5
        int_type = torch.uint16
        float_type = torch.float16
    elif precision == 32:
        mant_bits = 23
        exp_bits = 8
        int_type = torch.uint32
        float_type = torch.float32
    elif precision == 64:
        mant_bits = 52
        exp_bits = 11
        int_type = torch.int64
        float_type = torch.float64
    
    if q_type == 'bfloat16':
        mant_bits = 7
        exp_bits = 8
        int_type = torch.uint16
        float_type = torch.bfloat16
    
    batch_size = binary_tensor.size(0)
    total_bits = 1 + exp_bits + mant_bits
  
    if precision == 64:
        # Reconstruct the integer using numpy with uint64 to avoid float precision issues
        binary_np = binary_tensor.cpu().numpy().astype(np.uint64)
        powers_np = np.power(np.uint64(2), np.arange(total_bits - 1, -1, -1, dtype=np.uint64))
        int_vals_np = np.sum(binary_np * np.expand_dims(powers_np, 0), axis=1)
    else:
        # Use float64 for bit_powers to maintain precision during summation for 32-bit and 16-bit floats
        bit_powers = torch.pow(2, torch.arange(total_bits - 1, -1, -1, dtype=torch.float64, device=binary_tensor.device))
        int_vals = torch.sum(binary_tensor.to(torch.float64) * bit_powers.unsqueeze(0), dim=1).long()
    
    if q_type == 'bfloat16':
        typed_vals = int_vals.to(torch.uint16)
        float_vals = typed_vals.view(torch.bfloat16).float()
    elif precision == 16:
        # Float16
        typed_vals = int_vals.to(torch.uint16)
        float_vals = typed_vals.view(torch.float16).float()
    elif precision == 32:
        # Float32
        typed_vals = int_vals.to(torch.uint32)
        float_vals = typed_vals.view(torch.float32)
    elif precision == 64:
        # Float64
        numpy_float_vals = int_vals_np.view(np.float64)
        float_vals = torch.from_numpy(numpy_float_vals).to(binary_tensor.device)
    
    return float_vals

def get_integers(float_array, num_bits):
    #using twos complement representation
    max_val = (2**(num_bits-1)) -1
    min_val = -(2**(num_bits-1))
    array_int = float_array.round().to(torch.int32)
    mask = array_int > max_val
    array_int[mask] = max_val
    mask = array_int < min_val
    array_int[mask] = min_val

    return array_int
  
def get_binary_array_mat(orig_flt, rep_conf, int_bits, frac_bits, exp_bias, q_type):
  """
  Format an input float value into binary array for bit-level fault injection

  :param orig_flt: input value (floating point)
  :param rep_conf: mapping from number of bits in input value to number of levels per NVM storage cell (e.g., [2, 2] for a 2 bit input into SLCs)
  :param int_bits: number of integer bits for data format (if applicable)
  :param frac_bits: number of fractional bits for data format (if applicable)
  :param exp_bias: exponent bias for data format (if applicable)
  :param q_type: data format choice (e.g., signed, unsigned, adaptive floating point)
  """

  # format into binary array according to data format
  x = torch.zeros(orig_flt.size()[0], int_bits+frac_bits, device=fi_config.pt_device, dtype=torch.float32)
  current_ =  torch.zeros(orig_flt.size()[0], device=fi_config.pt_device, dtype=torch.float32)
  xid = 0
  if q_type == 'afloat':
    sign, mant, exp, mask = get_q_afloat(orig_flt, int_bits+frac_bits, frac_bits, exp_bias) 
    mant_bits = int_bits-1
    x[:, 0] = sign
    xid = 1
    for mid in range(1, mant_bits+1):
      x[:, xid] = torch.sign(current_ + 0.5**(mid) - mant) <= 0
      current_ = current_ + 0.5**(mid)*x[:, xid]
      xid += 1
    
    current_ =  torch.zeros(orig_flt.size()[0], device=fi_config.pt_device, dtype=torch.float32)
    for eid in list(reversed(range(frac_bits))):
      x[:, xid] = torch.sign(current_ + 2.**(eid) - exp) <= 0
      current_ = current_ + 2.**(eid)*x[:, xid]
      xid += 1
  else:
    mask = 0
    if q_type == 'int':
      num_bits = int_bits + frac_bits
      # Symmetric quantization: scale = abs_max / (2^(n-1) - 1)
      abs_max = torch.max(torch.abs(orig_flt))
      if abs_max == 0:
        scale = 1.0
      else:
        scale = abs_max / (2**(num_bits-1) - 1)

      scaled_flt = orig_flt / scale
      current_ = get_integers(scaled_flt, num_bits)

      # Convert to n-bit unsigned two's complement to avoid sign extension
      int_vals = current_.to(torch.int64)
      if num_bits < 64:
        mask_bits = (1 << num_bits) - 1
        uvals = int_vals & mask_bits
      else:
        uvals = int_vals

      for bit_pos in range(num_bits):
        x[:, num_bits - 1 - bit_pos] = ((uvals >> bit_pos) & 1).float()

      return x, scale
    elif q_type == 'float16':
      binary_matrix = get_floating_point_binary(orig_flt, 16, q_type)
      return binary_matrix, mask
    elif q_type == 'float32':
      binary_matrix = get_floating_point_binary(orig_flt, 32, q_type)
      return binary_matrix, mask
    elif q_type == 'float64':
      binary_matrix = get_floating_point_binary(orig_flt, 64, q_type)
      return binary_matrix, mask
    elif q_type == 'bfloat16':
      binary_matrix = get_floating_point_binary(orig_flt, 16, q_type)
      return binary_matrix, mask
    elif q_type == 'signed':
      x[:, 0] = torch.sign(orig_flt) < 0
      current_ = -1.*2**(int_bits-1)*x[:, 0]
      xid = 1
    for iid in list(reversed(range(int_bits-xid))):
      x[:, xid] = torch.sign(current_ + 2.**(iid) - orig_flt) <= 0
      current_ = current_ + 2.**(iid)*x[:, xid]
      xid += 1
    for fid in range(1, frac_bits+1):
      x[:, xid] = torch.sign(current_ + 0.5**(fid) - orig_flt) <= 0
      current_ = current_ + 0.5**(fid)*x[:, xid]
      xid += 1

  return x, mask


def convert_mlc_mat(num_float, rep_conf, int_bits, frac_bits, exp_bias, q_type):
  """
  Format an entire input matrix into per-memory-cell array under MLC config for bit-level fault injection

  :param num_float: input value (floating point)
  :param rep_conf: mapping from number of bits in input value to number of levels per NVM storage cell (e.g., [2, 2] for a 2 bit input into SLCs)
  :param int_bits: number of integer bits for data format (if applicable)
  :param frac_bits: number of fractional bits for data format (if applicable)
  :param exp_bias: exponent bias for data format (if applicable)
  :param q_type: data format choice (e.g., signed, unsigned, adaptive floating point)
  """
  # format data into MLCs according to data format
  rep_conf_ = torch.from_numpy(rep_conf)
  x_bin, mask = get_binary_array_mat(num_float, rep_conf_, int_bits, frac_bits, exp_bias, q_type)
  x_mlc = torch.zeros(num_float.size()[0], len(rep_conf), device=fi_config.pt_device)
  idx = 0

  rep_conf = torch.tensor(rep_conf, dtype=torch.float32, device=fi_config.pt_device)
  for i in range(len(rep_conf)):
    idx_end = idx + int(torch.log2(rep_conf[i]))
    x_mlc[:, i] = torch.sum(x_bin[:, idx:idx_end]*(2**(torch.arange(int(torch.log2(rep_conf[i])), 0, -1, device=fi_config.pt_device, dtype=torch.float32)-1)), 1)
    idx = idx_end
  return x_mlc, mask


def convert_f_mat(v_mlc, conf, int_bits, frac_bits, exp_bias, q_type, mask=None):
  """
  Convert MLC-packed per-storage-cell values back to floating point values

  :param v_mlc: vector of per-storage-cell values (possible MLC encoding)
  :param conf: mapping from number of bits in input value to number of levels per NVM storage cell (e.g., [2, 2] for a 2 bit input into SLCs)
  :param int_bits: number of integer bits for data format (if applicable)
  :param frac_bits: number of fractional bits for data format (if applicable)
  :param exp_bias: exponent bias for data format (if applicable)
  :param q_type: data format choice (e.g., signed, unsigned, adaptive floating point)
  :param mask: mask for afloat processing (if applicable)
  """
  current = torch.zeros(v_mlc.size()[0], device=fi_config.pt_device, dtype = torch.float32)
  
  if q_type in ['float16', 'bfloat16']:
    total_bits = 16
  elif q_type == 'float32':
    total_bits = 32
  elif q_type == 'float64':
    total_bits = 64
  else:
    total_bits = int_bits + frac_bits
  
  x = torch.zeros(v_mlc.size()[0], total_bits, device=fi_config.pt_device)
  
  idx = 0
  conf = torch.tensor(conf, dtype = torch.float32, device=fi_config.pt_device)
  bin_lut = torch.tensor([[0., 0., 0., 0.], 
                          [0., 0., 0., 1.], 
                          [0., 0., 1., 0.], 
                          [0., 0., 1., 1.], 
                          [0., 1., 0., 0.], 
                          [0., 1., 0., 1.], 
                          [0., 1., 1., 0.], 
                          [0., 1., 1., 1.], 
                          [1., 0., 0., 0.], 
                          [1., 0., 0., 1.], 
                          [1., 0., 1., 0.], 
                          [1., 0., 1., 1.], 
                          [1., 1., 0., 0.], 
                          [1., 1., 0., 1.], 
                          [1., 1., 1., 0.], 
                          [1., 1., 1., 1.]], device=fi_config.pt_device) 

  
  for i in range(len(conf)):
    idx_end = idx + int(torch.log2(conf[i]))
    x[:, idx:idx_end] = bin_lut[v_mlc[:, i].long(), (4-int(torch.log2(conf[i]))):]
    idx = idx_end
  xid = 0
  
  if q_type == 'afloat':
    mant_bits = int_bits-1
    is_valid = (x[:, 0] == 0).clone().detach().to(dtype=torch.float32, device=fi_config.pt_device)
    sign = is_valid*2 - 1
    xid = 1
    mant = torch.zeros(v_mlc.size()[0], device=fi_config.pt_device, dtype=torch.float32)
    for mid in range(1, mant_bits+1):
      is_valid = (x[:, xid] == 1).clone().detach().to(dtype=torch.float32, device=fi_config.pt_device)
      mant = mant + (0.5**(mid))*is_valid
      xid += 1
    if fi_config.true_con and mask is not None:
        mant[mask] = mant[mask] + 1
    else:
        mant = mant + 1
    exp = torch.zeros(v_mlc.size()[0], device=fi_config.pt_device, dtype=torch.float32)
    for eid in list(reversed(range(frac_bits))):
      is_valid = (x[:,xid] == 1).clone().detach().to(dtype=torch.float32, device=fi_config.pt_device)
      exp = exp + (2.**(eid))*is_valid
      xid += 1
    power_exp = torch.exp2(exp+exp_bias) 
    current = sign*power_exp*mant
  elif q_type == 'int':
    num_bits = int_bits + frac_bits
    # Two's complement conversion using bit shift
    sign_bit = (x[:, 0] == 1).to(dtype=torch.int64, device=fi_config.pt_device)
    current = -(1 << (num_bits - 1)) * sign_bit

    if num_bits > 1:
        bit_powers = (1 << torch.arange(num_bits - 2, -1, -1, dtype=torch.int64, device=fi_config.pt_device))
        remaining_bits = (x[:, 1:] == 1).to(dtype=torch.int64, device=fi_config.pt_device)
        current = current + torch.sum(remaining_bits * bit_powers.unsqueeze(0), dim=1)

    # Dequantize: r' = q * scale (scale passed through mask parameter)
    if mask is not None and isinstance(mask, (float, int, torch.Tensor)):
        scale = mask
        current = current.to(torch.float32) * scale
    else:
        current = current.to(torch.float32)

    return current
  elif q_type == 'float16':
    current = binary_float_conversion(x, 16, q_type)
    return current
  elif q_type == 'float32':
    current = binary_float_conversion(x, 32, q_type)
    return current
  elif q_type == 'float64':
    current = binary_float_conversion(x, 64, q_type)
    return current
  elif q_type == 'bfloat16':
    current = binary_float_conversion(x, 16, q_type)
    return current
  else:
    if q_type == 'signed':
      is_valid = (x[:, 0] == 1).clone().detach().to(dtype=torch.float32, device=fi_config.pt_device)
      current = current - (2.**(int_bits-1))*is_valid
      xid = 1
    for iid in list(reversed(range(int_bits-xid))):
      is_valid = (x[:, xid] == 1).clone().detach().to(dtype=torch.float32, device=fi_config.pt_device)
      current = current + (2.**(iid))*is_valid
      xid += 1
    for fid in range(1, frac_bits+1):
      is_valid = (x[:, xid] == 1).clone().detach().to(dtype=torch.float32, device=fi_config.pt_device)
      current = current + (0.5**(fid))*is_valid
      xid += 1
  #print(current)
  return current