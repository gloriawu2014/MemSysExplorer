import os
import torch
import numpy as np
import argparse
from scipy.io import loadmat
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def get_standardized_gap(value, target_min=-1, target_max=1, gap_min=-1, gap_max=1):
    gap_scale = (target_max - target_min) / (gap_max - gap_min)
    gap_normalized = target_min + (value - gap_min) * gap_scale
    return gap_normalized

class Constants:
    """
    Physical constants and material parameters for RRAM simulation.
    """
    def __init__(self, material='HfO2'):
        self.kb = 1.380649e-23
        self.q = 1.60217663e-19
        self.T0 = 273 + 25
        self.tox = 5e-9
        self.a0 = 0.25e-9
        self.gap_min = 0.1e-9
        self.gap_max = 1.7e-9
        self.Tau_th = 2.3e-10
        self.g1 = 1e-9

        self.material_params = {
            'HfO2': {'Eag': 1.241, 'Ear': 1.24, 'Cth': 3.05e-18},
            'Al2O3': {'Eag': 1.001, 'Ear': 1.0, 'Cth': 2.98e-18},
            'TiO2': {'Eag': 1.501, 'Ear': 1.50, 'Cth': 3.18e-18}
        }

        self.update_material(material)

        self.gamma0_pos = 15
        self.gamma0_neg = 8.5
        self.beta = 1.25
        self.Vel0_pos = 120
        self.Vel0_neg = 150

    def update_material(self, material):
        if material not in self.material_params:
            raise ValueError(f"Unknown material: {material}")

        params = self.material_params[material]
        self.Eag = params['Eag']
        self.Ear = params['Ear']
        self.Cth = params['Cth']
        self.current_material = material

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Optional, Union
from collections import defaultdict
import logging

def format_voltage(voltage: Union[float, np.ndarray, torch.Tensor], precision: int = 2) -> float:
    if isinstance(voltage, (np.ndarray, torch.Tensor)):
        voltage = float(voltage)
    return round(voltage, precision)

def init_weights(module):
    if isinstance(module, nn.Linear):
        gain = nn.init.calculate_gain('tanh')
        nn.init.xavier_uniform_(module.weight, gain=gain/2)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)

class RRAM_PINN(nn.Module):
    """
    Physics-Informed Neural Network for predicting RRAM Gap evolution.
    
    Inputs:
        - t: Time
        - dt: Time step
        - v: Voltage
        - material_idx: Index of the material type
        
    Outputs:
        - gap_pred: Predicted gap size (standardized)
    """
    def __init__(self, hidden_size=32, embedding_size=5, const=None):
        super(RRAM_PINN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.const = const
        
        self.material_embedding = nn.Embedding(3, self.embedding_size)

        self.timestep_encoder = nn.Sequential(
            nn.Linear(1, 2),
            nn.LeakyReLU(0.2)
        )
        
        self.gru = nn.GRU(
            input_size=self.embedding_size + 2 + 2,
            hidden_size=self.hidden_size,
            num_layers=1,
            batch_first=True
        )
        
        self.feature_enhancer = nn.Sequential(
            nn.Linear(self.hidden_size + 2, self.hidden_size),
            nn.LeakyReLU(0.2),
            nn.LayerNorm(self.hidden_size)
        )
        
        self.gap_predictor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size//2),
            nn.Tanh(),
            nn.Linear(hidden_size//2, 1),
            nn.Tanh()
        )
        
        self.apply(init_weights)
    
    def forward(self, t, dt, v, material_idx, hidden=None):
        material_emb = self.material_embedding(material_idx)
        material_emb = material_emb.expand(len(t), -1)

        dt = dt.unsqueeze(-1)
        dt_encoded = torch.sign(dt) * torch.log1p(torch.abs(dt) * 1e12)
        dt_features = self.timestep_encoder(dt_encoded)

        t = t.unsqueeze(-1)
        v = v.unsqueeze(-1)

        x = torch.cat([t, v, material_emb, dt_features], dim=-1).unsqueeze(0)
        gru_out, _ = self.gru(x)
        gru_features = gru_out.squeeze(0)

        enhanced_features = self.feature_enhancer(
            torch.cat([gru_features, dt_features], dim=-1)
        )

        gap_pred = self.gap_predictor(enhanced_features).squeeze(-1)
        return gap_pred

    
class MLP_Current(nn.Module):
    """
    MLP for predicting RRAM Current based on Gap state.
    
    Inputs:
        - gap: Gap size
        - v: Voltage
        - initial_I: Initial current reference
        - material_idx: Index of the material type
        
    Outputs:
        - current: Predicted current (standardized)
    """
    def __init__(self, hidden_size=32, embedding_size=8):
        super(MLP_Current, self).__init__()
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.material_embedding = nn.Embedding(3, self.embedding_size)

        input_size = self.embedding_size + 3
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, 1)
        )
        self.apply(init_weights)

    def forward(self, gap, v, initial_I, material_idx):
        material_emb = self.material_embedding(material_idx)
        material_emb = material_emb.expand(len(gap), -1)

        x = torch.cat([gap.unsqueeze(-1), v.unsqueeze(-1), initial_I.unsqueeze(-1), material_emb], dim=-1)
        current = self.net(x)
        return current.squeeze(-1)


class RRAMDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        fit_scaler: bool = False,
        is_train: bool = True,
        seed: Optional[int] = None,
        logger = None,
        split_ratio: float = 0.8,
        use_full_dataset: bool = True
    ):
        self.logger = logger if logger is not None else logging.getLogger(__name__)
        self.scalers = None

        self.logger.info(f"{'Training' if is_train else 'Validation'} Dataset: Loading data...")
        data = loadmat(data_path)

        num_sequences = len([k for k in data.keys() if k.startswith('time_')])
        self.logger.info(f"Found {num_sequences} sequences")

        self.sequences = []
        self.material_to_idx = {
            'HfO2': 0,
            'Al2O3': 1, 
            'TiO2': 2
        }
        
        for i in range(num_sequences):
            time_orig = data[f'time_{i}'].flatten()
            initial_time = np.array([7e-12])
            time_with_initial = np.concatenate([initial_time, time_orig])
            dt = time_with_initial[1:] - time_with_initial[:-1]

            voltage_data = data[f'v_{i}'].flatten()
            formatted_voltage = np.array([format_voltage(v) for v in voltage_data], dtype=np.float32)

            sequence = {
                'time': torch.FloatTensor(time_orig),
                'dt': torch.FloatTensor(dt),
                'voltage': torch.tensor(formatted_voltage, dtype=torch.float32),
                'current': torch.FloatTensor(data[f'current_{i}'].flatten()),
                'material': data[f'material_{i}'][0],
                'material_idx': torch.tensor(self.material_to_idx[data[f'material_{i}'][0]])
            }
            self.sequences.append(sequence)

        rng = np.random.RandomState(seed) if seed is not None else np.random.RandomState()
        
        if use_full_dataset:
            all_indices = np.arange(len(self.sequences))
            rng.shuffle(all_indices)
            train_size = int(len(self.sequences) * split_ratio)
            if is_train:
                final_indices = all_indices[:train_size]
            else:
                final_indices = all_indices[train_size:]
        else:
            final_indices = np.arange(len(self.sequences))
            rng.shuffle(final_indices)

        self.sequences = [self.sequences[i] for i in final_indices]

        self.logger.info(f"Dataset split information:")
        self.logger.info(f"{'Training' if is_train else 'Validation'} set size: {len(self.sequences)}")
        if num_sequences > 0:
             self.logger.info(f"Percentage of total: {len(self.sequences)/num_sequences*100:.1f}%")

        if fit_scaler:
            self.scalers = self._fit_scalers()
            self._apply_scaling()
            
    def _fit_scalers(self):
        scalers = {}
        all_time = torch.cat([seq['time'] for seq in self.sequences])
        all_dt = torch.cat([seq['dt'] for seq in self.sequences])
        all_voltage = torch.cat([seq['voltage'] for seq in self.sequences])
        all_current = torch.cat([seq['current'] for seq in self.sequences])

        scalers['time'] = StandardScaler().fit(all_time.reshape(-1, 1))
        scalers['dt'] = StandardScaler().fit(all_dt.reshape(-1, 1))
        scalers['voltage'] = StandardScaler().fit(all_voltage.reshape(-1, 1))
        scalers['current'] = StandardScaler().fit(all_current.reshape(-1, 1))
        return scalers

    def _apply_scaling(self):
        if self.scalers is None:
            raise ValueError("Scalers not fitted yet!")

        for seq in self.sequences:
            seq['time'] = torch.FloatTensor(
                self.scalers['time'].transform(seq['time'].reshape(-1, 1))
            ).flatten()
            seq['dt'] = torch.FloatTensor(
                self.scalers['dt'].transform(seq['dt'].reshape(-1, 1))
            ).flatten()
            seq['voltage'] = torch.FloatTensor(
                self.scalers['voltage'].transform(seq['voltage'].reshape(-1, 1))
            ).flatten()
            seq['current'] = torch.FloatTensor(
                self.scalers['current'].transform(seq['current'].reshape(-1, 1))
            ).flatten()

    def get_scalers(self):
        if self.scalers is None:
            raise ValueError("Dataset was not initialized with fit_scaler=True")
        return self.scalers

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]

    def set_scalers(self, scalers):
        self.scalers = scalers
        self._apply_scaling()

    def get_material_distribution(self):
        material_counts = {}
        for sequence in self.sequences:
            material = sequence['material']
            material_counts[material] = material_counts.get(material, 0) + 1
        return material_counts


class RRAMCycleEvaluator:
    """
    Evaluator for simulating RRAM operation cycles using trained PINN models.
    
    This class handles:
    1. Loading the pre-trained PINN and MLP models.
    2. Loading and scaling dataset parameters.
    3. Simulating full Read-Set-Read-Reset cycles to extract resistance characteristics.
    """
    def __init__(self, model_path, data_path, device=None):
        self.device = torch.device('cpu')
        print(f"Using device: {self.device}")

        self.const = Constants()
        self.checkpoint = self._load_checkpoint(model_path)
        self.scalers = self.checkpoint['scalers']
        self.pinn_model, self.mlp_model = self.load_model(self.checkpoint)
        self.dataset = self.load_data(data_path, self.scalers)

        self.materials = ['HfO2', 'Al2O3', 'TiO2']
        self.material_to_idx = {mat: idx for idx, mat in enumerate(self.materials)}

        self.read_voltage = 0.4
        self.set_voltage = 1.8
        self.reset_voltage = -1.8
        self.half_reset_voltage = self.reset_voltage / 2
        self.pulse_duration = 10e-9
        self.wait_duration = 5e-9

    def _load_checkpoint(self, model_path):
        print(f"Loading checkpoint: {model_path}")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model checkpoint not found at {model_path}")
        try:
            return torch.load(model_path, map_location=self.device, weights_only=True)
        except Exception:
            return torch.load(model_path, map_location=self.device, weights_only=False)

    def load_model(self, checkpoint):
        print("Initializing models...")
        hidden_size = 10
        embedding_size = 3

        pinn_model = RRAM_PINN(
            hidden_size=hidden_size,
            embedding_size=embedding_size,
            const=self.const
        ).to(self.device)

        mlp_model = MLP_Current(
            hidden_size=hidden_size,
            embedding_size=embedding_size
        ).to(self.device)

        pinn_model.load_state_dict(checkpoint['pinn_model_state_dict'])
        mlp_model.load_state_dict(checkpoint['mlp_model_state_dict'])

        pinn_model.eval()
        mlp_model.eval()
        return pinn_model, mlp_model

    def load_data(self, data_path, scalers):
        dataset = RRAMDataset(
            data_path=data_path,
            fit_scaler=False,
            use_full_dataset=True,
            split_ratio=1.0,
            seed=122
        )
        dataset.set_scalers(scalers)
        return dataset

    def _get_time_sequence(self, target_duration):
        best_seq = None
        min_diff = float('inf')

        time_scale = self.scalers['time'].scale_[0]
        time_mean = self.scalers['time'].mean_[0]

        for seq in self.dataset.sequences:
            time_real_temp = seq['time'].numpy() * time_scale + time_mean
            duration = time_real_temp[-1] - time_real_temp[0]
            if duration <= 0:
                continue
            diff = abs(duration - target_duration)
            if diff < min_diff:
                min_diff = diff
                best_seq = seq

        time_real = best_seq['time'].numpy() * time_scale + time_mean
        dt_real = best_seq['dt'].numpy() * self.scalers['dt'].scale_[0] + self.scalers['dt'].mean_[0]
        original_duration = time_real[-1] - time_real[0]

        if original_duration > 0:
            scale_factor = target_duration / original_duration
        else:
            scale_factor = 1.0

        scaled_time = (time_real - time_real[0]) * scale_factor
        scaled_dt = dt_real * scale_factor
        return torch.FloatTensor(scaled_time).to(self.device), torch.FloatTensor(scaled_dt).to(self.device)

    def _predict_step(self, initial_gap_real, voltage_level, time_real, dt_real, material_idx):
        time_mean = self.scalers['time'].mean_[0]
        time_scale = self.scalers['time'].scale_[0]
        time_norm = (time_real - time_mean) / time_scale

        voltage_mean = self.scalers['voltage'].mean_[0]
        voltage_scale = self.scalers['voltage'].scale_[0]
        voltage_norm_val = (voltage_level - voltage_mean) / voltage_scale
        voltage_norm = torch.full_like(time_norm, voltage_norm_val)

        with torch.no_grad():
            gap_standardized_seq = self.pinn_model(time_norm, dt_real, voltage_norm, material_idx)

        def denormalize_gap(standardized_gap):
            scale = 2 / (self.const.gap_max - self.const.gap_min)
            return (standardized_gap - (-1)) / scale + self.const.gap_min

        gap_physical_seq_model = denormalize_gap(gap_standardized_seq.cpu().numpy())
        delta_gap_model = gap_physical_seq_model[-1] - gap_physical_seq_model[0]
        final_gap_real = np.clip(
            initial_gap_real + delta_gap_model,
            self.const.gap_min,
            self.const.gap_max
        )

        initial_gap_standardized = get_standardized_gap(initial_gap_real, -1, 1, self.const.gap_min, self.const.gap_max)
        shifted_gap_standardized_seq = initial_gap_standardized + (gap_standardized_seq - gap_standardized_seq[0])

        initial_I_norm = (0 - self.scalers['current'].mean_[0]) / self.scalers['current'].scale_[0]
        initial_I_tensor = torch.full_like(voltage_norm, initial_I_norm)

        with torch.no_grad():
            pred_current_norm = self.mlp_model(shifted_gap_standardized_seq, voltage_norm, initial_I_tensor, material_idx)

        current_scale = self.scalers['current'].scale_[0]
        current_mean = self.scalers['current'].mean_[0]
        pred_current_real = pred_current_norm * current_scale + current_mean
        return pred_current_real.squeeze(), final_gap_real

    def _run_operation_cycle(self, material, operations, initial_gap):
        """
        Simulates a sequence of voltage operations (e.g., READ, SET, RESET) for a specific material.
        
        Args:
            material (str): Material name (e.g., 'HfO2').
            operations (list): List of tuples (op_name, voltage, duration).
            initial_gap (float): Initial gap size in meters.
            
        Returns:
            dict: Simulation results including time, voltage, current traces, and resistance measurements.
        """
        material_idx = torch.tensor(self.material_to_idx[material], device=self.device)

        full_time = []
        full_voltage = []
        full_current = []
        phase_markers = []
        resistance_measurements = {}

        current_gap = initial_gap
        last_time = 0.0

        for op_name, voltage, duration in operations:
            if duration == 0:
                continue

            time_seq, dt_seq = self._get_time_sequence(duration)

            if op_name == 'WAIT':
                pred_current = torch.zeros_like(time_seq)
                final_gap = current_gap
            else:
                pred_current, final_gap = self._predict_step(
                    current_gap, voltage, time_seq, dt_seq, material_idx
                )

            if op_name in ['READ', 'SET', 'RESET', 'HALF_RESET']:
                current_pulse = pred_current.cpu().numpy()
                middle_index = len(current_pulse) // 2
                middle_current = current_pulse[middle_index]
                if abs(middle_current) > 1e-12:
                    resistance = abs(voltage / middle_current)
                    resistance_measurements[op_name] = {
                        'resistance': resistance,
                        'voltage': voltage,
                        'current': middle_current,
                        'gap': current_gap
                    }

            full_time.append(time_seq.cpu().numpy() + last_time)
            full_voltage.append(np.full_like(time_seq.cpu().numpy(), voltage))
            full_current.append(pred_current.cpu().numpy())

            phase_markers.append({
                'name': op_name,
                'start': last_time,
                'end': last_time + time_seq[-1].cpu().item()
            })

            last_time += time_seq[-1].cpu().item()
            current_gap = final_gap

        return {
            'time': np.concatenate(full_time),
            'voltage': np.concatenate(full_voltage),
            'current': np.concatenate(full_current),
            'phases': phase_markers,
            'material': material,
            'resistance_measurements': resistance_measurements,
            'final_gap': current_gap
        }
        

    def run_full_evaluation(self, materials=None):
        """
        Runs a standard characterization protocol (Read, Set, Reset, etc.) for specified materials.
        
        Extracts key resistance metrics:
        - Ron/Roff at Set/Reset voltages
        - Ron/Roff at Read voltage (0.4V)
        - Ron at Half-Reset voltage (for nonlinearity checks)
        
        Args:
            materials (list, optional): List of material names to evaluate. Defaults to all available.
            
        Returns:
            dict: Dictionary of resistance metrics for each material.
        """
        if materials is None:
            materials = self.materials

        results = {}

        for material in materials:

            ops_off_read = [('READ', self.read_voltage, self.pulse_duration)]
            result_off_read = self._run_operation_cycle(material, ops_off_read, initial_gap=self.const.gap_max)
            resistance_off_at_read = result_off_read['resistance_measurements']['READ']['resistance']

            ops_off_set = [('SET', self.set_voltage, self.pulse_duration)]
            result_off_set = self._run_operation_cycle(material, ops_off_set, initial_gap=self.const.gap_max)
            resistance_off_at_set = result_off_set['resistance_measurements']['SET']['resistance']

            ops_off_reset = [('RESET', self.reset_voltage, self.pulse_duration)]
            result_off_reset = self._run_operation_cycle(material, ops_off_reset, initial_gap=self.const.gap_max)
            resistance_off_at_reset = result_off_reset['resistance_measurements']['RESET']['resistance']

            ops_set_transition = [('SET', self.set_voltage, self.pulse_duration)]
            result_set = self._run_operation_cycle(material, ops_set_transition, initial_gap=self.const.gap_max)
            gap_after_set = result_set['final_gap']

            ops_on_read = [('READ', self.read_voltage, self.pulse_duration)]
            result_on_read = self._run_operation_cycle(material, ops_on_read, initial_gap=gap_after_set)
            resistance_on_at_read = result_on_read['resistance_measurements']['READ']['resistance']

            ops_on_set = [('SET', self.set_voltage, self.pulse_duration)]
            result_on_set = self._run_operation_cycle(material, ops_on_set, initial_gap=gap_after_set)
            resistance_on_at_set = result_on_set['resistance_measurements']['SET']['resistance']

            ops_on_reset = [('RESET', self.reset_voltage, self.pulse_duration)]
            result_on_reset = self._run_operation_cycle(material, ops_on_reset, initial_gap=gap_after_set)
            resistance_on_at_reset = result_on_reset['resistance_measurements']['RESET']['resistance']

            ops_on_half_reset = [('HALF_RESET', self.half_reset_voltage, self.pulse_duration)]
            result_on_half_reset = self._run_operation_cycle(material, ops_on_half_reset, initial_gap=gap_after_set)
            resistance_on_at_half_reset = result_on_half_reset['resistance_measurements']['HALF_RESET']['resistance']

            def assign_on_off(val_on, val_off):
                return min(val_on, val_off), max(val_on, val_off)

            resistance_on_at_set, resistance_off_at_set = assign_on_off(resistance_on_at_set, resistance_off_at_set)
            resistance_on_at_reset, resistance_off_at_reset = assign_on_off(resistance_on_at_reset, resistance_off_at_reset)
            resistance_on_at_read, resistance_off_at_read = assign_on_off(resistance_on_at_read, resistance_off_at_read)

            results[material] = {
                'ResistanceOnAtSetVoltage': resistance_on_at_set,
                'ResistanceOffAtSetVoltage': resistance_off_at_set,
                'ResistanceOnAtResetVoltage': resistance_on_at_reset,
                'ResistanceOffAtResetVoltage': resistance_off_at_reset,
                'ResistanceOnAtReadVoltage': resistance_on_at_read,
                'ResistanceOffAtReadVoltage': resistance_off_at_read,
                'ResistanceOnAtHalfResetVoltage': resistance_on_at_half_reset,
            }

        return results


def parse_args():
    parser = argparse.ArgumentParser(description='RRAM Continuous Cycle Evaluator')
    parser.add_argument('--model_path', type=str, default='results/checkpoint.pth', help='Path to the model checkpoint.')
    parser.add_argument('--data_path', type=str, default='data/rram.mat', help='Path to the dataset.')
    parser.add_argument('--materials', nargs='+', default=['HfO2', 'Al2O3', 'TiO2'], help='List of materials to evaluate.')
    return parser.parse_args()

def main():
    args = parse_args()

    evaluator = RRAMCycleEvaluator(
        model_path=args.model_path,
        data_path=args.data_path
    )

    results = evaluator.run_full_evaluation(materials=args.materials)
    return results

if __name__ == '__main__':
    main() 