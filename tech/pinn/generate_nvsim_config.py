#!/usr/bin/env python3
"""
NVSim configuration generator for RRAM cells using PINN predictions.
"""

import os
import re
import argparse
import subprocess
from pathlib import Path


class NVSimConfigGenerator:
    """
    Automates the generation of NVSim configuration files using PINN-derived device characteristics.
    
    This class manages the workflow of:
    1. Parsing a sample cell file to get target voltage parameters.
    2. Running the RRAM PINN evaluator to get resistance values.
    3. Normalizing resistance values to match target specifications.
    4. Generating new .cell and .cfg files for NVSim.
    5. Optionally running NVSim simulations.
    """
    def __init__(self,
                 model_path='tech/pinn/checkpoints/checkpoint.pth',
                 data_path='tech/pinn/data/rram.mat',
                 tech_dir='tech',
                 materials=None):
        # Auto-detect project root
        self.script_dir = Path(__file__).parent  # tech/pinn
        self.project_root = self.script_dir.parent.parent  # project root (two levels up)

        # Resolve paths relative to project root
        self.model_path = self._resolve_path(model_path)
        self.data_path = self._resolve_path(data_path)
        self.tech_dir = self._resolve_path(tech_dir)
        self.materials = materials or ['HfO2', 'Al2O3', 'TiO2']

        self.sample_cells_dir = self.tech_dir / 'ArrayCharacterization' / 'sample_cells'
        self.sample_configs_dir = self.tech_dir / 'ArrayCharacterization' / 'sample_configs'

    def _resolve_path(self, path_str):
        path = Path(path_str)

        if path.is_absolute() and path.exists():
            return path

        if path.exists():
            return path.resolve()

        project_path = self.project_root / path
        if project_path.exists():
            return project_path

        script_path = self.script_dir / path
        if script_path.exists():
            return script_path

        return project_path

    def parse_cell_file(self, cell_file_path):
        """
        Parses an NVSim cell file to extract voltage and pulse parameters.
        
        Args:
            cell_file_path (Path): Path to the .cell file.
            
        Returns:
            dict: Dictionary of extracted parameters (e.g., ReadVoltage, SetPulse).
        """
        params = {}

        with open(cell_file_path, 'r') as f:
            content = f.read()

        voltage_pulse_params = {
            'ReadVoltage': r'-ReadVoltage \(V\):\s*([\d.+-]+)',
            'ResetVoltage': r'-ResetVoltage \(V\):\s*([\d.+-]+)',
            'SetVoltage': r'-SetVoltage \(V\):\s*([\d.+-]+)',
            'ResetPulse': r'-ResetPulse \(ns\):\s*([\d.+-]+)',
            'SetPulse': r'-SetPulse \(ns\):\s*([\d.+-]+)',
            'ResistanceOnAtHalfResetVoltage': r'-ResistanceOnAtHalfResetVoltage \(ohm\):\s*([\d.+-]+)'
        }

        for param_name, pattern in voltage_pulse_params.items():
            match = re.search(pattern, content)
            if match:
                params[param_name] = float(match.group(1))

        return params

    def run_evaluator(self, voltage_params, silent=False):
        from rram_cycle_evaluator import RRAMCycleEvaluator
        import sys
        from io import StringIO

        evaluator = RRAMCycleEvaluator(
            model_path=self.model_path,
            data_path=self.data_path
        )

        evaluator.read_voltage = voltage_params.get('ReadVoltage', 0.4)
        evaluator.set_voltage = voltage_params.get('SetVoltage', 1.8)
        evaluator.reset_voltage = voltage_params.get('ResetVoltage', -1.8)
        evaluator.half_reset_voltage = evaluator.reset_voltage / 2
        evaluator.pulse_duration = voltage_params.get('SetPulse', 10) * 1e-9
        evaluator.wait_duration = evaluator.pulse_duration / 2

        if not silent:
            print(f"\n{'='*60}")
            print(f"Running device characterization with parameters:")
            print(f"  Read Voltage: {evaluator.read_voltage} V")
            print(f"  Set Voltage: {evaluator.set_voltage} V")
            print(f"  Reset Voltage: {evaluator.reset_voltage} V")
            print(f"  Pulse Duration: {evaluator.pulse_duration*1e9} ns")
            print(f"{'='*60}")

        old_stdout = sys.stdout
        sys.stdout = StringIO()
        try:
            results = evaluator.run_full_evaluation(materials=self.materials)
        finally:
            sys.stdout = old_stdout

        return results

    def generate_cell_file(self, resistance_values, voltage_params, output_path):
        read_voltage = voltage_params.get('ReadVoltage', 0.4)
        reset_voltage = voltage_params.get('ResetVoltage', 2.0)
        set_voltage = voltage_params.get('SetVoltage', 2.0)
        reset_pulse = voltage_params.get('ResetPulse', 10)
        set_pulse = voltage_params.get('SetPulse', 10)

        r_on_set = resistance_values['ResistanceOnAtSetVoltage']
        r_off_set = resistance_values['ResistanceOffAtSetVoltage']
        r_on_reset = resistance_values['ResistanceOnAtResetVoltage']
        r_off_reset = resistance_values['ResistanceOffAtResetVoltage']
        r_on_read = resistance_values['ResistanceOnAtReadVoltage']
        r_off_read = resistance_values['ResistanceOffAtReadVoltage']
        r_on_half_reset = resistance_values['ResistanceOnAtHalfResetVoltage']

        read_power = (read_voltage**2 / r_on_read) * 1e6

        r_avg_set = (r_off_set + r_on_set) / 2
        set_energy = (set_voltage**2 / r_avg_set) * set_pulse * 1e-9 * 1e12

        r_avg_reset = (r_on_reset + r_off_reset) / 2
        reset_energy = (reset_voltage**2 / r_avg_reset) * reset_pulse * 1e-9 * 1e12

        cell_content = f"""-MemCellType: memristor

-CellArea (F^2): 4
-CellAspectRatio: 1

-ResistanceOnAtSetVoltage (ohm): {r_on_set:.0f}
-ResistanceOffAtSetVoltage (ohm): {r_off_set:.0f}
-ResistanceOnAtResetVoltage (ohm): {r_on_reset:.0f}
-ResistanceOffAtResetVoltage (ohm): {r_off_reset:.0f}
-ResistanceOnAtReadVoltage (ohm): {r_on_read:.0f}
-ResistanceOffAtReadVoltage (ohm): {r_off_read:.0f}
-ResistanceOnAtHalfResetVoltage (ohm): {r_on_half_reset:.0f}

-CapacitanceOn (F): 1e-16
-CapacitanceOff (F): 1e-16

-ReadMode: current
-ReadVoltage (V): {read_voltage}
-ReadPower (uW): {read_power:.2f}

-ResetMode: voltage
-ResetVoltage (V): {abs(reset_voltage)}
-ResetPulse (ns): {reset_pulse}
-ResetEnergy (pJ): {reset_energy:.2f}

-SetMode: voltage
-SetVoltage (V): {set_voltage}
-SetPulse (ns): {set_pulse}
-SetEnergy (pJ): {set_energy:.2f}

-AccessType: None
//-AccessType: diode
//-VoltageDropAccessDevice (V): 0.5

-ReadFloating: false

"""

        with open(output_path, 'w') as f:
            f.write(cell_content)

        print(f"\nGenerated cell file: {output_path}")

    def generate_config_file(self, cell_filename, output_path, template_config=None,
                            adjust_for_low_resistance=True):
        if template_config and os.path.exists(template_config):
            with open(template_config, 'r') as f:
                config_content = f.read()

            config_content = re.sub(
                r'-MemoryCellInputFile:.*',
                f'-MemoryCellInputFile: sample_cells/{cell_filename}',
                config_content
            )

            if adjust_for_low_resistance:
                config_content = re.sub(
                    r'-Capacity \(KB\):.*',
                    '-Capacity (KB): 64',
                    config_content
                )
                config_content = re.sub(
                    r'-Associativity \(for cache only\):.*',
                    '-Associativity (for cache only): 4',
                    config_content
                )
        else:
            config_content = f"""-MemoryCellInputFile: sample_cells/{cell_filename}

-ProcessNode: 32
-DeviceRoadmap: LOP

-DesignTarget: cache

-CacheAccessMode: Normal
-Associativity (for cache only): 8

-OptimizationTarget: WriteEDP

-OutputFilePrefix: test
-EnablePruning: Yes

-Capacity (KB): 128
-WordWidth (bit): 128

-LocalWireType: LocalAggressive
-LocalWireRepeaterType: RepeatedNone

-LocalWireUseLowSwing: No

-GlobalWireType: GlobalAggressive
-GlobalWireRepeaterType: RepeatedNone
-GlobalWireUseLowSwing: No

-Routing: H-tree

-InternalSensing: true

-Temperature (K): 370
-RetentionTime (us): 40

-BufferDesignOptimization: latency

"""

        with open(output_path, 'w') as f:
            f.write(config_content)

        print(f"Generated config file: {output_path}")

    def run_nvsim(self, config_file):
        original_dir = os.getcwd()

        try:
            # Change to ArrayCharacterization directory where nvsim binary is located
            array_char_dir = self.tech_dir / 'ArrayCharacterization'
            os.chdir(array_char_dir)

            cmd = ['./nvsim', f'sample_configs/{config_file}']
            print(f"\n{'='*60}")
            print(f"Running NVSim: {' '.join(cmd)}")
            print(f"Working directory: {array_char_dir}")
            print(f"{'='*60}\n")

            result = subprocess.run(cmd, capture_output=True, text=True)

            print(result.stdout)
            if result.stderr:
                print("STDERR:", result.stderr)

            return result.returncode == 0

        except Exception as e:
            print(f"Error running NVSim: {e}")
            return False
        finally:
            os.chdir(original_dir)

    def _normalize_resistance_values(self, r_half_reset, target_r_half_reset):
        if target_r_half_reset is None:
            target_r_half_reset = 5e5
        target_r_half_reset = target_r_half_reset / 2
        exact_factor = target_r_half_reset / r_half_reset
        integer_factor = round(exact_factor)
        integer_factor = max(1, integer_factor)

        return integer_factor

    def run_workflow(self, sample_cell_file, run_nvsim=False, template_config=None):
        """
        Executes the full generation workflow.
        
        Args:
            sample_cell_file (str): Filename of the sample cell in 'tech/ArrayCharacterization/sample_cells'.
            run_nvsim (bool): Whether to run NVSim simulation after generation.
            template_config (str, optional): Path to a template .cfg file.
        """
        print(f"\n{'#'*60}")
        print(f"# PINN-Based RRAM Cell Generation Workflow")
        print(f"{'#'*60}\n")

        sample_cell_path = self.sample_cells_dir / sample_cell_file

        if not sample_cell_path.exists():
            raise FileNotFoundError(f"Sample cell file not found: {sample_cell_path}")

        print(f"Step 1: Parsing sample cell file: {sample_cell_file}")
        voltage_params = self.parse_cell_file(sample_cell_path)

        print(f"\nExtracted parameters:")
        for key, value in voltage_params.items():
            print(f"  {key}: {value}")

        print(f"\nStep 2: Running device characterization for materials: {', '.join(self.materials)}")
        results = self.run_evaluator(voltage_params)

        for material in self.materials:
            r_on_reset = results[material]['ResistanceOnAtResetVoltage']
            results[material]['ResistanceOnAtHalfResetVoltage'] = r_on_reset / 2

        # Use HfO2 as reference if available, otherwise use first material
        reference_material = 'HfO2' if 'HfO2' in results else self.materials[0]

        ref_r_half = results[reference_material]['ResistanceOnAtHalfResetVoltage']
        target_r_half = voltage_params.get('ResistanceOnAtHalfResetVoltage', None)
        
        norm_factor = self._normalize_resistance_values(ref_r_half, target_r_half)

        for material in self.materials:
            for key in results[material].keys():
                if 'Resistance' in key:
                    results[material][key] *= norm_factor

        for material in self.materials:
            print(f"\n{'='*60}")
            print(f"Evaluating material: {material}")
            print(f"{'='*60}\n")

            r = results[material]
            print(f"-ResistanceOnAtSetVoltage (ohm): {r['ResistanceOnAtSetVoltage']:.0f}")
            print(f"-ResistanceOffAtSetVoltage (ohm): {r['ResistanceOffAtSetVoltage']:.0f}")
            print(f"-ResistanceOnAtResetVoltage (ohm): {r['ResistanceOnAtResetVoltage']:.0f}")
            print(f"-ResistanceOffAtResetVoltage (ohm): {r['ResistanceOffAtResetVoltage']:.0f}")
            print(f"-ResistanceOnAtReadVoltage (ohm): {r['ResistanceOnAtReadVoltage']:.0f}")
            print(f"-ResistanceOffAtReadVoltage (ohm): {r['ResistanceOffAtReadVoltage']:.0f}")
            print(f"-ResistanceOnAtHalfResetVoltage (ohm): {r['ResistanceOnAtHalfResetVoltage']:.0f}")

            ratio = r['ResistanceOffAtReadVoltage'] / r['ResistanceOnAtReadVoltage']
            print(f"\nOn/Off Ratio: {ratio:.2f}x")

        print(f"\nStep 3: Generating cell and config files")
        generated_files = []

        for material in self.materials:
            cell_filename = f"sample_RRAM_PIGEN_{material}.cell"
            config_filename = f"sample_RRAM_PIGEN_{material}.cfg"

            cell_output_path = self.sample_cells_dir / cell_filename
            config_output_path = self.sample_configs_dir / config_filename

            self.generate_cell_file(
                results[material],
                voltage_params,
                cell_output_path
            )

            self.generate_config_file(
                cell_filename,
                config_output_path,
                template_config
            )

            generated_files.append({
                'material': material,
                'cell_file': cell_filename,
                'config_file': config_filename
            })

        if run_nvsim:
            print(f"\nStep 4: Running NVSim simulations")
            for file_info in generated_files:
                print(f"\nSimulating {file_info['material']}...")
                success = self.run_nvsim(file_info['config_file'])
                if success:
                    print(f"✓ NVSim simulation completed for {file_info['material']}")
                else:
                    print(f"✗ NVSim simulation failed for {file_info['material']}")

        print(f"\n{'#'*60}")
        print(f"# Workflow Complete!")
        print(f"{'#'*60}\n")

        print("Generated files:")
        for file_info in generated_files:
            print(f"\n  Material: {file_info['material']}")
            print(f"    Cell file:   sample_cells/{file_info['cell_file']}")
            print(f"    Config file: sample_configs/{file_info['config_file']}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate NVSim configurations from PINN predictions',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate cell files only
  python generate_nvsim_config.py

  # Generate and run NVSim simulation
  python generate_nvsim_config.py --run-nvsim

  # Use custom sample cell file
  python generate_nvsim_config.py --sample-cell my_rram.cell

  # Specify materials to simulate
  python generate_nvsim_config.py --materials HfO2 TiO2
        """
    )

    parser.add_argument('--model-path', type=str, default='tech/pinn/checkpoints/checkpoint.pth',
                        help='Path to PINN model checkpoint (default: tech/pinn/checkpoints/checkpoint.pth)')
    parser.add_argument('--data-path', type=str, default='tech/pinn/data/rram.mat',
                        help='Path to RRAM dataset (default: tech/pinn/data/rram.mat)')
    parser.add_argument('--tech-dir', type=str, default='tech',
                        help='Path to tech directory (default: tech)')
    parser.add_argument('--sample-cell', type=str, default='sample_RRAM.cell',
                        help='Sample cell file to use as template (default: sample_RRAM.cell)')
    parser.add_argument('--template-config', type=str,
                        default='tech/ArrayCharacterization/sample_configs/sample_RRAM_32nm.cfg',
                        help='Template config file (default: tech/ArrayCharacterization/sample_configs/sample_RRAM_32nm.cfg)')
    parser.add_argument('--materials', nargs='+', default=['HfO2', 'Al2O3', 'TiO2'],
                        help='Materials to evaluate')
    parser.add_argument('--run-nvsim', action='store_true',
                        help='Run NVSim simulation after generating files')

    args = parser.parse_args()

    generator = NVSimConfigGenerator(
        model_path=args.model_path,
        data_path=args.data_path,
        tech_dir=args.tech_dir,
        materials=args.materials
    )

    generator.run_workflow(
        sample_cell_file=args.sample_cell,
        run_nvsim=args.run_nvsim,
        template_config=args.template_config
    )


if __name__ == '__main__':
    main()