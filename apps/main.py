import os
import argparse
import sys
import json
import inspect
from profilers.FrontendInterface import FrontendInterface
from profilers.FrontendDriver import register_profilers, register_PatternConfig, register_MetadataClasses
from profilers.PatternConfig import PatternConfig
from profilers.BaseMetadata import BaseMetadata

def get_unique_filename(base_name, ext=".json"):
    """
    Generate a unique filename by appending a counter if needed.
    """
    counter = 0
    filename = f"{base_name}{ext}"
    while os.path.exists(filename):
        counter += 1
        filename = f"{base_name}_{counter}{ext}"
    return filename

def generate_and_save_pattern_config(profiler_name, raw_metrics, metadata=None, level=None, kernel_based=False, output_suffix=None):
    """
    Generate and save PatternConfig files based on raw metrics.

    Parameters
    ----------
    profiler_name : str
        Name of the profiler used (e.g., 'sniper', 'nvbit', 'ncu').
    raw_metrics : dict or list
        Dictionary or list of dictionaries containing metrics data.
    metadata : BaseMetadata, optional
        Metadata object containing system-level info.
    level : str, optional
        Cache level to filter metrics ('l1', 'l2', 'l3', 'dram').
    kernel_based : bool
        Whether profiling results are per-kernel (e.g., for NCU).
    output_suffix : str, optional
        Custom suffix for the output JSON filename.

    Returns
    -------
    None
    """

    config_class = PatternConfig.get_config(profiler_name)
    meta_dict = metadata.as_dict() if metadata else None

    def call_populating(config_class, *args, **kwargs):
        """Call populating only with accepted keyword arguments."""
        sig = inspect.signature(config_class.populating)
        accepted_args = sig.parameters.keys()
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in accepted_args}
        return config_class.populating(*args, **filtered_kwargs)

    if kernel_based:
        for kernel_data in raw_metrics:
            kernel_name = kernel_data.get("Kernel", "UnknownKernel")
            print(f"Processing kernel: {kernel_name}")
            pattern_config = call_populating(config_class, kernel_data, level=level, metadata=None)
            print(json.dumps(pattern_config.to_dict(), indent=2))
            output_file = get_unique_filename(f"memsyspatternconfig_{kernel_name}")
            with open(output_file, "w") as f:
                json.dump(pattern_config.to_dict(), f, indent=2)
    else:
        pattern_config = call_populating(config_class, raw_metrics, level=level, metadata=None)
        pretty_json = json.dumps(
            [cfg.to_dict() for cfg in pattern_config] if isinstance(pattern_config, list) else pattern_config.to_dict(),
            indent=2
        )
        print(pretty_json)
        output_file = get_unique_filename(f"memsyspatternconfig_{output_suffix or profiler_name}")
        with open(output_file, "w") as f:
            f.write(pretty_json)

def main():
    """
    Main CLI entry point for running and extracting metrics from profilers.

    Steps:
    -------
    1. Register available profilers, metadata, and config loaders.
    2. Parse CLI arguments.
    3. Instantiate the selected profiler with required arguments.
    4. Perform profiling if selected.
    5. Extract metrics from the profiling output.
    6. Generate metadata (optional).
    7. Generate pattern configuration JSON output.
    """
    register_profilers()
    register_PatternConfig()
    register_MetadataClasses()

    parser = argparse.ArgumentParser(description="Profiling Tool Driver")
    parser.add_argument("-p", "--profiler", required=True, help="Select the profiler to use.")
    parser.add_argument("-a", "--action", required=True, choices=["profiling", "extract_metrics", "both"])

    # Parse known to get profiler/action
    initial_args, _ = parser.parse_known_args()
    profiler_name = initial_args.profiler
    action = initial_args.action

    if profiler_name not in FrontendInterface.registered_profilers:
        print(f"Error: Profiler '{profiler_name}' is not registered.")
        return

    profiler_class = FrontendInterface.registered_profilers[profiler_name]

    # Collect required arguments
    required_args = []
    if action in ["profiling", "both"]:
        required_args += profiler_class.required_profiling_args()
    if action in ["extract_metrics", "both"]:
        required_args += profiler_class.required_extract_args(action)

    # Add dynamic arguments based on required_args
    for arg in required_args:
        if arg == "executable":
            parser.add_argument(f"--{arg}", required=True, nargs=argparse.REMAINDER)
        elif arg == "level":
            parser.add_argument(f"--{arg}", required=False, choices=["l1", "l2", "l3", "dram"])
        elif arg == "config":
            parser.add_argument(f"--{arg}", required=False, help="Path to profiler configuration file")
        else:
            parser.add_argument(f"--{arg}", required=True)

    # Add optional memory tracking flag (for DynamoRIO)
    parser.add_argument("--enable-memory-stats", dest="enable_memory_stats", action="store_true",
                       help="Enable memory usage tracking (DynamoRIO -stats -mem flags)")

    # Parse full arguments now that dynamic ones are registered
    args = parser.parse_args()
    executable_path = args.executable[0] if isinstance(args.executable, list) else args.executable
    safe_kernel_name = os.path.splitext(os.path.basename(executable_path))[0]

    profiler = profiler_class(**vars(args))
    raw_metrics = None

    # Step 1: Profiling
    if action in ["profiling", "both"]:
        profiling_kwargs = {arg: getattr(args, arg) for arg in profiler_class.required_profiling_args() if getattr(args, arg, None)}
        profiler.profiling(**profiling_kwargs)
        print("Profiling completed successfully.")

    # Step 2: Extract Metrics
    if action in ["extract_metrics", "both"]:
        extract_kwargs = {arg: getattr(args, arg) for arg in profiler_class.required_extract_args(action) if getattr(args, arg, None)}
        raw_metrics = profiler.extract_metrics(**extract_kwargs)
        print("Extracted Metrics:", raw_metrics)

    # Step 3: Metadata (optional)
    metadata_class = FrontendInterface.get_metadata_class(profiler_name)
    metadata = metadata_class() if metadata_class else None

    if metadata:
        print(f"\n--- {profiler_name.upper()} Metadata ---")
        print(metadata)
        meta_file = get_unique_filename(f"memsysmetadata_{profiler_name}")
        with open(meta_file, "w") as f:
            json.dump(metadata.as_dict(), f, indent=2)

    # Step 4: PatternConfig generation
    if raw_metrics:
        kernel_based = profiler_name in ["ncu"]
        output_suffix = safe_kernel_name if profiler_name not in ["ncu"] else None
        level = getattr(args, "level", None)

        generate_and_save_pattern_config(
            profiler_name=profiler_name,
            raw_metrics=raw_metrics,
            metadata=metadata,
            level=level,
            kernel_based=kernel_based,
            output_suffix=output_suffix
        )


if __name__ == "__main__":
    main()

