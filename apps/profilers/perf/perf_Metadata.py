import subprocess
import os
from profilers.BaseMetadata import BaseMetadata

class PerfMetadata(BaseMetadata):
    def __init__(self):
        """
        Initialize the PerfMetadata object.

        Collects:
        - perf version (via `perf --version`)
        - standard hardware/software metadata from BaseMetadata
        """
        self.perf_version = None
        # Pass the current profiler directory to BaseMetadata
        current_profiler_dir = os.path.dirname(os.path.abspath(__file__))
        super().__init__(profiler_dir=current_profiler_dir)
        self._collect_perf_version()

    def _collect_perf_version(self):
        """
        Collect the version string of the installed `perf` tool.

        Uses subprocess to call `perf --version`. On failure, stores error message.
        Example output: "perf version 6.5.0"
        """
        try:
            output = subprocess.check_output(["perf", "--version"], text=True)
            self.perf_version = output.strip()  # e.g., "perf version 6.5.0"
        except Exception as e:
            self.perf_version = f"Error: {e}"

    def as_dict(self):
        """
        Convert collected metadata to a dictionary.

        Returns
        -------
        dict
            Dictionary containing system metadata and perf version.
        """
        base = super().as_dict()
        base["perf_version"] = self.perf_version
        return base

    def full_metadata(self):
        """
        Return complete metadata in dictionary format.

        Returns
        -------
        dict
            Alias for `as_dict()` for compatibility.
        """
        return self.as_dict()

    def __repr__(self):
        """
        Human-readable string representation of the metadata.

        Returns
        -------
        str
            Summary including perf version, GPU/CPU info, and DRAM/cache sizes.
        """
        return (f"PerfMetadata(\n"
                f"  Version: {self.perf_version}\n"
                f"  GPU: {self.gpu_name} ({self.memory_size} MB), Driver: {self.driver_version}\n"
                f"  CPU: {self.cpu_info_data.get('Model name', 'Unknown')}, DRAM: {self.dram_size_MB} MB\n"
                f"  Cache: {self.cache_info_data}\n)")

