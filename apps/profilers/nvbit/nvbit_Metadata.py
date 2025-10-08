import subprocess
import os
from profilers.BaseMetadata import BaseMetadata

class NVBitMetadata(BaseMetadata):
    def __init__(self):
        """
        Initialize the NVBitMetadata collector.

        Attributes
        ----------
        nvbit_version : str
            Statically assigned version of NVBit in use (default "1.7.4").
        cuda_version : str or None
            Detected CUDA version from `nvcc --version`, or an error message if failed.
        """
        self.nvbit_version = "1.7.4"
        self.cuda_version = None
        # Pass the current profiler directory to BaseMetadata
        current_profiler_dir = os.path.dirname(os.path.abspath(__file__))
        super().__init__(profiler_dir=current_profiler_dir)
        self._collect_cuda_version()

    def _collect_cuda_version(self):
        """
        Collect the CUDA version using `nvcc --version`.

        This method runs the `nvcc` command and parses the version string,
        storing it in `self.cuda_version`. On failure, an error message is stored instead.
        """
        try:
            output = subprocess.check_output(["nvcc", "--version"], text=True)
            for line in output.splitlines():
                if "release" in line:
                    self.cuda_version = line.strip()
                    break
        except Exception as e:
            self.cuda_version = f"Error: {e}"

    def as_dict(self):
        """
        Convert all metadata to a dictionary format.

        Returns
        -------
        dict
            Dictionary containing base system metadata along with NVBit and CUDA versions.
        """
        base = super().as_dict()
        base["nvbit_version"] = self.nvbit_version
        base["cuda_version"] = self.cuda_version
        return base

    def full_metadata(self):
        """
        Return complete metadata dictionary.

        Returns
        -------
        dict
            Alias for `as_dict()`, included for interface consistency.
        """
        return self.as_dict()

    def __repr__(self):
        """
        String representation of the NVBit metadata.

        Returns
        -------
        str
            Human-readable summary of GPU, CUDA, NVBit, CPU, DRAM, and cache info.
        """
        return (f"NVBitMetadata(\n"
                f"  GPU: {self.gpu_name} ({self.memory_size} MB), Driver: {self.driver_version}\n"
                f"  CUDA: {self.cuda_version}, NVBit: {self.nvbit_version}\n"
                f"  CPU: {self.cpu_info_data.get('Model name', 'Unknown')}, DRAM: {self.dram_size_MB} MB\n"
                f"  Cache: {self.cache_info_data}\n)")

