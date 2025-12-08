import subprocess
import os
from profilers.BaseMetadata import BaseMetadata

class NsightMetadata(BaseMetadata):
    def __init__(self):
        """
        Initialize NsightMetadata by collecting base system metadata and
        additional CUDA/Nsight version information.
        """
        self.cuda_version = None
        self.nsight_version = None
        # Pass the current profiler directory to BaseMetadata
        current_profiler_dir = os.path.dirname(os.path.abspath(__file__))
        super().__init__(profiler_dir=current_profiler_dir)
        self._collect_nsight_info()

    def _collect_nsight_info(self):
        """
        Collect CUDA and Nsight Compute version information using system tools.

        - CUDA version is determined via `nvcc --version`.
        - Nsight Compute version is determined via `ncu --version`.

        Sets:
        -------
        self.cuda_version : str
        self.nsight_version : str
        """
        # Get CUDA version (via nvcc)
        try:
            cuda_output = subprocess.check_output(["nvcc", "--version"], text=True)
            for line in cuda_output.split("\n"):
                if "release" in line:
                    self.cuda_version = line.strip()
                    break
        except Exception as e:
            self.cuda_version = f"Error: {e}"

        # Get Nsight Compute version (via ncu)
        try:
            nsight_output = subprocess.check_output(["ncu", "--version"], text=True)
            for line in nsight_output.split("\n"):
                if "NVIDIA (R) Nsight Compute" in line:
                    self.nsight_version = line.strip()
                    break
        except Exception as e:
            self.nsight_version = f"Error: {e}"

    def as_dict(self):
        """
        Return metadata as a dictionary including CUDA and Nsight versions.

        Returns
        -------
        dict
            Complete metadata dictionary with toolchain versions.
        """
        base = super().as_dict()
        base["cuda_version"] = self.cuda_version
        base["nsight_version"] = self.nsight_version
        return base

    def full_metadata(self):
        """
        Return full metadata in dictionary format.

        Returns
        -------
        dict
            Alias for as_dict(), for compatibility.
        """
        return self.as_dict()

    def __repr__(self):
        """
        Return a human-readable string representation of the collected metadata.

        Returns
        -------
        str
            Pretty-formatted string summarizing GPU, CUDA, Nsight, CPU, DRAM, and cache info.
        """
        return (f"NsightMetadata(\n"
                f"  GPU: {self.gpu_name} ({self.memory_size} MB), Driver: {self.driver_version}\n"
                f"  CUDA: {self.cuda_version}, Nsight Compute: {self.nsight_version}\n"
                f"  CPU: {self.cpu_info_data.get('Model name', 'Unknown')}, DRAM: {self.dram_size_MB} MB\n"
                f"  Cache: {self.cache_info_data}\n)")

