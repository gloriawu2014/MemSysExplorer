import os
import glob
from profilers.BaseMetadata import BaseMetadata

class DrioMetadata(BaseMetadata):
    def __init__(self, output_dir=None):
        """
        Initialize DrioMetadata with additional DynamoRIO version info.

        This constructor extends `BaseMetadata.__init__` by appending a
        fixed version string representing the DynamoRIO tool version.

        Args:
            output_dir (str): Directory to search for protobuf time-series files
        """
        self.dynamorio_version = "11.3.0"
        self.output_dir = output_dir or os.getcwd()
        self.timeseries_file = None
        self.trace_file = None

        # Pass the current profiler directory to BaseMetadata
        current_profiler_dir = os.path.dirname(os.path.abspath(__file__))
        super().__init__(profiler_dir=current_profiler_dir)

        # Find protobuf output files
        self._find_protobuf_files()

    def _find_protobuf_files(self):
        """
        Find protobuf time-series and trace files in output directory.

        Looks for:
        - timeseries_*.pb - Time-series metrics (WSS sampling)
        - memtrace_*.pb - Detailed memory traces
        """
        try:
            # Find time-series file (should be only one per run)
            timeseries_files = glob.glob(os.path.join(self.output_dir, "timeseries_*.pb"))
            if timeseries_files:
                self.timeseries_file = os.path.basename(timeseries_files[0])

            # Find trace file (should be only one per run)
            trace_files = glob.glob(os.path.join(self.output_dir, "memtrace_*.pb"))
            if trace_files:
                self.trace_file = os.path.basename(trace_files[0])
        except Exception as e:
            # Silently fail if files not found
            pass

    def as_dict(self):
        """
        Convert metadata to a dictionary including base and custom fields.

        Returns
        -------
        dict
            Dictionary of all collected metadata, with an added key
            'dynamorio_version' indicating the profiling tool version.
            Also includes references to protobuf output files if found.
        """
        base = super().as_dict()
        base["dynamorio_version"] = self.dynamorio_version

        # Add protobuf file references if found
        if self.timeseries_file or self.trace_file:
            base["protobuf_outputs"] = {}
            if self.timeseries_file:
                base["protobuf_outputs"]["timeseries_file"] = self.timeseries_file
            if self.trace_file:
                base["protobuf_outputs"]["trace_file"] = self.trace_file

        return base

    def full_metadata(self):
        """
        Return a complete dictionary representation of metadata.

        This method provides an alias for `as_dict()` for compatibility
        with any extended interface.

        Returns
        -------
        dict
            Full metadata dictionary.
        """
        return self.as_dict()

    def __repr__(self):
        """
        Pretty-print the metadata summary for interactive inspection.

        Returns
        -------
        str
            Human-readable representation of key metadata fields.
        """
        return (f"DrioMetadata(\n"
                f"  Version: {self.dynamorio_version}\n"
                f"  CPU: {self.cpu_info_data.get('Model name', 'Unknown')}, DRAM: {self.dram_size_MB} MB\n"
                f"  DRAM: {self.dram_size_MB}"
                f"  Cache: {self.cache_info_data}\n)")

