import os
from profilers.BaseMetadata import BaseMetadata

class DrioMetadata(BaseMetadata):
    def __init__(self):
        """
        Initialize DrioMetadata with additional DynamoRIO version info.

        This constructor extends `BaseMetadata.__init__` by appending a
        fixed version string representing the DynamoRIO tool version.
        """
        self.dynamorio_version = "11.3.0"
        # Pass the current profiler directory to BaseMetadata
        current_profiler_dir = os.path.dirname(os.path.abspath(__file__))
        super().__init__(profiler_dir=current_profiler_dir)

    def as_dict(self):
        """
        Convert metadata to a dictionary including base and custom fields.

        Returns
        -------
        dict
            Dictionary of all collected metadata, with an added key
            'dynamorio_version' indicating the profiling tool version.
        """
        base = super().as_dict()
        base["dynamorio_version"] = self.dynamorio_version
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

