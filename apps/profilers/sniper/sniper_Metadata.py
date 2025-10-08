import os
import re
import configparser
from profilers.BaseMetadata import BaseMetadata

class SniperMetadata(BaseMetadata):
    def __init__(self, config_path="."):
        """
        Initialize SniperMetadata with optional path to Sniper's config directory.

        Parameters
        ----------
        config_path : str
            Path to the directory containing `sim.cfg`. Defaults to current directory.
        """
        self.sniper_version = "8.2"
        self.sniper_config_path = os.path.join(config_path, "sim.cfg")
        self.simulated_cpu_types = set()
        self.sniper_config_contents = {}
        # Pass the current profiler directory to BaseMetadata
        current_profiler_dir = os.path.dirname(os.path.abspath(__file__))
        super().__init__(profiler_dir=current_profiler_dir)
        self._parse_config_file()

    def _parse_config_file(self):
        """
        Load and parse the Sniper sim.cfg configuration file, if it exists.

        Populates:
        - `self.sniper_config_contents` as a nested dictionary of [section][key] = value.
        - `self.simulated_cpu_types` as a set of detected core types (e.g., ooo, in_order).
        """
        if not os.path.isfile(self.sniper_config_path):
            return

        # Load full config into structured dictionary
        config = configparser.ConfigParser()
        config.optionxform = str  # preserve key case
        try:
            config.read(self.sniper_config_path)

            for section in config.sections():
                self.sniper_config_contents[section] = {}
                for key, value in config.items(section):
                    self.sniper_config_contents[section][key] = value

                    # Heuristic: Detect core_type definitions
                    if "core_type" in key and re.match(r"(ooo|in_order)", value):
                        self.simulated_cpu_types.add(value)

        except Exception as e:
            self.sniper_config_contents = {"error": str(e)}

    def as_dict(self):
        """
        Convert all metadata (base + Sniper-specific) to dictionary format.

        Returns
        -------
        dict
            Full metadata including base system and Sniper configuration values.
        """
        base = super().as_dict()
        base.update({
            "sniper_version": self.sniper_version,
            "sniper_config_path": self.sniper_config_path,
            "simulated_cpu_types": list(self.simulated_cpu_types),
            "sniper_config_contents": self.sniper_config_contents,
        })
        return base

    def full_metadata(self):
        """
        Alias for as_dict() to maintain interface compatibility.

        Returns
        -------
        dict
            Full metadata.
        """
        return self.as_dict()

    def __repr__(self):
        """
        Return a human-readable string representation of the Sniper metadata.

        Returns
        -------
        str
            Summary string including config file, CPU types, and base metadata.
        """
        return (f"SniperMetadata(\n"
                f"  Version: {self.sniper_version}\n"
                f"  Config File: {self.sniper_config_path}\n"
                f"  Simulated CPU Types: {', '.join(self.simulated_cpu_types)}\n"
                f"  GPU: {self.gpu_name} ({self.memory_size} MB), Driver: {self.driver_version}\n"
                f"  CPU: {self.cpu_info_data.get('Model name', 'Unknown')}, DRAM: {self.dram_size_MB} MB\n"
                f"  Cache: {self.cache_info_data}\n"
                f"  Config Sections: {list(self.sniper_config_contents.keys())}\n)")

