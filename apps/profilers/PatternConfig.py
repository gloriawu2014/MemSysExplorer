"""
Pattern Configuration Abstraction for Profiling Output.

The `PatternConfig` class encapsulates structured performance metrics derived from
raw profiler data. Each configuration holds frequency, volume, and size of memory
operations along with workload and metadata descriptors.

This class also provides a registry mechanism to dynamically bind profiler-specific
config classes (e.g., SniperConfig, NVBitConfig), which must implement the `populating`
method for raw data transformation.

Typical Use
-----------
1. Subclass `PatternConfig` and implement `populating(data, metadata)`.
2. Register the subclass using `register_config`.
3. Use `create_config` to instantiate and populate the config.

Fields
------
- exp_name : str
- benchmark_name : str
- read_freq / write_freq : float (e.g., bytes/sec)
- total_reads / total_writes : int
- read_size / write_size : int
- total_reads_d / total_reads_i / total_writes_d / total_writes_i : int
- workingset_size : int
- metadata : BaseMetadata or similar
- unit : dict of units for each field (overridable)
"""


class PatternConfig:
    registered_configs = {}

    def __init__(self, exp_name="default", benchmark_name="test",
                 read_freq=-1, total_reads=-1, write_freq=-1, total_writes=-1,
                 total_reads_d=1, total_reads_i=1,
                 total_writes_d=1, total_writes_i=1,
                 read_size=1, write_size=1, workingset_size=-1,
                 execution_time=None,
                 peak_memory_kb=None,
                 metadata=None,
                 unit=None):  # <-- Allow override from child

        """
        Initialize a PatternConfig instance with default or custom profiling data.

        Parameters
        ----------
        exp_name : str
            Name of the profiling experiment.
        benchmark_name : str
            Name of the benchmark or application profiled.
        read_freq : float
            Read bandwidth (e.g., bytes/sec).
        total_reads : int
            Total number of read operations.
        write_freq : float
            Write bandwidth (e.g., bytes/sec).
        total_writes : int
            Total number of write operations.
        total_reads_d : int
            Data read operations.
        total_reads_i : int
            Instruction read operations.
        total_writes_d : int
            Data write operations.
        total_writes_i : int
            Instruction write operations.
        read_size : int
            Size of a read in bytes.
        write_size : int
            Size of a write in bytes.
        workingset_size : int
            Estimated working set size in bytes.
        metadata : object
            Optional metadata object (e.g., BaseMetadata).
        unit : dict
            Dictionary mapping metric names to string unit labels.
        """

        # Default values
        self.exp_name = exp_name
        self.benchmark_name = benchmark_name
        self.read_freq = read_freq
        self.total_reads = total_reads
        self.write_freq = write_freq
        self.total_writes = total_writes
        self.read_size = read_size
        self.write_size = write_size
        self.total_writes_i = total_writes_i
        self.total_writes_d = total_writes_d
        self.total_reads_d = total_reads_d
        self.total_reads_i = total_reads_i
        self.workingset_size = workingset_size
        self.execution_time = execution_time
        self.peak_memory_kb = peak_memory_kb
        self.metadata = metadata

        # Default units
        default_unit = {
            "read_freq": "bytes/s",
            "write_freq": "bytes/s",
            "total_reads": "count",
            "total_writes": "count",
            "read_size": "bytes",
            "write_size": "bytes",
            "workingset_size": "bytes",
            "execution_time": "seconds",
            "peak_memory_kb": "KB"
        }

        # Override if provided
        self.unit = unit if unit else default_unit

    @classmethod
    def register_config(cls, name, config_class):
        """
        Register a config class for a specific profiler name.

        Parameters
        ----------
        name : str
            Profiler name identifier (e.g., 'sniper', 'nvbit').
        config_class : class
            Subclass of PatternConfig that implements `populating()`.
        """
        cls.registered_configs[name] = config_class

    @classmethod
    def get_config(cls, name):
        """
        Retrieve a registered config class by profiler name.

        Parameters
        ----------
        name : str
            Profiler identifier string.

        Returns
        -------
        class
            Registered config class.

        Raises
        ------
        ValueError
            If no config is registered for the given name.
        """

        if name not in cls.registered_configs:
            raise ValueError(f"PatternConfig for profiler '{name}' is not registered.")
        return cls.registered_configs[name]

    @classmethod
    def create_config(cls, name, raw_data, metadata=None):
        """
        Create and populate a PatternConfig using a registered subclass.

        Parameters
        ----------
        name : str
            Profiler name for which to generate a config.
        raw_data : dict
            Raw metrics data (from extract_metrics).
        metadata : object, optional
            Metadata object to include.

        Returns
        -------
        PatternConfig
            An instance populated with extracted and interpreted data.
        """
        config_class = cls.get_config(name)
        return config_class.populating(raw_data, metadata)

    def populating(self, data):
        """
        Abstract method: Populate the config fields from raw data.

        Parameters
        ----------
        data : dict
            Raw data from profiler output.

        Raises
        ------
        NotImplementedError
            Must be overridden by subclasses.
        """

        raise NotImplementedError("Subclasses must implement this method to populate from raw data.")

    def __repr__(self):
        """
        Generate a human-readable summary of the PatternConfig.

        Returns
        -------
        str
            Formatted string describing key metrics.
        """
        metadata_str = f", metadata={self.metadata}" if self.metadata else ""
        return (f"PatternConfig(exp_name={self.exp_name}, "
                f"benchmark_name={self.benchmark_name}, "
                f"read_freq={self.read_freq} {self.unit.get('read_freq', '')}, "
                f"total_reads={self.total_reads} {self.unit.get('total_reads', '')}, "
                f"write_freq={self.write_freq} {self.unit.get('write_freq', '')}, "
                f"total_writes={self.total_writes} {self.unit.get('total_writes', '')}, "
                f"read_size={self.read_size} {self.unit.get('read_size', '')}, "
                f"write_size={self.write_size} {self.unit.get('write_size', '')}, "
                f"workingset_size={self.workingset_size} {self.unit.get('workingset_size', '')}{metadata_str})")

    def to_dict(self):
        """
        Convert the PatternConfig to a dictionary, including nested metadata.

        Returns
        -------
        dict
            Serializable representation of the pattern config and its metadata.
        """
        data = self.__dict__.copy()
        if hasattr(self.metadata, "as_dict"):
            data["metadata"] = self.metadata.as_dict()
        return data

