from profilers.PatternConfig import PatternConfig

class DrioConfig(PatternConfig):
    def __init__(self, **kwargs):
        """
        Initialize DrioConfig with additional size histogram fields.

        Parameters
        ----------
        **kwargs : dict
            Configuration parameters, including size histogram data
        """
        # Extract size histograms before calling parent init
        self.read_size_histogram = kwargs.pop('read_size_histogram', {})
        self.write_size_histogram = kwargs.pop('write_size_histogram', {})

        # Call parent constructor
        super().__init__(**kwargs)

    @classmethod
    def populating(cls, report_data, metadata=None):
        """
        Populate the PatternConfig attributes using DynamoRIO raw data.

        Parameters
        ----------
        report_data : dict
            Dictionary of memory access metrics, typically returned from
            `DrioProfilers.extract_metrics()`. Expected keys include:
            - 'read_freq'
            - 'write_freq'
            - 'total_reads'
            - 'total_writes'
            - 'workingset_size'
            - 'Memory' (used as the benchmark name)
            - 'read_size_X' (size-specific read counts)
            - 'write_size_X' (size-specific write counts)

        metadata : BaseMetadata, optional
            Optional system metadata object (e.g., CPU, cache, DRAM info).

        Returns
        -------
        DrioConfig
            A fully populated configuration object for downstream modeling.
        """

        # Extract execution time (use microseconds for maximum precision)
        execution_time_us = report_data.get("execution_time_us")

        # Calculate read/write frequencies (accesses per microsecond)
        # Keep full floating-point precision
        read_freq = None
        write_freq = None
        if execution_time_us and execution_time_us > 0:
            total_reads = report_data.get("total_reads", 0)
            total_writes = report_data.get("total_writes", 0)
            # Use float division to maintain precision
            read_freq = float(total_reads) / float(execution_time_us)
            write_freq = float(total_writes) / float(execution_time_us)

        # Unit overrides for this config
        # Use microseconds as base unit (matches DynamoRIO's native precision)
        unit_overrides = {
            "read_freq": "accesses/us",
            "write_freq": "accesses/us",
            "total_reads": "count",
            "total_writes": "count",
            "workingset_size": "count",
            "execution_time": "microseconds",
            "peak_memory_kb": "KB"
        }

        # Build read size histogram
        read_size_histogram = {
            "1": report_data.get("read_size_1", 0),
            "2": report_data.get("read_size_2", 0),
            "4": report_data.get("read_size_4", 0),
            "8": report_data.get("read_size_8", 0),
            "16": report_data.get("read_size_16", 0),
            "32": report_data.get("read_size_32", 0),
            "64": report_data.get("read_size_64", 0),
            "other": report_data.get("read_size_other", 0)
        }

        # Build write size histogram
        write_size_histogram = {
            "1": report_data.get("write_size_1", 0),
            "2": report_data.get("write_size_2", 0),
            "4": report_data.get("write_size_4", 0),
            "8": report_data.get("write_size_8", 0),
            "16": report_data.get("write_size_16", 0),
            "32": report_data.get("write_size_32", 0),
            "64": report_data.get("write_size_64", 0),
            "other": report_data.get("write_size_other", 0)
        }

        # Calculate dominant size (most common) for backward compatibility
        dominant_read_size = max(read_size_histogram.items(), key=lambda x: x[1])[0] if sum(read_size_histogram.values()) > 0 else "32"
        dominant_write_size = max(write_size_histogram.items(), key=lambda x: x[1])[0] if sum(write_size_histogram.values()) > 0 else "32"

        return cls(
            exp_name="DynamoRIOProfilers",
            benchmark_name=report_data.get("Memory", " "),
            read_freq=read_freq,
            total_reads=report_data.get("total_reads"),
            write_freq=write_freq,
            total_writes=report_data.get("total_writes"),
            workingset_size=report_data.get("workingset_size"),
            execution_time=execution_time_us,  # Store in microseconds
            peak_memory_kb=report_data.get("peak_memory_kb"),  # Peak memory usage
            read_size=int(dominant_read_size) if dominant_read_size != "other" else 32,
            write_size=int(dominant_write_size) if dominant_write_size != "other" else 32,
            read_size_histogram=read_size_histogram,
            write_size_histogram=write_size_histogram,
            metadata=metadata,
            unit=unit_overrides
        )



