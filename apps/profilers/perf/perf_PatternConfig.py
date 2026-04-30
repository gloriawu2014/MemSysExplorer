from profilers.PatternConfig import PatternConfig

class PerfConfig(PatternConfig):
    def __init__(self, **kwargs):
        """
        Initialize PerfConfig with additional cache hit/miss fields.

        Parameters
        ----------
        **kwargs : dict
            Configuration parameters, including cache hit/miss stats
        """
        # Extract cache hit/miss stats before calling parent init
        self.load_hits = kwargs.pop('load_hits', 0)
        self.load_misses = kwargs.pop('load_misses', 0)
        self.store_hits = kwargs.pop('store_hits', 0)
        self.store_misses = kwargs.pop('store_misses', 0)

        # Preserve raw L2 counters so they are included in JSON output
        self.l2_pf_hit_l3 = kwargs.pop('l2_pf_hit_l3', 0)
        self.l2_pf_miss_l3 = kwargs.pop('l2_pf_miss_l3', 0)
        self.l2_cache_accesses_from_dc_misses = kwargs.pop('l2_cache_accesses_from_dc_misses', 0)

        # Call parent constructor
        super().__init__(**kwargs)

    @classmethod
    def populating(cls, report_data, metadata=None):
        """
        Normalize raw perf counters into standard cascade metrics.

        Works with ONLY the raw counters available for the specified level.
        No cross-level dependencies required.

        Parameters
        ----------
        report_data : dict
            Dictionary containing raw event counts from perf_profilers.py.
            Expected keys by level:
              - L1: l1d_loads, l1d_load_misses, l1d_stores, l1i_load_misses
              - L2: l2_load_hits, l2_load_misses, l2_rfo_total, l2_rfo_hits, l2_rfo_misses
              - L3: l3_hits, l3_misses, llc_loads, llc_load_misses, llc_stores
              - DRAM: dram_local, dram_remote, dram_write_local, dram_write_remote
              - Always: time_elapsed, level
        metadata : BaseMetadata, optional
            Optional system metadata.

        Returns
        -------
        PerfConfig
            An initialized config object with standard metrics:
            total_reads, total_writes, load_hits, load_misses, store_hits, store_misses
        """
        # Get level from data (included by perf_profilers)
        level = report_data.get("level", "l2")
        time_elapsed = report_data.get("time_elapsed", 0)

        # Initialize standard outputs
        total_reads = 0
        total_writes = 0
        load_hits = 0
        load_misses = 0
        store_hits = 0
        store_misses = 0

        # Default raw L2 event fields for JSON output
        l2_pf_hit_l3 = 0
        l2_pf_miss_l3 = 0
        l2_cache_accesses_from_dc_misses = 0

        # =================================================================
        # Map raw counters to standard metrics based on level
        # =================================================================
        if level == "l1":
            # Raw counters: l1d_loads, l1d_load_misses, l1d_stores, l1i_load_misses
            l1d_loads = report_data.get("l1d_loads", 0)
            l1d_load_misses = report_data.get("l1d_load_misses", 0)
            l1d_stores = report_data.get("l1d_stores", 0)

            total_reads = l1d_loads
            total_writes = l1d_stores
            load_misses = l1d_load_misses
            load_hits = max(0, total_reads - load_misses)
            # L1 store misses not directly available without L2 RFO data
            store_hits = total_writes
            store_misses = 0

        elif level == "l2":
            # Raw counters: l2_load_hits, l2_load_misses, l2_rfo_total, l2_rfo_hits, l2_rfo_misses
            load_hits = report_data.get("l2_load_hits", 0)
            load_misses = report_data.get("l2_load_misses", 0)
            total_reads = load_hits + load_misses

            # RFO counters (Intel has these, AMD may not)
            store_hits = report_data.get("l2_rfo_hits", 0)
            store_misses = report_data.get("l2_rfo_misses", 0)
            total_writes = report_data.get("l2_rfo_total", 0)
            # Fallback if rfo_total missing but hits/misses present
            if total_writes == 0:
                total_writes = store_hits + store_misses

            # Preserve additional L2 event counters for JSON output
            l2_pf_hit_l3 = report_data.get("l2_pf_hit_l3", 0)
            l2_pf_miss_l3 = report_data.get("l2_pf_miss_l3", 0)
            l2_cache_accesses_from_dc_misses = report_data.get("l2_cache_accesses_from_dc_misses", 0)

        elif level == "l3":
            # Raw counters: l3_hits, l3_misses, llc_loads, llc_load_misses, llc_stores
            l3_hits = report_data.get("l3_hits", 0)
            l3_misses = report_data.get("l3_misses", 0)
            llc_loads = report_data.get("llc_loads", 0)
            llc_load_misses = report_data.get("llc_load_misses", 0)
            llc_stores = report_data.get("llc_stores", 0)

            # Prefer specific l3_hits/l3_misses, fallback to LLC generic
            if l3_hits > 0 or l3_misses > 0:
                load_hits = l3_hits
                load_misses = l3_misses
                total_reads = l3_hits + l3_misses
            else:
                load_misses = llc_load_misses
                load_hits = max(0, llc_loads - llc_load_misses)
                total_reads = llc_loads

            total_writes = llc_stores
            store_hits = llc_stores  # No direct miss counter for stores
            store_misses = 0

        elif level == "dram":
            # Raw counters: dram_local, dram_remote, dram_write_local, dram_write_remote
            dram_local = report_data.get("dram_local", 0)
            dram_remote = report_data.get("dram_remote", 0)
            dram_write_local = report_data.get("dram_write_local", 0)
            dram_write_remote = report_data.get("dram_write_remote", 0)

            total_reads = dram_local + dram_remote
            total_writes = dram_write_local + dram_write_remote

            # DRAM is final level - all accesses are served
            load_hits = total_reads
            load_misses = 0
            store_hits = total_writes
            store_misses = 0

        # =================================================================
        # Compute rates
        # =================================================================
        read_freq = total_reads / time_elapsed if time_elapsed > 0 else 0
        write_freq = total_writes / time_elapsed if time_elapsed > 0 else 0

        # Unit overrides
        unit_overrides = {
            "read_freq": "count/s",
            "write_freq": "count/s",
            "total_reads": "count",
            "total_writes": "count",
            "load_hits": "count",
            "load_misses": "count",
            "store_hits": "count",
            "store_misses": "count"
        }

        return cls(
            exp_name="PerfProfilers",
            benchmark_name=report_data.get("benchmark", "Benchmark1"),
            total_reads=total_reads,
            total_writes=total_writes,
            read_freq=read_freq,
            write_freq=write_freq,
            total_reads_d=total_reads,
            total_reads_i=0,
            total_writes_d=total_writes,
            total_writes_i=0,
            read_size=64,
            write_size=64,
            execution_time=time_elapsed,
            load_hits=load_hits,
            load_misses=load_misses,
            store_hits=store_hits,
            store_misses=store_misses,
            l2_pf_hit_l3=l2_pf_hit_l3,
            l2_pf_miss_l3=l2_pf_miss_l3,
            l2_cache_accesses_from_dc_misses=l2_cache_accesses_from_dc_misses,
            metadata=metadata,
            unit=unit_overrides
        )
