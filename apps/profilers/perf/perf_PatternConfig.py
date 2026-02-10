from profilers.PatternConfig import PatternConfig

class PerfConfig(PatternConfig):
    @classmethod
    def populating(cls, report_data, metadata=None, level="custom"):
        """
        Populate the PatternConfig attributes using Perf raw data.

        Parameters
        ----------
        report_data : dict
            Dictionary containing raw event counts and derived metrics from perf_profilers.py.
            Expected keys include:
              - L1: l1d_loads, l1d_load_misses, l1d_stores, l1i_load_misses
              - L2 loads: l2_load_hits, l2_load_misses, l2_total_reads
              - L2 stores (RFO): l2_rfo_total, l2_rfo_hits, l2_rfo_misses, l2_total_writes
              - L3/LLC: l3_hits, l3_misses, llc_loads, llc_load_misses
              - DRAM: dram_total_reads, dram_local, dram_remote
              - General: time_elapsed
        metadata : BaseMetadata, optional
            Optional system metadata.
        level : str
            Cache/memory level to focus on: "l1", "l2", "l3", or "dram".

        Returns
        -------
        PerfConfig
            An initialized config object containing read/write frequencies and totals.
        """
        # =================================================================
        # Extract raw metrics from perf_profilers.py output
        # =================================================================

        # L1 Data Cache
        l1d_loads = report_data.get('l1d_loads', 0)
        l1d_load_misses = report_data.get('l1d_load_misses', 0)
        l1d_stores = report_data.get('l1d_stores', 0)
        l1i_load_misses = report_data.get('l1i_load_misses', 0)

        # L2 Cache - Loads (from derived metrics)
        l2_total_reads = report_data.get('l2_total_reads', 0)
        l2_read_hits = report_data.get('l2_read_hits', 0)
        l2_read_misses = report_data.get('l2_read_misses', 0)

        # L2 Cache - Stores via RFO (from derived metrics)
        l2_total_writes = report_data.get('l2_total_writes', 0)
        l2_write_hits = report_data.get('l2_write_hits', 0)
        l2_write_misses = report_data.get('l2_write_misses', 0)

        # Also check raw RFO events directly
        l2_rfo_total = report_data.get('l2_rfo_total', 0)
        if l2_total_writes == 0 and l2_rfo_total > 0:
            l2_total_writes = l2_rfo_total

        # L3/LLC Cache
        l3_total_reads = report_data.get('l3_total_reads', 0)
        l3_total_writes = report_data.get('l3_total_writes', 0)
        l3_read_hits = report_data.get('l3_read_hits', 0)
        l3_read_misses = report_data.get('l3_read_misses', 0)
        llc_loads = report_data.get('llc_loads', 0)
        llc_load_misses = report_data.get('llc_load_misses', 0)

        # DRAM
        dram_total_reads = report_data.get('dram_total_reads', 0)
        dram_local_reads = report_data.get('dram_local_reads', 0)
        dram_remote_reads = report_data.get('dram_remote_reads', 0)

        # Timing
        time_elapsed = report_data.get('time_elapsed', 0)

        # =================================================================
        # Initialize outputs
        # =================================================================
        read_freq = 0
        write_freq = 0
        total_reads = 0
        total_writes = 0
        total_reads_d = 0
        total_reads_i = 0
        total_writes_d = 0
        total_writes_i = 0

        # =================================================================
        # Level-specific calculations
        # =================================================================
        if level == "l1":
            # L1 level: direct loads/stores
            total_reads_d = l1d_loads
            total_reads_i = l1i_load_misses  # Instruction fetches that missed L1I
            total_reads = total_reads_d + total_reads_i

            total_writes_d = l1d_stores
            total_writes_i = 0
            total_writes = total_writes_d

            read_freq = total_reads / time_elapsed if time_elapsed else 0
            write_freq = total_writes / time_elapsed if time_elapsed else 0

        elif level == "l2":
            # L2 level: sees L1 misses (loads) and RFO requests (stores)
            # Total reads at L2 = L1 load misses (cascade rule)
            total_reads = l2_total_reads if l2_total_reads > 0 else (l1d_load_misses + l1i_load_misses)

            # Total writes at L2 = RFO requests = L1 store misses equivalent
            total_writes = l2_total_writes

            # Breakdown
            total_reads_d = l2_read_hits + l2_read_misses if (l2_read_hits + l2_read_misses) > 0 else l1d_load_misses
            total_reads_i = l1i_load_misses
            total_writes_d = l2_write_hits + l2_write_misses if (l2_write_hits + l2_write_misses) > 0 else l2_total_writes
            total_writes_i = 0

            read_freq = total_reads / time_elapsed if time_elapsed else 0
            write_freq = total_writes / time_elapsed if time_elapsed else 0

        elif level == "l3":
            # L3/LLC level: sees L2 misses (cascade rule)
            # Use derived l3_total_reads if available, otherwise fall back
            if l3_total_reads > 0:
                total_reads = l3_total_reads
            elif llc_loads > 0:
                total_reads = llc_loads
            elif l2_read_misses > 0:
                total_reads = l2_read_misses
            else:
                total_reads = 0

            # L3 writes = L2 RFO misses (stores that missed L2)
            total_writes = l3_total_writes

            # Breakdown
            total_reads_d = l3_read_hits + l3_read_misses if (l3_read_hits + l3_read_misses) > 0 else total_reads
            total_reads_i = 0
            total_writes_d = total_writes
            total_writes_i = 0

            read_freq = total_reads / time_elapsed if time_elapsed else 0
            write_freq = total_writes / time_elapsed if time_elapsed else 0

        elif level == "dram":
            # DRAM level: sees LLC/L3 misses (cascade rule)
            total_reads = dram_total_reads if dram_total_reads > 0 else llc_load_misses
            total_writes = 0  # Write-back data not directly measured by these events

            total_reads_d = dram_local_reads + dram_remote_reads if (dram_local_reads + dram_remote_reads) > 0 else total_reads
            total_reads_i = 0
            total_writes_d = 0
            total_writes_i = 0

            read_freq = total_reads / time_elapsed if time_elapsed else 0
            write_freq = total_writes / time_elapsed if time_elapsed else 0

        return cls(
            exp_name="PerfProfilers",
            benchmark_name=report_data.get("benchmark", "Benchmark1"),
            total_reads=total_reads,
            total_writes=total_writes,
            read_freq=read_freq,
            write_freq=write_freq,
            total_reads_d=total_reads_d,
            total_reads_i=total_reads_i,
            total_writes_d=total_writes_d,
            total_writes_i=total_writes_i,
            read_size=64,  # Cache line size (64 bytes)
            write_size=64,
            metadata=metadata
        )