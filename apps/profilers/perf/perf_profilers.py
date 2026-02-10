from profilers.FrontendInterface import FrontendInterface
import subprocess
import re
import os

# Cache line size in bytes
CACHE_LINE_SIZE = 64

# Generic events - work on both AMD and Intel
GENERIC_EVENTS = {
    "l1_data": [
        ("L1-dcache-loads:u", "l1d_loads"),
        ("L1-dcache-load-misses:u", "l1d_load_misses"),
        ("L1-dcache-stores:u", "l1d_stores"),
    ],
    "l1_instruction": [
        ("L1-icache-load-misses:u", "l1i_load_misses"),
    ],
    "llc": [
        ("LLC-loads:u", "llc_loads"),
        ("LLC-load-misses:u", "llc_load_misses"),
        ("LLC-stores:u", "llc_stores"),
    ],
    "general": [
        ("cycles:u", "cycles"),
        ("instructions:u", "instructions"),
    ],
}

# Intel-specific events
# Run `perf list | grep mem_load` to check naming on your system
INTEL_EVENTS = {
    "l2_loads": [
        ("mem_load_retired.l2_hit:u", "l2_load_hits"),
        ("mem_load_retired.l2_miss:u", "l2_load_misses"),
    ],
    "l2_stores": [
        ("l2_rqsts.all_rfo:u", "l2_rfo_total"),
        ("l2_rqsts.rfo_hit:u", "l2_rfo_hits"),
        ("l2_rqsts.rfo_miss:u", "l2_rfo_misses"),
    ],
    "l3": [
        ("mem_load_retired.l3_hit:u", "l3_hits"),
        ("mem_load_retired.l3_miss:u", "l3_misses"),
    ],
    "dram": [
        ("mem_load_l3_miss_retired.local_dram:u", "dram_local"),
        ("mem_load_l3_miss_retired.remote_dram:u", "dram_remote"),
    ],
}

# AMD Zen-specific events
AMD_EVENTS = {
    "l2": [
        ("l2_cache_req_stat.ic_dc_hit_in_l2:u", "l2_hits"),
        ("l2_cache_req_stat.ic_dc_miss_in_l2:u", "l2_misses"),
        ("l2_pf_miss_l2_hit_l3:u", "l2_pf_hit_l3"),
        ("l2_pf_miss_l2_l3:u", "l2_pf_miss_l3"),
    ],
    "l3": [
        ("l3_comb_clstr_state.request_miss:u", "l3_misses"),
    ],
    "interconnect": [
        ("xi_ccx_sdp_req1:u", "ccx_requests"),
    ],
    "memory": [
        ("ls_dmnd_fills_from_sys.mem_io_local:u", "dram_local_demand"),
        ("ls_dmnd_fills_from_sys.mem_io_remote:u", "dram_remote_demand"),
        ("ls_any_fills_from_sys.mem_io_local:u", "dram_local_any"),
        ("ls_any_fills_from_sys.mem_io_remote:u", "dram_remote_any"),
    ],
}

SUPPORTED_ARCHS = ["generic", "amd", "intel"]


def _safe_div(numerator, denominator, default=0.0):
    """Safe division that returns default if denominator is zero."""
    return numerator / denominator if denominator != 0 else default


class PerfProfilers(FrontendInterface):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.executable_cmd = self.config.get("executable")
        self.action = self.config.get("action")
        self.level = self.config.get("level", "custom")
        self.arch = self.config.get("arch", "generic").lower()
        self.repeat = self.config.get("repeat", 3)

        if self.arch not in SUPPORTED_ARCHS:
            raise ValueError(f"Unsupported arch '{self.arch}'. Must be one of: {SUPPORTED_ARCHS}")

        self.output = ""
        self.report = None
        self.data = {}
        self.active_events = self._get_events_for_arch()

    def _get_events_for_arch(self):
        """Get the list of events based on architecture."""
        events = []
        for category in GENERIC_EVENTS.values():
            events.extend(category)

        if self.arch == "amd":
            for category in AMD_EVENTS.values():
                events.extend(category)
        elif self.arch == "intel":
            for category in INTEL_EVENTS.values():
                events.extend(category)

        return events

    def _build_event_string(self):
        """Build comma-separated event string from active events."""
        return ",".join(event for event, _ in self.active_events)

    def construct_command(self):
        """Construct the perf command with target event counters."""
        if isinstance(self.executable_cmd, str):
            executable_with_args = self.executable_cmd.split()
        else:
            executable_with_args = list(self.executable_cmd)

        if hasattr(self.config, "get") and "executable_args" in self.config:
            exec_args = self.config.get("executable_args") or []
            if isinstance(exec_args, str):
                exec_args = [exec_args]
            executable_with_args += exec_args

        report = os.path.basename(executable_with_args[0])
        event_string = self._build_event_string()

        perf_command = [
            "perf", "stat",
            "-r", str(self.repeat),
            "-e", event_string,
            "-o", "/dev/stdout"
        ] + executable_with_args

        return perf_command, report

    def profiling(self, **kwargs):
        """Run the target executable under perf stat and save output."""
        perf_command, report = self.construct_command()
        try:
            print(f"Executing: {' '.join(perf_command)}")
            profiler_data = subprocess.run(
                perf_command,
                check=True,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
        except subprocess.CalledProcessError as e:
            print(f"Command failed with exit code {e.returncode}")
            print(f"stderr: {e.stderr}")
            raise

        self.output = profiler_data.stdout + profiler_data.stderr

        print("\n--- Raw perf output ---")
        print(self.output)
        print("--- End perf output ---\n")

        if self.action == "profiling":
            self.report = f"{report}.perf-rep"
            with open(self.report, 'w') as perf_report:
                perf_report.write(f"Profiling output:\n {self.output}")
            print(f"Output written to file {report}.perf-rep")

    def extract_metrics(self, report_file=None, **kwargs):
        """Extract performance metrics from perf output."""
        toparse = ""
        if self.action == "extract_metrics":
            with open(report_file) as file:
                toparse = file.read()
        if self.action == "both":
            toparse = self.output

        try:
            for event_name, metric_key in self.active_events:
                base_event = re.sub(r':[uk]+$', '', event_name)
                escaped_event = re.escape(base_event)
                pattern = rf"([\d,]+)\s+{escaped_event}"
                match = re.search(pattern, toparse)
                value = int(match.group(1).replace(',', '')) if match else 0
                self.data[metric_key] = value

            time_match = re.search(r"([\d.]+)\s+seconds time elapsed", toparse)
            self.data["time_elapsed"] = float(time_match.group(1)) if time_match else 0.0

            self._compute_derived_metrics()
            return self.data

        except AttributeError as e:
            print(f"Failed to extract data: {e}")
            raise

    def _compute_derived_metrics(self):
        """Compute derived metrics based on the memory hierarchy cascade model."""
        data = self.data

        # L1 Data Cache
        l1d_loads = data.get("l1d_loads", 0)
        l1d_load_misses = data.get("l1d_load_misses", 0)
        l1d_stores = data.get("l1d_stores", 0)

        data["l1d_total_reads"] = l1d_loads
        data["l1d_total_writes"] = l1d_stores
        data["l1d_read_hits"] = l1d_loads - l1d_load_misses
        data["l1d_read_hit_rate"] = _safe_div(l1d_loads - l1d_load_misses, l1d_loads)
        data["l1d_read_miss_rate"] = _safe_div(l1d_load_misses, l1d_loads)

        # L2 Cache
        l1i_load_misses = data.get("l1i_load_misses", 0)
        data["l2_total_reads"] = l1d_load_misses + l1i_load_misses

        l2_load_hits = data.get("l2_load_hits", 0) or data.get("l2_hits", 0)
        l2_load_misses = data.get("l2_load_misses", 0) or data.get("l2_misses", 0)

        if l2_load_hits > 0 or l2_load_misses > 0:
            data["l2_read_hits"] = l2_load_hits
            data["l2_read_misses"] = l2_load_misses
            l2_load_total = l2_load_hits + l2_load_misses
            data["l2_read_hit_rate"] = _safe_div(l2_load_hits, l2_load_total)
            data["l2_read_miss_rate"] = _safe_div(l2_load_misses, l2_load_total)
        else:
            data["l2_read_hits"] = 0
            data["l2_read_misses"] = 0
            data["l2_read_hit_rate"] = 0.0
            data["l2_read_miss_rate"] = 0.0

        # L2 Stores via RFO
        l2_rfo_total = data.get("l2_rfo_total", 0)
        l2_rfo_hits = data.get("l2_rfo_hits", 0)
        l2_rfo_misses = data.get("l2_rfo_misses", 0)

        data["l2_total_writes"] = l2_rfo_total
        data["l1d_store_misses"] = l2_rfo_total

        if l2_rfo_total > 0:
            data["l2_write_hits"] = l2_rfo_hits
            data["l2_write_misses"] = l2_rfo_misses
            data["l2_write_hit_rate"] = _safe_div(l2_rfo_hits, l2_rfo_total)
            data["l2_write_miss_rate"] = _safe_div(l2_rfo_misses, l2_rfo_total)
        else:
            data["l2_write_hits"] = 0
            data["l2_write_misses"] = 0
            data["l2_write_hit_rate"] = 0.0
            data["l2_write_miss_rate"] = 0.0

        # L3/LLC Cache
        llc_loads = data.get("llc_loads", 0)
        llc_load_misses = data.get("llc_load_misses", 0)
        llc_stores = data.get("llc_stores", 0)

        if l2_load_misses > 0:
            data["l3_total_reads"] = l2_load_misses
        else:
            data["l3_total_reads"] = llc_loads

        if l2_rfo_misses > 0:
            data["l3_total_writes"] = l2_rfo_misses
        else:
            data["l3_total_writes"] = llc_stores

        l3_hits = data.get("l3_hits", 0)
        l3_misses = data.get("l3_misses", 0)

        if l3_hits > 0 or l3_misses > 0:
            data["l3_read_hits"] = l3_hits
            data["l3_read_misses"] = l3_misses
            l3_total = l3_hits + l3_misses
            data["l3_read_hit_rate"] = _safe_div(l3_hits, l3_total)
            data["l3_read_miss_rate"] = _safe_div(l3_misses, l3_total)
        elif llc_loads > 0:
            data["l3_read_hits"] = llc_loads - llc_load_misses
            data["l3_read_misses"] = llc_load_misses
            data["l3_read_hit_rate"] = _safe_div(llc_loads - llc_load_misses, llc_loads)
            data["l3_read_miss_rate"] = _safe_div(llc_load_misses, llc_loads)
        else:
            data["l3_read_hits"] = 0
            data["l3_read_misses"] = 0
            data["l3_read_hit_rate"] = 0.0
            data["l3_read_miss_rate"] = 0.0

        # DRAM
        if l3_misses > 0:
            data["dram_total_reads"] = l3_misses
        else:
            data["dram_total_reads"] = llc_load_misses

        dram_local = data.get("dram_local", 0) or data.get("dram_local_demand", 0)
        dram_remote = data.get("dram_remote", 0) or data.get("dram_remote_demand", 0)
        data["dram_local_reads"] = dram_local
        data["dram_remote_reads"] = dram_remote

        if dram_local > 0 or dram_remote > 0:
            total_dram = dram_local + dram_remote
            data["dram_local_ratio"] = _safe_div(dram_local, total_dram)
            data["dram_remote_ratio"] = _safe_div(dram_remote, total_dram)

        # Bandwidth
        l1_to_l2_load_bytes = l1d_load_misses * CACHE_LINE_SIZE
        l1_to_l2_store_bytes = l2_rfo_total * CACHE_LINE_SIZE
        data["l1_to_l2_bytes"] = l1_to_l2_load_bytes + l1_to_l2_store_bytes
        data["l1_to_l2_read_bytes"] = l1_to_l2_load_bytes
        data["l1_to_l2_write_bytes"] = l1_to_l2_store_bytes

        l2_to_l3_load_bytes = l2_load_misses * CACHE_LINE_SIZE if l2_load_misses > 0 else 0
        l2_to_l3_store_bytes = l2_rfo_misses * CACHE_LINE_SIZE if l2_rfo_misses > 0 else 0
        data["l2_to_l3_bytes"] = l2_to_l3_load_bytes + l2_to_l3_store_bytes
        data["l2_to_l3_read_bytes"] = l2_to_l3_load_bytes
        data["l2_to_l3_write_bytes"] = l2_to_l3_store_bytes

        data["l3_to_dram_bytes"] = data["dram_total_reads"] * CACHE_LINE_SIZE

        # General metrics
        cycles = data.get("cycles", 0)
        instructions = data.get("instructions", 0)
        time_elapsed = data.get("time_elapsed", 0.0)

        data["ipc"] = _safe_div(instructions, cycles)
        data["cpi"] = _safe_div(cycles, instructions)
        data["l1d_mpki"] = _safe_div(l1d_load_misses, instructions) * 1000
        data["llc_mpki"] = _safe_div(llc_load_misses, instructions) * 1000

        if time_elapsed > 0:
            data["l1_to_l2_bandwidth"] = data["l1_to_l2_bytes"] / time_elapsed
            data["l3_to_dram_bandwidth"] = data["l3_to_dram_bytes"] / time_elapsed

    def get_level_summary(self, level):
        """Get a summary of metrics for a specific memory level."""
        level = level.lower()
        summary = {}

        if level == "l1" or level == "l1d":
            summary = {
                "total_reads": self.data.get("l1d_total_reads", 0),
                "total_writes": self.data.get("l1d_total_writes", 0),
                "read_hits": self.data.get("l1d_read_hits", 0),
                "read_misses": self.data.get("l1d_load_misses", 0),
                "hit_rate": self.data.get("l1d_read_hit_rate", 0.0),
                "miss_rate": self.data.get("l1d_read_miss_rate", 0.0),
                "mpki": self.data.get("l1d_mpki", 0.0),
            }
        elif level == "l2":
            summary = {
                "total_reads": self.data.get("l2_total_reads", 0),
                "total_writes": self.data.get("l2_total_writes", 0),
                "read_hits": self.data.get("l2_read_hits", 0),
                "read_misses": self.data.get("l2_read_misses", 0),
                "hit_rate": self.data.get("l2_read_hit_rate", 0.0),
                "miss_rate": self.data.get("l2_read_miss_rate", 0.0),
            }
        elif level == "l3" or level == "llc":
            summary = {
                "total_reads": self.data.get("l3_total_reads", 0),
                "total_writes": self.data.get("l3_total_writes", 0),
                "read_hits": self.data.get("l3_read_hits", 0),
                "read_misses": self.data.get("l3_read_misses", 0),
                "hit_rate": self.data.get("l3_read_hit_rate", 0.0),
                "miss_rate": self.data.get("l3_read_miss_rate", 0.0),
                "mpki": self.data.get("llc_mpki", 0.0),
            }
        elif level == "dram" or level == "memory":
            summary = {
                "total_reads": self.data.get("dram_total_reads", 0),
                "local_reads": self.data.get("dram_local_reads", 0),
                "remote_reads": self.data.get("dram_remote_reads", 0),
                "local_ratio": self.data.get("dram_local_ratio", 0.0),
                "bandwidth_bytes": self.data.get("l3_to_dram_bytes", 0),
            }

        return summary

    def print_summary(self):
        """Print a formatted summary of all memory hierarchy metrics."""
        print("\n" + "=" * 70)
        print("MEMORY HIERARCHY SUMMARY")
        print("=" * 70)

        print(f"\nGeneral:")
        print(f"  Instructions:     {self.data.get('instructions', 0):,}")
        print(f"  Cycles:           {self.data.get('cycles', 0):,}")
        print(f"  IPC:              {self.data.get('ipc', 0):.3f}")
        print(f"  Time Elapsed:     {self.data.get('time_elapsed', 0):.3f}s")

        print(f"\nL1 Data Cache:")
        print(f"  Total Reads:      {self.data.get('l1d_total_reads', 0):,}")
        print(f"  Total Writes:     {self.data.get('l1d_total_writes', 0):,}")
        print(f"  Read Hits:        {self.data.get('l1d_read_hits', 0):,}")
        print(f"  Read Misses:      {self.data.get('l1d_load_misses', 0):,}")
        print(f"  Hit Rate:         {self.data.get('l1d_read_hit_rate', 0) * 100:.2f}%")
        print(f"  MPKI:             {self.data.get('l1d_mpki', 0):.2f}")

        print(f"\nL2 Cache:")
        print(f"  Total Reads:      {self.data.get('l2_total_reads', 0):,}")
        print(f"  Total Writes:     {self.data.get('l2_total_writes', 0):,}")
        if self.data.get('l2_read_hits', 0) > 0 or self.data.get('l2_read_misses', 0) > 0:
            print(f"  Read Hits:        {self.data.get('l2_read_hits', 0):,}")
            print(f"  Read Misses:      {self.data.get('l2_read_misses', 0):,}")
            print(f"  Hit Rate:         {self.data.get('l2_read_hit_rate', 0) * 100:.2f}%")

        print(f"\nL3/LLC Cache:")
        print(f"  Total Reads:      {self.data.get('l3_total_reads', 0):,}")
        print(f"  Total Writes:     {self.data.get('l3_total_writes', 0):,}")
        print(f"  Read Hits:        {self.data.get('l3_read_hits', 0):,}")
        print(f"  Read Misses:      {self.data.get('l3_read_misses', 0):,}")
        print(f"  Hit Rate:         {self.data.get('l3_read_hit_rate', 0) * 100:.2f}%")

        print(f"\nDRAM:")
        print(f"  Total Reads:      {self.data.get('dram_total_reads', 0):,}")
        if self.data.get('dram_local_reads', 0) > 0 or self.data.get('dram_remote_reads', 0) > 0:
            print(f"  Local Reads:      {self.data.get('dram_local_reads', 0):,}")
            print(f"  Remote Reads:     {self.data.get('dram_remote_reads', 0):,}")
            print(f"  Local Ratio:      {self.data.get('dram_local_ratio', 0) * 100:.2f}%")

        print(f"\nBandwidth Estimates:")
        print(f"  L1 -> L2:         {self.data.get('l1_to_l2_bytes', 0) / 1e6:.2f} MB")
        print(f"  L2 -> L3:         {self.data.get('l2_to_l3_bytes', 0) / 1e6:.2f} MB")
        print(f"  L3 -> DRAM:       {self.data.get('l3_to_dram_bytes', 0) / 1e6:.2f} MB")

        print("=" * 70 + "\n")

    @classmethod
    def required_profiling_args(cls):
        """Define required arguments for profiling."""
        return ["executable", "level"]

    @classmethod
    def optional_profiling_args(cls):
        """Define optional arguments for profiling."""
        return [
            {
                "name": "arch",
                "help": "CPU architecture: generic, amd, or intel",
                "choices": SUPPORTED_ARCHS,
                "default": "generic",
            },
            {
                "name": "repeat",
                "help": "Number of measurement repeats (default: 3)",
                "type": int,
                "default": 3,
            },
        ]

    @classmethod
    def required_extract_args(cls, action):
        """Define required arguments for metric extraction."""
        if action == "extract_metrics":
            return ["report_file"]
        else:
            return []
