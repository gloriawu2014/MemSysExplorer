from profilers.FrontendInterface import FrontendInterface
import subprocess
import re
import os

class PerfProfilers(FrontendInterface):
    def __init__(self, **kwargs):
        """
        Initialize the PerfProfiler with profiling parameters.

        Parameters
        ----------
        executable : list
            Executable command as a list.
        action : str
            One of 'profiling', 'extract_metrics', or 'both'.
        level : str
            Cache/memory level to extract statistics for (e.g., l1, l2, dram).
        """
        super().__init__(**kwargs)
        #self.executable_cmd = " ".join(self.config.get("executable", []))
        self.executable_cmd = self.config.get("executable")
        self.action = self.config.get("action")
        self.level = self.config.get("level", "custom")  # Default to custom if not provided
        
        self.output = ""
        self.report = None
        self.data = {}
    
    # return a command that runs perf stats 
    def constuct_command(self):
        """
        Construct the perf command with target event counters.

        Returns
        -------
        list
            Full command to run perf stat.
        str
            Basename of the executable for file naming.
        """        
        if isinstance(self.executable_cmd, str):
            executable_with_args = [self.executable_cmd]
        else:
            executable_with_args = self.executable_cmd.split()

        if hasattr(self.config, "get") and "executable_args" in self.config:
            exec_args = self.config.get("executable_args") or []
            if isinstance(exec_args, str):
                exec_args = [exec_args]
            executable_with_args += exec_args
        
        report = os.path.basename(executable_with_args[0]) 
        perf_command = [
            "perf", "stat", "-e", "L1-dcache-loads:u,L1-dcache-load-misses:u,L1-icache-loads:u,L1-icache-load-misses:u,L1-dcache-prefetches:u,L1-icache-prefetches:u,l2_cache_req_stat.ic_dc_hit_in_l2:u,l2_cache_req_stat.ic_dc_miss_in_l2:u,l2_pf_miss_l2_hit_l3:u,l2_pf_miss_l2_l3:u,l3_comb_clstr_state.request_miss:u,xi_ccx_sdp_req1:u,ls_dmnd_fills_from_sys.mem_io_local:u,ls_dmnd_fills_from_sys.mem_io_remote:u,ls_any_fills_from_sys.mem_io_remote:u,ls_any_fills_from_sys.mem_io_local:u",
            "-o", "/dev/stdout"
        ] + executable_with_args
        return perf_command, report


    # collect all the data that will be stored in a log file
    def profiling(self, **kwargs):
        """
        Run the target executable under `perf stat` and save output.

        Raises
        ------
        CalledProcessError
            If perf execution fails.
        """
        # self.validate_paths()
        perf_command, report = self.constuct_command() 
        try:
            print(f"Executing: {' '.join(perf_command)}")
            profiler_data = subprocess.run(perf_command, check=True, text=True, stdout=subprocess.PIPE)
        except subprocess.CalledProcessError as e:
            print(f"Command failed with exit code {e.stderr}")
            raise
        self.output = profiler_data.stdout


        # store output to file
        if self.action == "profiling":
            self.report = f"{report}.perf-rep"
            with open(self.report, 'w') as perf_report:
                perf_report.write(f"Profiling output:\n {self.output}")
            print(f"Output written to file {report}.perf-rep")

    def extract_metrics(self, report_file=None, **kwargs):
        """
        Extract performance metrics from perf output (file or in-memory).

        Parameters
        ----------
        report_file : str, optional
            File path to the perf output (used when action is 'extract_metrics').

        Returns
        -------
        dict
            Dictionary of parsed performance counters.

        Raises
        ------
        AttributeError
            If regex matching fails.
        """
        toparse = ""
        if self.action == "extract_metrics":
            with open(report_file) as file:
                 toparse = file.read()
        if self.action == "both":
            toparse = self.output

        # Extract the memory statistics
        try:
       
            # Extract the cache and memory-related statistics from perf output safely
            def extract_perf_metric(pattern, text):
                """Extracts a numeric value from perf output safely, returning None if not found."""
                match = re.search(pattern, text)
                return int(match.group(1).replace(',', '')) if match else None

            # Extract memory statistics
            l1_dcache_loads = extract_perf_metric(r"([\d,]+)\s+L1-dcache-loads:u", toparse)
            l1_dcache_load_misses = extract_perf_metric(r"([\d,]+)\s+L1-dcache-load-misses:u", toparse)
            l1_icache_loads = extract_perf_metric(r"([\d,]+)\s+L1-icache-loads:u", toparse)
            l1_icache_load_misses = extract_perf_metric(r"([\d,]+)\s+L1-icache-load-misses:u", toparse)
            l1_dcache_prefetches = extract_perf_metric(r"([\d,]+)\s+L1-dcache-prefetches:u", toparse)
            l1_icache_prefetches = extract_perf_metric(r"([\d,]+)\s+L1-icache-prefetches:u", toparse)
            l2_ic_dc_hit_in_l2 = extract_perf_metric(r"([\d,]+)\s+l2_cache_req_stat.ic_dc_hit_in_l2:u", toparse)
            l2_ic_dc_miss_in_l2 = extract_perf_metric(r"([\d,]+)\s+l2_cache_req_stat.ic_dc_miss_in_l2:u", toparse)
            l2_pf_miss_l2_hit_l3 = extract_perf_metric(r"([\d,]+)\s+l2_pf_miss_l2_hit_l3:u", toparse)
            l2_pf_miss_l2_l3 = extract_perf_metric(r"([\d,]+)\s+l2_pf_miss_l2_l3:u", toparse)
            l3_request_miss = extract_perf_metric(r"([\d,]+)\s+l3_comb_clstr_state.request_miss:u", toparse)
            xi_ccx_sdp_req1 = extract_perf_metric(r"([\d,]+)\s+xi_ccx_sdp_req1:u", toparse)
            mem_io_local_dmnd = extract_perf_metric(r"([\d,]+)\s+ls_dmnd_fills_from_sys.mem_io_local:u", toparse)
            mem_io_remote_dmnd = extract_perf_metric(r"([\d,]+)\s+ls_dmnd_fills_from_sys.mem_io_remote:u", toparse)
            mem_io_remote_any = extract_perf_metric(r"([\d,]+)\s+ls_any_fills_from_sys.mem_io_remote:u", toparse)
            mem_io_local_any = extract_perf_metric(r"([\d,]+)\s+ls_any_fills_from_sys.mem_io_local:u", toparse)
            time_elapsed = extract_perf_metric(r"([\d,]+)\s+seconds time elapsed", toparse) 



            # Update the data dictionary with the extracted values
            self.data.update({
            "l1_dcache_loads": int(l1_dcache_loads) if l1_dcache_loads else 0,
            "l1_dcache_load_misses": int(l1_dcache_load_misses) if l1_dcache_load_misses else 0,
            "l1_icache_loads": int(l1_icache_loads) if l1_icache_loads else 0,
            "l1_icache_load_misses": int(l1_icache_load_misses) if l1_icache_load_misses else 0,
            "l1_dcache_prefetches": int(l1_dcache_prefetches) if l1_dcache_prefetches else 0,
            "l1_icache_prefetches": int(l1_icache_prefetches) if l1_icache_prefetches else 0,
            "l2_ic_dc_hit_in_l2": int(l2_ic_dc_hit_in_l2) if l2_ic_dc_hit_in_l2 else 0,
            "l2_ic_dc_miss_in_l2": int(l2_ic_dc_miss_in_l2) if l2_ic_dc_miss_in_l2 else 0,
            "l2_pf_miss_l2_hit_l3": int(l2_pf_miss_l2_hit_l3) if l2_pf_miss_l2_hit_l3 else 0,
            "l2_pf_miss_l2_l3": int(l2_pf_miss_l2_l3) if l2_pf_miss_l2_l3 else 0,
            "l3_request_miss": int(l3_request_miss) if l3_request_miss else 0,
            "xi_ccx_sdp_req1": int(xi_ccx_sdp_req1) if xi_ccx_sdp_req1 else 0,
            "mem_io_local_dmnd": int(mem_io_local_dmnd) if mem_io_local_dmnd else 0,
            "mem_io_remote_dmnd": int(mem_io_remote_dmnd) if mem_io_remote_dmnd else 0,
            "mem_io_remote_any": int(mem_io_remote_any) if mem_io_remote_any else 0,
            "mem_io_local_any": int(mem_io_local_any) if mem_io_local_any else 0,
            "time_elapsed":  int(time_elapsed) if time_elapsed else 0
            })

            return self.data
        except AttributeError as e:
            print(f"Failed to extract data: {e}")
            raise

    @classmethod
    def required_profiling_args(cls):
        """
        Define required arguments for profiling.

        Returns
        -------
        list
            List of required argument names.
        """
        return ["executable", "level"] 

    @classmethod
    def required_extract_args(cls, action):
        """
        Define required arguments for metric extraction.

        Parameters
        ----------
        action : str
            Profiling action ("extract_metrics" or "both").

        Returns
        -------
        list
            Required keys depending on mode.
        """        
        if action == "extract_metrics":
            return ["report_file"]
        else:
            return [] 
