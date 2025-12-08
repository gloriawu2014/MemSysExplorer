from profilers.FrontendInterface import FrontendInterface
import subprocess
import re
import os

class DrioProfilers(FrontendInterface):
    def __init__(self, **kwargs):
        """
        Initialize the profiler with user-specified configuration.

        Parameters
        ----------
        **kwargs : dict
            Dictionary that may contain:

            - executable : list
              A list including the executable and its arguments.
            - action : str
              One of "profiling", "extract_metrics", or "both".
            - config_file : str, optional
              Path to memcount configuration file.
            - enable_memory_stats : bool, optional
              Enable DynamoRIO memory statistics tracking.

        """

        super().__init__(**kwargs)
        self.executable_cmd = " ".join(self.config.get("executable", []))
        self.action = self.config.get("action")
        self.config_file = self.config.get("config")
        self.enable_memory_stats = self.config.get("enable_memory_stats", False)

        #Get the directory of this script 
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.client = os.path.join(base_dir, "build", "libmemcount.so")

        # Locate DynamoRIO installation dynamically
        dynamorio_install = os.path.join(base_dir, "dynamorio_install")
        dynamorio_dirs = [d for d in os.listdir(dynamorio_install) if d.startswith("DynamoRIO-")]

        if not dynamorio_dirs:
            raise FileNotFoundError("No DynamoRIO installation found in dynamorio_install/")

        self.dynamorio_home = os.path.join(dynamorio_install, dynamorio_dirs[0])
        self.run = os.path.join(self.dynamorio_home, "build", "bin64", "drrun")

        #Original Line
        #self.run = f"{os.path.expandvars('$DYNAMORIO_HOME')}/bin64/drrun" 
        #self.client = f"{os.path.expandvars('$APPS_HOME')}/profilers/dynamorio/build/libmemcount.so"
        
        self.output = ""
        self.report = None
        self.data = {}

    # validate if client has correct path 
    def validate_paths(self):
        """
        (Disabled) Validate the path of the executable, client, and drrun binary.

        This method is currently unused but can be enabled for strict validation.
        """

        # FIXME: rewrite this code
        # if not self.executable_cmd:
        #     raise ValueError("Executable path is required.")
        # executable = self.executable_cmd.split()[1]
        # if not os.path.isfile(executable) or not os.access(executable, os.X_OK):
        #     raise FileNotFoundError(f"'{self.executable_cmd}' is not valid or not executable.")
        
        # # check environmental variables
        # if not os.path.isfile(self.client):
        #     raise FileNotFoundError(f"{self.client} is not valid. Check $APPS_HOME environmental variable.")
        # if not os.path.isfile(self.run):
        #     raise FileNotFoundError(f"{self.run} is not valid. Check $DYNAMORIO_HOME environmental variable.")
        pass
    
    def constuct_command(self):
        """
        Construct the full DynamoRIO instrumentation command.

        If memory tracking is enabled, wraps the command with /usr/bin/time
        to capture peak memory usage of the instrumented process.

        Returns
        -------
        tuple
            - list of command components for subprocess.run()
            - str: report name (derived from executable)
        """
        executable_with_args = self.executable_cmd.split()
        report = os.path.basename(executable_with_args[0])

        drio_command = [
            self.run,
            "-c",
            self.client,
        ]

        # Add config file if specified
        if self.config_file:
            drio_command.extend(["-config", self.config_file])

        drio_command.append("--")
        drio_command.extend(executable_with_args)

        # Wrap with /usr/bin/time for memory tracking if enabled
        # Format: %M = maximum resident set size in KB
        # Format: %E = elapsed real time (wall clock)
        if self.enable_memory_stats:
            drio_command = ["/usr/bin/time", "-f", "TIMESTAT: Peak_Memory=%MKB Elapsed=%E"] + drio_command

        return drio_command, report

    # collect all the data that will be stored in a log file
    def profiling(self, **kwargs):
        """
        Run the profiler using the constructed DynamoRIO command.

        Captures stdout and stderr from the profiler. Stdout contains the
        memcount statistics, stderr contains DynamoRIO's memory tracking stats
        (if enabled). Stores output in a `.drio-rep` file if action is "profiling".

        Raises
        ------
        subprocess.CalledProcessError
            If the command execution fails.
        """
        # self.validate_paths()
        drio_command, report = self.constuct_command()
        try:
            print(f"Executing: {' '.join(drio_command)}")
            profiler_data = subprocess.run(drio_command, check=True, text=True,
                                          stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as e:
            print(f"Command failed with exit code {e.returncode}")
            raise
        self.output = profiler_data.stdout
        self.stderr_output = profiler_data.stderr

        # store output to file
        if self.action == "profiling":
            self.report = f"{report}.drio-rep"
            with open(self.report, 'w') as drio_report:
                drio_report.write(f"Profiling output:\n{self.output}")
                if self.stderr_output:
                    drio_report.write(f"\n\nDynamoRIO Stats:\n{self.stderr_output}")
            print(f"Output written to file {report}.drio-rep")

    def extract_metrics(self, report_file=None, **kwargs):
        """
        Extract memory access metrics from profiling output.

        Parameters
        ----------
        report_file : str, optional
            File to read from if `action == "extract_metrics"`.

        Returns
        -------
        dict
            Extracted metrics including:
            - read_freq
            - write_freq
            - total_reads
            - total_writes
            - workingset_size
            - peak_memory_usage (if memory stats enabled)

        Raises
        ------
        AttributeError
            If expected patterns are not found in the report.
        """
        toparse = ""
        stderr_output = ""
        if self.action == "extract_metrics":
            with open(report_file) as file:
                 content = file.read()
                 # Split content into stdout and stderr sections
                 if "DynamoRIO Stats:" in content:
                     parts = content.split("DynamoRIO Stats:")
                     toparse = parts[0]
                     stderr_output = parts[1] if len(parts) > 1 else ""
                 else:
                     toparse = content
        if self.action == "both":
            toparse = self.output
            stderr_output = getattr(self, 'stderr_output', '')

        print(toparse)

        # Extract the memory statistics
        try:
            memory_refs = re.search(r"saw (\d+) memory references", toparse).group(1)
            reads = re.search(r"number of reads: (\d+)", toparse).group(1)
            writes = re.search(r"number of writes: (\d+)", toparse).group(1)
            working_set_size = re.search(r"working set size: (\d+)", toparse).group(1)

            # Extract execution time (prefer microseconds for precision)
            execution_time_us_match = re.search(r"execution time \(us\): (\d+)", toparse)
            execution_time_ms_match = re.search(r"execution time \(ms\): ([\d.]+)", toparse)
            execution_time_s_match = re.search(r"execution time \(s\): ([\d.]+)", toparse)

            # Store execution time in microseconds (raw precision from DynamoRIO)
            execution_time_us = int(execution_time_us_match.group(1)) if execution_time_us_match else None
            execution_time_ms = float(execution_time_ms_match.group(1)) if execution_time_ms_match else None
            execution_time_s = float(execution_time_s_match.group(1)) if execution_time_s_match else None

            # Extract size-specific read counts
            read_size_1 = re.search(r"1-byte reads: (\d+)", toparse)
            read_size_2 = re.search(r"2-byte reads: (\d+)", toparse)
            read_size_4 = re.search(r"4-byte reads: (\d+)", toparse)
            read_size_8 = re.search(r"8-byte reads: (\d+)", toparse)
            read_size_16 = re.search(r"16-byte reads: (\d+)", toparse)
            read_size_32 = re.search(r"32-byte reads: (\d+)", toparse)
            read_size_64 = re.search(r"64-byte reads: (\d+)", toparse)
            read_size_other = re.search(r"other-size reads: (\d+)", toparse)

            # Extract size-specific write counts
            write_size_1 = re.search(r"1-byte writes: (\d+)", toparse)
            write_size_2 = re.search(r"2-byte writes: (\d+)", toparse)
            write_size_4 = re.search(r"4-byte writes: (\d+)", toparse)
            write_size_8 = re.search(r"8-byte writes: (\d+)", toparse)
            write_size_16 = re.search(r"16-byte writes: (\d+)", toparse)
            write_size_32 = re.search(r"32-byte writes: (\d+)", toparse)
            write_size_64 = re.search(r"64-byte writes: (\d+)", toparse)
            write_size_other = re.search(r"other-size writes: (\d+)", toparse)

            # Note: DynamoRIO memcount doesn't provide execution time, so we'll use a placeholder
            # or calculate frequency differently
            total_memory_refs = int(memory_refs)

            # Update the data dictionary with the extracted values
            self.data.update({
                "total_memory_refs": total_memory_refs,
                "total_reads": int(reads),
                "total_writes": int(writes),
                "workingset_size": int(working_set_size),
                "execution_time_us": execution_time_us,
                "execution_time_ms": execution_time_ms,
                "execution_time_s": execution_time_s,
                # Size-specific read counts
                "read_size_1": int(read_size_1.group(1)) if read_size_1 else 0,
                "read_size_2": int(read_size_2.group(1)) if read_size_2 else 0,
                "read_size_4": int(read_size_4.group(1)) if read_size_4 else 0,
                "read_size_8": int(read_size_8.group(1)) if read_size_8 else 0,
                "read_size_16": int(read_size_16.group(1)) if read_size_16 else 0,
                "read_size_32": int(read_size_32.group(1)) if read_size_32 else 0,
                "read_size_64": int(read_size_64.group(1)) if read_size_64 else 0,
                "read_size_other": int(read_size_other.group(1)) if read_size_other else 0,
                # Size-specific write counts
                "write_size_1": int(write_size_1.group(1)) if write_size_1 else 0,
                "write_size_2": int(write_size_2.group(1)) if write_size_2 else 0,
                "write_size_4": int(write_size_4.group(1)) if write_size_4 else 0,
                "write_size_8": int(write_size_8.group(1)) if write_size_8 else 0,
                "write_size_16": int(write_size_16.group(1)) if write_size_16 else 0,
                "write_size_32": int(write_size_32.group(1)) if write_size_32 else 0,
                "write_size_64": int(write_size_64.group(1)) if write_size_64 else 0,
                "write_size_other": int(write_size_other.group(1)) if write_size_other else 0,
            })

            # Extract memory statistics if available (from /usr/bin/time wrapper)
            if stderr_output and self.enable_memory_stats:
                try:
                    # Parse /usr/bin/time output format: "TIMESTAT: Peak_Memory=123456KB Elapsed=0:12.34"
                    peak_mem_match = re.search(r"TIMESTAT:.*Peak_Memory=(\d+)KB", stderr_output)
                    if peak_mem_match:
                        self.data["peak_memory_kb"] = int(peak_mem_match.group(1))

                    # Optionally extract wall clock time as well
                    elapsed_match = re.search(r"TIMESTAT:.*Elapsed=([\d:.]+)", stderr_output)
                    if elapsed_match:
                        self.data["wall_clock_time"] = elapsed_match.group(1)
                except (AttributeError, ValueError):
                    # Memory stats not found, not an error
                    pass

            return self.data
        except AttributeError as e:
            print(f"Failed to extract data: {e}")
            raise

    @classmethod
    def required_profiling_args(cls):
        """
        Declare required arguments for `profiling()`.

        Returns
        -------
        list
            Required argument names (["executable", "config"]).
        """
        return ["executable", "config"] 

    @classmethod
    def required_extract_args(cls, action):
        """
        Declare required arguments for `extract_metrics()`.

        Parameters
        ----------
        action : str
            The command-line action ("extract_metrics" or "both").

        Returns
        -------
        list
            Required argument names depending on the action.
        """
        if action == "extract_metrics":
            return ["report_file"]
        else:
            return [] 
