"""
Frontend Interface for Profiler Integration in MemSysExplorer.

This abstract base class defines the core interface that all profilers
must implement. It also includes class-level registries for available
profilers and associated metadata classes.

Subclasses must implement the `profiling`, `extract_metrics`, and `__init__`
methods and can optionally override `required_profiling_args` and 
`required_extract_args` to declare required CLI arguments.

Registry Features
-----------------
- Profiler classes can be registered using `register_profiler`.
- Metadata classes can be registered using `register_metadata`.
- Instantiation and lookup of registered components is supported.

This interface enables modular extension of new profilers into the system
without changing the core driver logic.
"""

from abc import ABC, abstractmethod

class FrontendInterface(ABC):
    registered_profilers = {}
    registered_metadata = {}

    @abstractmethod
    def __init__(self, **kwargs):
        """
        Initialize the profiler instance with user-defined configuration.

        Parameters
        ----------
        **kwargs : dict
            Arbitrary configuration parameters to initialize the profiler,
            typically passed from command-line argument parsing.
        """        

        self.config = kwargs

    @abstractmethod
    def profiling(self, **kwargs):
        """
        Execute the profiling pass for the target application.

        This method should launch the profiler on the specified executable or
        workload and prepare internal data structures for postprocessing.

        Parameters
        ----------
        **kwargs : dict
            Arguments required by the specific profiler implementation,
            such as path to executable, arguments, or number of cores.
        """

        pass

    @abstractmethod
    def extract_metrics(self, **kwargs):
        """
        Extract and return structured profiling metrics.

        This function is invoked after profiling is complete and is expected
        to return a dictionary (or list of dictionaries) with parsed metrics.

        Parameters
        ----------
        **kwargs : dict
            Arguments specific to metric extraction, like selected cache level.

        Returns
        -------
        dict or list of dict
            Structured metrics extracted from raw profiler output.
        """

        pass
    @classmethod
    def required_profiling_args(cls):
        """
        Return the list of required arguments for profiling phase.

        Subclasses may override this to return profiler-specific required
        arguments (e.g., executable path, number of threads).

        Returns
        -------
        list
            Names of required arguments for the `profiling` method.
        """        

        return []

    @classmethod
    def required_extract_args(cls):
        """
        Return the list of required arguments for metric extraction phase.

        Subclasses may override this to return specific arguments for
        controlling the extraction process (e.g., 'level').

        Returns
        -------
        list
            Names of required arguments for the `extract_metrics` method.
        """

        return []

    @classmethod
    def optional_profiling_args(cls):
        """
        Return a list of optional argument definitions for profiling phase.

        Subclasses may override this to return profiler-specific optional
        arguments (e.g., ROI control, instruction counts, sampling options).

        Each argument definition is a dictionary with keys:
        - name: str - argument name (without --)
        - type: type - argument type (int, str, etc.), optional
        - default: any - default value, optional
        - choices: list - valid choices, optional
        - action: str - argparse action ('store_true', etc.), optional
        - help: str - help text

        Returns
        -------
        list of dict
            List of optional argument definitions for argparse.
        """
        return []

    @classmethod
    def register_profiler(cls, name, profiler_class):
        """
        Register a profiler class under a given name.

        Parameters
        ----------
        name : str
            Identifier used to refer to the profiler (e.g., 'ncu', 'perf').
        profiler_class : type
            Class implementing the FrontendInterface.
        """

        cls.registered_profilers[name] = profiler_class

    @classmethod
    def create_profiler(cls, name, **kwargs):
        """
        Instantiate a profiler from the registry using the given name.

        Parameters
        ----------
        name : str
            Name of the registered profiler.
        **kwargs : dict
            Initialization parameters passed to the profiler class.

        Returns
        -------
        FrontendInterface
            An instance of the requested profiler.

        Raises
        ------
        ValueError
            If the profiler name is not registered.
        """        

        if name not in cls.registered_profilers:
            raise ValueError(f"Profiler '{name}' is not registered.")
        return cls.registered_profilers[name](**kwargs)

    @classmethod
    def register_metadata(cls, name, metadata_class):
        """
        Register a metadata class associated with a profiler.

        Parameters
        ----------
        name : str
            Name of the profiler the metadata class is associated with.
        metadata_class : type
            Class implementing system metadata collection logic.
        """        

        cls.registered_metadata[name] = metadata_class

    @classmethod
    def get_metadata_class(cls, name):
        """
        Get the registered metadata class for a specific profiler.

        Parameters
        ----------
        name : str
            Profiler name for which to retrieve metadata class.

        Returns
        -------
        type or None
            The metadata class if registered, otherwise None.
        """        

        return cls.registered_metadata.get(name)


