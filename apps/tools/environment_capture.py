#!/usr/bin/env python3
"""
Python wrapper for the environment capture C library

This module provides Python access to the standalone environment capture
library that can be used by BaseMetadata.py and other profiling tools.

Usage:
    from environment_capture import EnvironmentCapture
    
    env = EnvironmentCapture()
    print(f"OS: {env.os_name}")
    print(f"User: {env.get_variable('USER')}")
    print(f"All vars: {env.get_all_variables()}")
"""

import ctypes
import os
import sys
from typing import Dict, Optional, Any

class EnvironmentCapture:
    """Python wrapper for environment capture C library"""
    
    def __init__(self, lib_path: Optional[str] = None):
        """Initialize the environment capture
        
        Args:
            lib_path: Optional path to libprofiler_common.so
                     If None, will search in common build directory
        """
        self._lib = None
        self._env_ptr = None
        
        # Try to load the library
        if not lib_path:
            # Default path relative to this file
            script_dir = os.path.dirname(os.path.abspath(__file__))
            lib_path = os.path.join(script_dir, '../profilers/common/build/libprofiler_common.a')
            
            # For shared library, try .so extension
            so_path = lib_path.replace('.a', '.so')
            if os.path.exists(so_path):
                lib_path = so_path
        
        try:
            self._lib = ctypes.CDLL(lib_path)
        except OSError as e:
            # Fallback: try to build a minimal shared library
            self._build_shared_library()
            if not self._lib:
                raise RuntimeError(f"Could not load environment capture library: {e}")
        
        self._setup_function_signatures()
        self._capture_environment()
    
    def _build_shared_library(self):
        """Build a shared library version for Python to use"""
        import subprocess
        import tempfile
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        src_dir = os.path.join(script_dir, '../profilers/common/src')
        inc_dir = os.path.join(script_dir, '../profilers/common/include')
        
        src_file = os.path.join(src_dir, 'environment_capture.c')
        
        if not os.path.exists(src_file):
            return
            
        try:
            # Create temporary shared library
            with tempfile.NamedTemporaryFile(suffix='.so', delete=False) as tmp:
                cmd = [
                    'gcc', '-shared', '-fPIC', '-o', tmp.name,
                    f'-I{inc_dir}', src_file
                ]
                subprocess.run(cmd, check=True, capture_output=True)
                self._lib = ctypes.CDLL(tmp.name)
                self._temp_lib_path = tmp.name
        except (subprocess.CalledProcessError, OSError):
            pass
    
    def _setup_function_signatures(self):
        """Setup ctypes function signatures"""
        if not self._lib:
            return
            
        # Define structure for system_environment_t
        class SystemEnvironment(ctypes.Structure):
            _fields_ = [
                ('hostname', ctypes.c_char_p),
                ('os_name', ctypes.c_char_p), 
                ('os_version', ctypes.c_char_p),
                ('architecture', ctypes.c_char_p),
                ('working_directory', ctypes.c_char_p),
                ('env_names', ctypes.POINTER(ctypes.c_char_p)),
                ('env_values', ctypes.POINTER(ctypes.c_char_p)),
                ('env_count', ctypes.c_size_t)
            ]
        
        # Function signatures
        self._lib.environment_capture_create.restype = ctypes.POINTER(SystemEnvironment)
        self._lib.environment_capture_create.argtypes = []
        
        self._lib.environment_capture_destroy.restype = None
        self._lib.environment_capture_destroy.argtypes = [ctypes.POINTER(SystemEnvironment)]
        
        self._lib.environment_capture_get_var.restype = ctypes.c_char_p
        self._lib.environment_capture_get_var.argtypes = [ctypes.POINTER(SystemEnvironment), ctypes.c_char_p]
        
        self._lib.environment_capture_timestamp_ns.restype = ctypes.c_uint64
        self._lib.environment_capture_timestamp_ns.argtypes = []
        
        self._lib.environment_capture_process_id.restype = ctypes.c_uint32
        self._lib.environment_capture_process_id.argtypes = []
        
        self._SystemEnvironment = SystemEnvironment
    
    def _capture_environment(self):
        """Capture the current environment"""
        if not self._lib:
            # Fallback to pure Python implementation
            self._use_python_fallback()
            return
            
        self._env_ptr = self._lib.environment_capture_create()
        if not self._env_ptr:
            self._use_python_fallback()
    
    def _use_python_fallback(self):
        """Fallback to pure Python environment capture"""
        import platform
        import time
        import socket
        
        self._fallback_data = {
            'hostname': socket.gethostname(),
            'os_name': platform.system(),
            'os_version': platform.release(),
            'architecture': platform.machine(),
            'working_directory': os.getcwd(),
            'environment_variables': dict(os.environ),
            'timestamp_ns': int(time.time() * 1_000_000_000),
            'process_id': os.getpid()
        }
    
    @property
    def hostname(self) -> str:
        """Get system hostname"""
        if self._env_ptr:
            return self._env_ptr.contents.hostname.decode() if self._env_ptr.contents.hostname else 'unknown'
        return self._fallback_data.get('hostname', 'unknown')
    
    @property 
    def os_name(self) -> str:
        """Get operating system name"""
        if self._env_ptr:
            return self._env_ptr.contents.os_name.decode() if self._env_ptr.contents.os_name else 'unknown'
        return self._fallback_data.get('os_name', 'unknown')
    
    @property
    def os_version(self) -> str:
        """Get operating system version"""
        if self._env_ptr:
            return self._env_ptr.contents.os_version.decode() if self._env_ptr.contents.os_version else 'unknown'
        return self._fallback_data.get('os_version', 'unknown')
    
    @property
    def architecture(self) -> str:
        """Get system architecture"""
        if self._env_ptr:
            return self._env_ptr.contents.architecture.decode() if self._env_ptr.contents.architecture else 'unknown'
        return self._fallback_data.get('architecture', 'unknown')
    
    @property
    def working_directory(self) -> str:
        """Get working directory"""
        if self._env_ptr:
            return self._env_ptr.contents.working_directory.decode() if self._env_ptr.contents.working_directory else 'unknown'
        return self._fallback_data.get('working_directory', 'unknown')
    
    @property
    def process_id(self) -> int:
        """Get current process ID"""
        if self._lib:
            return self._lib.environment_capture_process_id()
        return self._fallback_data.get('process_id', os.getpid())
    
    @property
    def timestamp_ns(self) -> int:
        """Get current timestamp in nanoseconds"""
        if self._lib:
            return self._lib.environment_capture_timestamp_ns()
        return self._fallback_data.get('timestamp_ns', 0)
    
    def get_variable(self, name: str) -> Optional[str]:
        """Get specific environment variable value
        
        Args:
            name: Environment variable name
            
        Returns:
            Variable value or None if not found
        """
        if self._env_ptr and self._lib:
            result = self._lib.environment_capture_get_var(self._env_ptr, name.encode())
            return result.decode() if result else None
        return self._fallback_data.get('environment_variables', {}).get(name)
    
    def get_all_variables(self) -> Dict[str, str]:
        """Get all environment variables as dictionary
        
        Returns:
            Dictionary of environment variable name -> value
        """
        if self._env_ptr:
            env_dict = {}
            env_count = self._env_ptr.contents.env_count
            
            for i in range(env_count):
                name_ptr = self._env_ptr.contents.env_names[i]
                value_ptr = self._env_ptr.contents.env_values[i]
                
                if name_ptr and value_ptr:
                    name = name_ptr.decode()
                    value = value_ptr.decode()
                    env_dict[name] = value
            
            return env_dict
        
        return self._fallback_data.get('environment_variables', {})
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert environment capture to dictionary
        
        Returns:
            Dictionary with all environment information
        """
        return {
            'hostname': self.hostname,
            'os_name': self.os_name,
            'os_version': self.os_version,
            'architecture': self.architecture,
            'working_directory': self.working_directory,
            'process_id': self.process_id,
            'timestamp_ns': self.timestamp_ns,
            'environment_variables': self.get_all_variables()
        }
    
    def __del__(self):
        """Cleanup resources"""
        if self._env_ptr and self._lib:
            self._lib.environment_capture_destroy(self._env_ptr)
        
        # Clean up temporary library if created
        if hasattr(self, '_temp_lib_path'):
            try:
                os.unlink(self._temp_lib_path)
            except OSError:
                pass


def main():
    """Test the environment capture module"""
    print("Environment Capture Python Wrapper Test")
    print("========================================")
    
    try:
        env = EnvironmentCapture()
        
        print(f"Hostname: {env.hostname}")
        print(f"OS: {env.os_name} {env.os_version}")
        print(f"Architecture: {env.architecture}")
        print(f"Working Directory: {env.working_directory}")
        print(f"Process ID: {env.process_id}")
        print(f"Timestamp: {env.timestamp_ns}")
        
        print(f"\nAll Environment Variables:")
        all_vars = env.get_all_variables()
        for name, value in sorted(all_vars.items()):
            display_value = value[:80] + '...' if value and len(value) > 80 else value
            print(f"  {name}: {display_value}")
        
        print(f"\nTotal Environment Variables: {len(all_vars)}")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
