#!/usr/bin/env python3
"""
Makefile Parser for extracting build metadata

This module parses Makefiles to extract build configuration information
including targets, variables, dependencies, compiler settings, and version
requirements.

Usage:
    from makefile_parser import MakefileParser
    
    parser = MakefileParser()
    metadata = parser.parse_makefile('/path/to/Makefile')
    print(metadata)
"""

import os
import re
from typing import Dict, List, Optional, Any, Set
from pathlib import Path

class MakefileParser:
    """Parser for extracting metadata from Makefiles"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset parser state"""
        self.variables = {}
        self.targets = {}
        self.dependencies = {}
        self.phony_targets = set()
        self.includes = []
        self.version_info = {}
        self.compiler_info = {}
        self.paths = {}
        
    def parse_makefile(self, makefile_path: str) -> Dict[str, Any]:
        """Parse a Makefile and extract metadata
        
        Args:
            makefile_path: Path to the Makefile
            
        Returns:
            Dictionary containing extracted metadata
        """
        self.reset()
        
        if not os.path.exists(makefile_path):
            return {"error": f"Makefile not found: {makefile_path}"}
        
        try:
            with open(makefile_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except Exception as e:
            return {"error": f"Failed to read Makefile: {e}"}
        
        self._parse_content(content)
        
        return {
            "makefile_path": makefile_path,
            "variables": self.variables,
            "targets": self.targets,
            "dependencies": self.dependencies,
            "phony_targets": list(self.phony_targets),
            "includes": self.includes,
            "version_info": self.version_info,
            "compiler_info": self.compiler_info,
            "paths": self.paths,
            "summary": self._generate_summary()
        }
    
    def _parse_content(self, content: str):
        """Parse Makefile content"""
        lines = content.split('\n')
        current_target = None
        
        for i, line in enumerate(lines):
            line = line.strip()
            
            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue
            
            # Handle line continuations
            while line.endswith('\\') and i + 1 < len(lines):
                i += 1
                line = line[:-1] + ' ' + lines[i].strip()
            
            # Parse different line types
            if self._is_variable_assignment(line):
                self._parse_variable(line)
            elif self._is_target_definition(line):
                current_target = self._parse_target(line)
            elif self._is_phony_directive(line):
                self._parse_phony(line)
            elif self._is_include_directive(line):
                self._parse_include(line)
            elif current_target and line.startswith('\t'):
                self._parse_recipe(current_target, line[1:])
    
    def _is_variable_assignment(self, line: str) -> bool:
        """Check if line is a variable assignment"""
        return any(op in line for op in [':=', '=', '?=', '+='])
    
    def _is_target_definition(self, line: str) -> bool:
        """Check if line defines a target"""
        return ':' in line and not line.startswith('\t') and not self._is_variable_assignment(line)
    
    def _is_phony_directive(self, line: str) -> bool:
        """Check if line is a .PHONY directive"""
        return line.startswith('.PHONY:')
    
    def _is_include_directive(self, line: str) -> bool:
        """Check if line is an include directive"""
        return line.startswith('include ') or line.startswith('-include ')
    
    def _parse_variable(self, line: str):
        """Parse variable assignment"""
        for op in [':=', '?=', '+=', '=']:
            if op in line:
                name, value = line.split(op, 1)
                name = name.strip()
                value = value.strip()
                
                # Store with assignment operator info
                self.variables[name] = {
                    'value': value,
                    'operator': op
                }
                
                # Extract special metadata
                self._extract_special_variables(name, value)
                break
    
    def _extract_special_variables(self, name: str, value: str):
        """Extract special variables for metadata"""
        name_lower = name.lower()
        
        # Version information
        if 'version' in name_lower or 'ver' in name_lower:
            self.version_info[name] = value
        
        # Compiler information
        if any(compiler in name_lower for compiler in ['cc', 'cxx', 'gcc', 'clang', 'nvcc']):
            self.compiler_info[name] = value
        
        # Path information
        if any(path_word in name_lower for path_word in ['dir', 'path', 'home', 'install']):
            self.paths[name] = value
    
    def _parse_target(self, line: str) -> str:
        """Parse target definition"""
        parts = line.split(':', 1)
        target_name = parts[0].strip()
        
        dependencies = []
        if len(parts) > 1 and parts[1].strip():
            dependencies = [dep.strip() for dep in parts[1].split()]
        
        self.targets[target_name] = {
            'dependencies': dependencies,
            'recipes': []
        }
        
        self.dependencies[target_name] = dependencies
        return target_name
    
    def _parse_phony(self, line: str):
        """Parse .PHONY directive"""
        phony_targets = line.replace('.PHONY:', '').strip().split()
        self.phony_targets.update(phony_targets)
    
    def _parse_include(self, line: str):
        """Parse include directive"""
        include_file = line.split(None, 1)[1] if ' ' in line else ''
        if include_file:
            self.includes.append(include_file)
    
    def _parse_recipe(self, target: str, recipe: str):
        """Parse recipe line for a target"""
        if target in self.targets:
            self.targets[target]['recipes'].append(recipe)
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate summary statistics"""
        return {
            'total_variables': len(self.variables),
            'total_targets': len(self.targets),
            'phony_targets_count': len(self.phony_targets),
            'includes_count': len(self.includes),
            'has_version_info': len(self.version_info) > 0,
            'has_compiler_info': len(self.compiler_info) > 0,
            'main_targets': list(self.targets.keys())[:5]  # First 5 targets
        }

def parse_multiple_makefiles(directory: str, recursive: bool = True) -> Dict[str, Any]:
    """Parse multiple Makefiles in a directory
    
    Args:
        directory: Directory to search for Makefiles
        recursive: Whether to search recursively
        
    Returns:
        Dictionary mapping Makefile paths to their metadata
    """
    parser = MakefileParser()
    results = {}
    
    if recursive:
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.lower() in ['makefile', 'makefile.am', 'makefile.in'] or file.startswith('makefile'):
                    makefile_path = os.path.join(root, file)
                    results[makefile_path] = parser.parse_makefile(makefile_path)
    else:
        for file in os.listdir(directory):
            if file.lower() in ['makefile', 'makefile.am', 'makefile.in'] or file.startswith('makefile'):
                makefile_path = os.path.join(directory, file)
                if os.path.isfile(makefile_path):
                    results[makefile_path] = parser.parse_makefile(makefile_path)
    
    return results

def get_profiler_build_metadata(profilers_dir: str) -> Dict[str, Any]:
    """Get build metadata for all profilers
    
    Args:
        profilers_dir: Path to profilers directory
        
    Returns:
        Dictionary with build metadata for each profiler
    """
    profiler_metadata = {}
    
    # Get immediate subdirectories (profiler directories)
    if not os.path.exists(profilers_dir):
        return {"error": f"Profilers directory not found: {profilers_dir}"}
    
    for item in os.listdir(profilers_dir):
        profiler_path = os.path.join(profilers_dir, item)
        if os.path.isdir(profiler_path):
            # Look for Makefiles in this profiler directory
            makefiles = {}
            
            # Check for main Makefile
            main_makefile = os.path.join(profiler_path, 'Makefile')
            if os.path.exists(main_makefile):
                parser = MakefileParser()
                makefiles['main'] = parser.parse_makefile(main_makefile)
            
            # Look for other common Makefiles
            for makefile_name in ['makefile', 'Makefile.am', 'Makefile.in']:
                makefile_path = os.path.join(profiler_path, makefile_name)
                if os.path.exists(makefile_path) and makefile_path != main_makefile:
                    parser = MakefileParser()
                    makefiles[makefile_name] = parser.parse_makefile(makefile_path)
            
            if makefiles:
                profiler_metadata[item] = {
                    'profiler_name': item,
                    'profiler_path': profiler_path,
                    'makefiles': makefiles,
                    'build_summary': _summarize_profiler_build(makefiles)
                }
    
    return profiler_metadata

def get_single_profiler_build_metadata(profiler_dir: str) -> Dict[str, Any]:
    """Get build metadata for a single profiler directory
    
    Args:
        profiler_dir: Path to the specific profiler directory
        
    Returns:
        Dictionary with build metadata for the profiler
    """
    if not os.path.exists(profiler_dir):
        return {"error": f"Profiler directory not found: {profiler_dir}"}
    
    profiler_name = os.path.basename(profiler_dir)
    makefiles = {}
    
    # Check for main Makefile
    main_makefile = os.path.join(profiler_dir, 'Makefile')
    if os.path.exists(main_makefile):
        parser = MakefileParser()
        makefiles['main'] = parser.parse_makefile(main_makefile)
    
    # Look for other common Makefiles
    for makefile_name in ['makefile', 'Makefile.am', 'Makefile.in']:
        makefile_path = os.path.join(profiler_dir, makefile_name)
        if os.path.exists(makefile_path) and makefile_path != main_makefile:
            parser = MakefileParser()
            makefiles[makefile_name] = parser.parse_makefile(makefile_path)
    
    if not makefiles:
        return {
            "profiler_name": profiler_name,
            "profiler_path": profiler_dir,
            "message": "No Makefiles found in this profiler directory"
        }
    
    return {
        'profiler_name': profiler_name,
        'profiler_path': profiler_dir,
        'makefiles': makefiles,
        'build_summary': _summarize_profiler_build(makefiles)
    }

def _summarize_profiler_build(makefiles: Dict[str, Any]) -> Dict[str, Any]:
    """Summarize build information for a profiler"""
    summary = {
        'total_makefiles': len(makefiles),
        'version_requirements': {},
        'compiler_requirements': {},
        'main_targets': [],
        'dependencies': []
    }
    
    for makefile_name, metadata in makefiles.items():
        if 'error' in metadata:
            continue
        
        # Collect version info
        if 'version_info' in metadata:
            summary['version_requirements'].update(metadata['version_info'])
        
        # Collect compiler info
        if 'compiler_info' in metadata:
            summary['compiler_requirements'].update(metadata['compiler_info'])
        
        # Collect main targets
        if 'targets' in metadata:
            summary['main_targets'].extend(metadata['targets'].keys())
        
        # Collect phony targets (these are usually the main build targets)
        if 'phony_targets' in metadata:
            summary['dependencies'].extend(metadata['phony_targets'])
    
    # Remove duplicates
    summary['main_targets'] = list(set(summary['main_targets']))
    summary['dependencies'] = list(set(summary['dependencies']))
    
    return summary

def main():
    """Test the Makefile parser"""
    print("Makefile Parser Test")
    print("===================")
    
    # Test with profilers directory
    profilers_dir = os.path.join(os.path.dirname(__file__), '..', 'profilers')
    
    if os.path.exists(profilers_dir):
        print(f"Parsing Makefiles in: {profilers_dir}")
        metadata = get_profiler_build_metadata(profilers_dir)
        
        for profiler_name, data in metadata.items():
            print(f"\n{profiler_name}:")
            print(f"  Path: {data.get('profiler_path', 'N/A')}")
            print(f"  Makefiles: {data.get('build_summary', {}).get('total_makefiles', 0)}")
            
            version_reqs = data.get('build_summary', {}).get('version_requirements', {})
            if version_reqs:
                print(f"  Version Requirements: {version_reqs}")
            
            compiler_reqs = data.get('build_summary', {}).get('compiler_requirements', {})
            if compiler_reqs:
                print(f"  Compiler Requirements: {compiler_reqs}")
            
            targets = data.get('build_summary', {}).get('main_targets', [])
            if targets:
                print(f"  Main Targets: {targets[:5]}")  # Show first 5
    else:
        print(f"Profilers directory not found: {profilers_dir}")

if __name__ == "__main__":
    main()