#!/usr/bin/env python3
"""
Profiler Flag Parser for extracting profiler-specific command flags

This module analyzes profiler Python files to extract the specific flags
and command-line arguments used by each profiler during execution.

Usage:
    from profiler_flag_parser import ProfilerFlagParser
    
    parser = ProfilerFlagParser()
    flags = parser.extract_profiler_flags('/path/to/profiler_dir')
    print(flags)
"""

import os
import re
import ast
import inspect
from typing import Dict, List, Optional, Any

class ProfilerFlagParser:
    """Parser for extracting profiler-specific command flags from source code"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset parser state"""
        self.flags = {}
        self.commands = {}
        self.configuration_flags = {}
        self.environment_variables = {}
        
    def extract_flags(self, profiler_dir: str) -> Dict[str, Any]:
        """Extract flags from a specific profiler directory
        
        Args:
            profiler_dir: Path to the profiler directory
            
        Returns:
            Dictionary containing extracted flag information
        """
        self.reset()
        
        if not os.path.exists(profiler_dir):
            return {"error": f"Profiler directory not found: {profiler_dir}"}
        
        profiler_name = os.path.basename(profiler_dir)
        
        # Look for the main profiler file - try multiple naming patterns
        possible_files = [
            f"{profiler_name}_profilers.py",
            # Handle special cases
            "drio_profilers.py" if profiler_name == "dynamorio" else None,
            "ncu_profilers.py" if profiler_name == "ncu" else None,
        ]
        
        profiler_file = None
        for filename in possible_files:
            if filename:
                candidate = os.path.join(profiler_dir, filename)
                if os.path.exists(candidate):
                    profiler_file = candidate
                    break
        
        if not profiler_file:
            # Try to find any *profiler*.py file in the directory
            for file in os.listdir(profiler_dir):
                if file.endswith('_profilers.py') or 'profiler' in file.lower():
                    profiler_file = os.path.join(profiler_dir, file)
                    break
        
        if not profiler_file:
            return {
                "profiler_name": profiler_name,
                "profiler_dir": profiler_dir,
                "message": "No profiler file found",
                "attempted_files": [f for f in possible_files if f is not None]
            }
        
        try:
            self._analyze_file(profiler_file)
            
            return {
                "profiler_name": profiler_name,
                "profiler_dir": profiler_dir,
                "profiler_file": profiler_file,
                "command_flags": self.flags,
                "base_commands": self.commands,
                "configuration_options": self.configuration_flags,
                "environment_variables": self.environment_variables,
                "summary": self._generate_summary()
            }
        except Exception as e:
            return {
                "profiler_name": profiler_name,
                "profiler_dir": profiler_dir,
                "error": f"Failed to parse profiler file: {str(e)}"
            }
    
    def _analyze_file(self, profiler_file: str):
        """Analyze a profiler Python file to extract flags"""
        with open(profiler_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Parse the AST to find command construction
        try:
            tree = ast.parse(content)
            self._analyze_ast(tree)
        except:
            pass  # Fall back to regex analysis
        
        # Also do regex-based analysis for additional patterns
        self._analyze_with_regex(content)
        
        # Extract profiler-specific patterns
        self._extract_specific_patterns(content)
    
    def _analyze_ast(self, tree: ast.AST):
        """Analyze AST to find command construction patterns"""
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if node.name in ['construct_command', 'profiling', '__init__']:
                    self._analyze_function_node(node)
            elif isinstance(node, ast.Assign):
                self._analyze_assignment_node(node)
    
    def _analyze_function_node(self, node: ast.FunctionDef):
        """Analyze a function node for command construction"""
        for child in ast.walk(node):
            if isinstance(child, ast.List):
                # Look for command list construction
                self._extract_command_from_list(child, node.name)
            elif isinstance(child, ast.Assign):
                self._analyze_assignment_in_function(child, node.name)
    
    def _extract_command_from_list(self, list_node: ast.List, context: str):
        """Extract command flags from a list literal"""
        command_parts = []
        for element in list_node.elts:
            if isinstance(element, ast.Constant):
                command_parts.append(element.value)
            elif isinstance(element, ast.Str):  # Python < 3.8
                command_parts.append(element.s)
        
        if command_parts and any(self._looks_like_command_flag(part) for part in command_parts):
            if context not in self.flags:
                self.flags[context] = []
            
            # Extract base command and flags
            base_cmd = None
            flags = []
            
            for i, part in enumerate(command_parts):
                if isinstance(part, str):
                    if i == 0 or not part.startswith('-'):
                        if not base_cmd and self._looks_like_base_command(part):
                            base_cmd = part
                            self.commands[context] = base_cmd
                    else:
                        flags.append(part)
                        # If next element exists and doesn't start with -, it's a value
                        if i + 1 < len(command_parts) and isinstance(command_parts[i + 1], str):
                            if not command_parts[i + 1].startswith('-'):
                                flags.append(command_parts[i + 1])
            
            if flags:
                self.flags[context].extend(flags)
    
    def _analyze_assignment_node(self, node: ast.Assign):
        """Analyze assignment nodes for environment variables and configs"""
        if len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Subscript) and isinstance(target.value, ast.Attribute):
                # os.environ["VAR"] = value
                if (hasattr(target.value, 'attr') and target.value.attr == 'environ' and
                    isinstance(target.slice, ast.Constant)):
                    env_var = target.slice.value
                    if isinstance(node.value, ast.Constant):
                        self.environment_variables[env_var] = node.value.value
    
    def _analyze_assignment_in_function(self, node: ast.Assign, function_name: str):
        """Analyze assignments within functions for configuration"""
        for target in node.targets:
            if isinstance(target, ast.Name):
                var_name = target.id
                if 'command' in var_name.lower() or 'cmd' in var_name.lower():
                    if isinstance(node.value, ast.List):
                        self._extract_command_from_list(node.value, function_name)
    
    def _analyze_with_regex(self, content: str):
        """Use regex patterns to find additional flag patterns"""
        # Pattern for command lists
        command_patterns = [
            r'(\w+_command|command)\s*=\s*\[(.*?)\]',
            r'(perf|ncu|sniper)\s+[^"\']*?(["\'][^"\']*["\'])',
            r'--\w+[^,\]]*',
            r'-\w+[^,\]]*'
        ]
        
        for pattern in command_patterns:
            matches = re.finditer(pattern, content, re.MULTILINE | re.DOTALL)
            for match in matches:
                self._process_regex_match(match, content)
        
        # Pattern for environment variables
        env_pattern = r'os\.environ\[["\'](\w+)["\']\]\s*=\s*["\']([^"\']*)["\']'
        env_matches = re.finditer(env_pattern, content)
        for match in env_matches:
            self.environment_variables[match.group(1)] = match.group(2)
        
        # Pattern for configuration options
        config_patterns = [
            r'self\.(\w+)\s*=\s*["\']([^"\']*)["\']',
            r'(\w+)\s*=\s*.*?\.get\(["\']([^"\']*)["\']',
        ]
        
        for pattern in config_patterns:
            matches = re.finditer(pattern, content)
            for match in matches:
                if len(match.groups()) >= 2:
                    self.configuration_flags[match.group(1)] = match.group(2)
    
    def _process_regex_match(self, match, content):
        """Process regex match to extract flags"""
        matched_text = match.group(0)
        
        # Extract flag-like patterns
        flag_patterns = re.findall(r'["\']([^"\']*)["\']', matched_text)
        
        for flag in flag_patterns:
            if self._looks_like_command_flag(flag):
                if 'regex_flags' not in self.flags:
                    self.flags['regex_flags'] = []
                if flag not in self.flags['regex_flags']:
                    self.flags['regex_flags'].append(flag)
    
    def _looks_like_command_flag(self, text: str) -> bool:
        """Check if text looks like a command-line flag"""
        if not isinstance(text, str):
            return False
        return (text.startswith('-') or 
                text in ['ncu', 'perf', 'stat', 'application', 'base', 'all'] or
                '=' in text or
                text.endswith('.so') or
                text.endswith('.conf'))
    
    def _looks_like_base_command(self, text: str) -> bool:
        """Check if text looks like a base command"""
        if not isinstance(text, str):
            return False
        return text in ['ncu', 'perf', 'drrun', 'sniper']
    
    def _extract_specific_patterns(self, content: str):
        """Extract profiler-specific command patterns"""
        # DynamoRIO patterns
        if 'drrun' in content or 'DynamoRIO' in content:
            self._extract_drio_flags(content)
        
        # NCU patterns
        if 'ncu' in content:
            self._extract_ncu_flags(content)
        
        # Perf patterns
        if 'perf stat' in content:
            self._extract_perf_flags(content)
        
        # Sniper patterns
        if 'sniper' in content.lower():
            self._extract_sniper_flags(content)
    
    def _extract_drio_flags(self, content: str):
        """Extract DynamoRIO specific flags"""
        # Look for drrun command construction
        drio_patterns = [
            r'self\.run[,\s]',
            r'"-c"[,\s]',
            r'self\.client[,\s]',
            r'"-config"[,\s]',
            r'"--"'
        ]
        
        for pattern in drio_patterns:
            if re.search(pattern, content):
                if 'drio_flags' not in self.flags:
                    self.flags['drio_flags'] = []
                
                # Extract the flag
                if '"-c"' in pattern:
                    self.flags['drio_flags'].append('-c')
                elif '"-config"' in pattern:
                    self.flags['drio_flags'].append('-config')
                elif '"--"' in pattern:
                    self.flags['drio_flags'].append('--')
        
        # Extract base command
        if 'drrun' in content:
            self.commands['drio'] = 'drrun'
    
    def _extract_ncu_flags(self, content: str):
        """Extract NCU specific flags"""
        ncu_patterns = [
            r'"ncu"',
            r'"-f"',
            r'"--replay-mode"',
            r'"--section-folder"',
            r'"--launch-count"',
            r'"--cache-control"',
            r'"--clock-control"'
        ]
        
        for pattern in ncu_patterns:
            matches = re.findall(pattern, content)
            for match in matches:
                clean_flag = match.strip('"')
                if 'ncu_flags' not in self.flags:
                    self.flags['ncu_flags'] = []
                if clean_flag not in self.flags['ncu_flags']:
                    self.flags['ncu_flags'].append(clean_flag)
        
        if 'ncu' in content:
            self.commands['ncu'] = 'ncu'
    
    def _extract_sniper_flags(self, content: str):
        """Extract Sniper specific flags"""
        sniper_patterns = [
            r'self\.sniper_path',
            r'"-c"',
            r'self\.config_path',
            r'"--"'
        ]
        
        for pattern in sniper_patterns:
            if re.search(pattern, content):
                if 'sniper_flags' not in self.flags:
                    self.flags['sniper_flags'] = []
                
                # Extract the flag
                if '"-c"' in pattern:
                    self.flags['sniper_flags'].append('-c')
                elif '"--"' in pattern:
                    self.flags['sniper_flags'].append('--')
        
        # Extract base command
        if 'sniper' in content.lower():
            self.commands['sniper'] = 'sniper'
    
    def _extract_perf_flags(self, content: str):
        """Extract Perf specific flags"""
        # Look for perf stat command
        perf_match = re.search(r'"perf",\s*"stat",\s*"-e",\s*"([^"]+)"', content)
        if perf_match:
            events = perf_match.group(1)
            if 'perf_flags' not in self.flags:
                self.flags['perf_flags'] = []
            self.flags['perf_flags'].extend(['perf', 'stat', '-e', events])
            self.commands['perf'] = 'perf'
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate summary of extracted flags"""
        total_flags = sum(len(flags) if isinstance(flags, list) else 1 
                         for flags in self.flags.values())
        
        return {
            'total_flag_groups': len(self.flags),
            'total_flags': total_flags,
            'base_commands': list(self.commands.values()),
            'environment_variables_count': len(self.environment_variables),
            'configuration_options_count': len(self.configuration_flags),
            'flag_groups': list(self.flags.keys())
        }

def extract_all_flags(profilers_dir: str) -> Dict[str, Any]:
    """Extract flags from all profilers in a directory
    
    Args:
        profilers_dir: Path to profilers directory
        
    Returns:
        Dictionary mapping profiler names to their flag data
    """
    parser = ProfilerFlagParser()
    results = {}
    
    if not os.path.exists(profilers_dir):
        return {"error": f"Profilers directory not found: {profilers_dir}"}
    
    for item in os.listdir(profilers_dir):
        profiler_path = os.path.join(profilers_dir, item)
        if os.path.isdir(profiler_path) and not item.startswith('.'):
            results[item] = parser.extract_flags(profiler_path)
    
    return results

def main():
    """Test the profiler flag parser"""
    print("Profiler Flag Parser Test")
    print("========================")
    
    # Test with profilers directory
    profilers_dir = os.path.join(os.path.dirname(__file__), '..', 'profilers')
    
    if os.path.exists(profilers_dir):
        print(f"Parsing profiler flags in: {profilers_dir}")
        results = extract_all_flags(profilers_dir)
        
        for profiler_name, data in results.items():
            if 'error' not in data:
                print(f"\n{profiler_name}:")
                print(f"  Base commands: {data.get('base_commands', {})}")
                
                flags = data.get('command_flags', {})
                for context, flag_list in flags.items():
                    if flag_list:
                        print(f"  {context} flags: {flag_list}")
                
                env_vars = data.get('environment_variables', {})
                if env_vars:
                    print(f"  Environment: {env_vars}")
                
                summary = data.get('summary', {})
                print(f"  Summary: {summary.get('total_flags', 0)} flags, {summary.get('total_flag_groups', 0)} groups")
            else:
                print(f"\n{profiler_name}: {data.get('error', 'Unknown error')}")
    else:
        print(f"Profilers directory not found: {profilers_dir}")

if __name__ == "__main__":
    main()