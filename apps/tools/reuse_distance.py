#!/usr/bin/env python3
"""
Reuse Distance Calculator for DynamoRIO Memory Traces

Calculates reuse distance for each unique address from DynamoRIO trace files.
Reuse distance = number of unique addresses accessed between consecutive accesses to the same address.

Supports both CSV and legacy trace formats:
- CSV: timestamp,addr,op,size
- Legacy: timestamp address operation size

Usage: python3 reuse_distance.py <input_trace> [--output file] [--window-size N]
"""

import argparse
import sys
import os
from collections import defaultdict
from typing import Dict, List, Optional, Set


class ReuseDistanceTracker:
    """
    Reuse distance tracker with configurable memory window.
    
    - window_size = -1: Track all history (unlimited memory)
    - window_size > 0: Use circular buffer with specified size
    """
    
    def __init__(self, window_size: int = -1):
        """
        Initialize the tracker.
        
        Parameters
        ----------
        window_size : int
            -1 for unlimited history, positive number for windowed tracking
        """
        self.window_size = window_size
        self.unlimited_mode = (window_size == -1)
        
        # Global access tracking
        self.global_access_index = 0
        self.last_access_index: Dict[int, int] = {}  # address -> last access index
        
        # Results storage: address -> [list of reuse distances]
        self.reuse_distances: Dict[int, List[int]] = defaultdict(list)
        
        if self.unlimited_mode:
            # Unlimited history mode
            self.access_history: Dict[int, int] = {}  # index -> address
        else:
            # Windowed mode
            self.recent_accesses = [None] * window_size
            self.buffer_start_index = 0
    
    def process_access(self, address: int):
        """
        Process a memory access and calculate reuse distance if applicable.
        
        Parameters
        ----------
        address : int
            Memory address being accessed
        """
        if address in self.last_access_index:
            # This is a reuse - calculate reuse distance
            last_index = self.last_access_index[address]
            reuse_distance = self._calculate_reuse_distance(address, last_index)
            if reuse_distance is not None:
                self.reuse_distances[address].append(reuse_distance)
        
        # Update tracking structures
        self._update_tracking(address)
        self.last_access_index[address] = self.global_access_index
        self.global_access_index += 1
    
    def _calculate_reuse_distance(self, address: int, last_index: int) -> Optional[int]:
        """Calculate reuse distance between last_index and current access."""
        if self.unlimited_mode:
            return self._calculate_unlimited(last_index)
        else:
            return self._calculate_windowed(last_index)
    
    def _calculate_unlimited(self, last_index: int) -> int:
        """Calculate reuse distance with unlimited history."""
        unique_addresses: Set[int] = set()
        for i in range(last_index + 1, self.global_access_index):
            if i in self.access_history:
                unique_addresses.add(self.access_history[i])
        return len(unique_addresses)
    
    def _calculate_windowed(self, last_index: int) -> Optional[int]:
        """Calculate reuse distance with windowed history."""
        # Check if last access is within our tracking window
        if last_index < self.buffer_start_index:
            return None  # Too far back, can't calculate
        
        unique_addresses: Set[int] = set()
        start_pos = last_index + 1 - self.buffer_start_index
        end_pos = self.global_access_index - self.buffer_start_index
        
        for i in range(start_pos, end_pos):
            buffer_idx = i % self.window_size
            addr_at_pos = self.recent_accesses[buffer_idx]
            if addr_at_pos is not None:
                unique_addresses.add(addr_at_pos)
        
        return len(unique_addresses)
    
    def _update_tracking(self, address: int):
        """Update internal tracking structures."""
        if self.unlimited_mode:
            self.access_history[self.global_access_index] = address
        else:
            # Update circular buffer
            buffer_pos = self.global_access_index % self.window_size
            self.recent_accesses[buffer_pos] = address
            
            # Update buffer window if needed
            if self.global_access_index >= self.window_size:
                self.buffer_start_index = self.global_access_index - self.window_size + 1


class TraceProcessor:
    """Processes DynamoRIO trace files and outputs reuse distance results."""
    
    def __init__(self, window_size: int = -1):
        """
        Initialize trace processor.
        
        Parameters
        ----------
        window_size : int
            Memory window size for reuse distance calculation
        """
        self.tracker = ReuseDistanceTracker(window_size)
        self.format_detected = None  # 'csv' or 'legacy'
    
    def parse_trace_line(self, line: str) -> Optional[int]:
        """
        Parse a DynamoRIO trace line and extract address.
        
        Supports both formats:
        - CSV: timestamp,addr,op,size  
        - Legacy: timestamp address operation size
        """
        line = line.strip()
        if not line or line.startswith('#'):
            return None
        
        # Detect format: CSV has commas, legacy has spaces
        if ',' in line:
            # New CSV format: timestamp,addr,op,size
            parts = line.split(',')
            if len(parts) < 2:
                return None
            address_str = parts[1].strip()
        else:
            # Legacy format: timestamp address operation size
            parts = line.split()
            if len(parts) < 2:
                return None
            address_str = parts[1]
        
        try:
            # Clean up the address string
            address_str = address_str.strip()
            
            # Address might have double 0x prefix, fix that
            if address_str.startswith('0x0x'):
                address_str = address_str[2:]  # Remove the extra '0x'
            
            # Handle different address formats
            if address_str.startswith('0x'):
                # Standard hex format: 0x123abc
                address = int(address_str, 16)
            elif address_str.isdigit():
                # Decimal format (unlikely but possible)
                address = int(address_str)
            else:
                # Try parsing as hex without 0x prefix
                address = int(address_str, 16)
            
            return address
        except (ValueError, IndexError) as e:
            # Debug: uncomment to see parsing failures
            # print(f"Failed to parse address: '{address_str}' - {e}", file=sys.stderr)
            return None
    
    def process_trace_file(self, input_file: str) -> bool:
        """
        Process a DynamoRIO trace file.
        
        Parameters
        ----------
        input_file : str
            Path to input trace file
        
        """
        if not os.path.exists(input_file):
            return False
        
        try:
            with open(input_file, 'r') as f:
                for line in f:
                    address = self.parse_trace_line(line)
                    if address is not None:
                        self.tracker.process_access(address)
            return True
        except Exception:
            return False
    
    def write_results(self, output_file: str) -> bool:
        """
        Write reuse distance results to output file.
        
        Format: address: [list of reuse distances]
        
        Parameters
        ----------
        output_file : str
            Path to output file
        
        """
        try:
            with open(output_file, 'w') as f:
                # Sort addresses for consistent output
                for address in sorted(self.tracker.reuse_distances.keys()):
                    distances = self.tracker.reuse_distances[address]
                    distance_str = '[' + ', '.join(map(str, distances)) + ']'
                    f.write(f"0x{address:x}: {distance_str}\n")
            return True
        except Exception:
            return False
    
    def get_stats(self) -> Dict[str, int]:
        """Get basic statistics about the processing."""
        return {
            'total_accesses': self.tracker.global_access_index,
            'addresses_with_reuse': len(self.tracker.reuse_distances),
            'total_addresses': len(self.tracker.last_access_index)
        }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Calculate reuse distances from DynamoRIO memory traces",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("input_trace", help="Input DynamoRIO trace file")
    parser.add_argument("--output", "-o", help="Output file for reuse distances (default: reuse_{input_name}.txt)")
    parser.add_argument("--window-size", type=int, default=-1,
                       help="Memory window size (-1 for unlimited, >0 for windowed)")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.input_trace):
        sys.stderr.write(f"Error: Input file '{args.input_trace}' not found\n")
        sys.exit(1)
    
    if args.window_size == 0:
        sys.stderr.write("Error: Window size cannot be 0\n")
        sys.exit(1)
    
    # Generate output filename if not provided
    if args.output is None:
        input_basename = os.path.basename(args.input_trace)
        # Remove extension if present
        input_name = os.path.splitext(input_basename)[0]
        args.output = f"reuse_{input_name}.txt"
    
    # Process trace
    processor = TraceProcessor(args.window_size)
    
    if not processor.process_trace_file(args.input_trace):
        sys.stderr.write("Error: Failed to process trace file\n")
        sys.exit(1)
    
    # Write results
    if not processor.write_results(args.output):
        sys.stderr.write("Error: Failed to write output file\n")
        sys.exit(1)
    
    # Optional: Write stats to stderr (won't interfere with main output)
    stats = processor.get_stats()
    sys.stderr.write(f"Processed {stats['total_accesses']} accesses, "
                    f"{stats['addresses_with_reuse']} addresses with reuse\n")


if __name__ == "__main__":
    main()