#!/usr/bin/env python3
"""
Time-Series Plotter

Uses timeseries_parser.py to parse .pb files and generate visualization plots.

Usage:
    python timeparser_plot.py <input.pb> [--output plot.png] [--thread THREAD_ID]

Examples:
    # Create plot from protobuf file
    python timeparser_plot.py timeseries_ls_12345.pb

    # Save to specific file
    python timeparser_plot.py timeseries_ls_12345.pb --output my_plot.png

    # Filter by thread
    python timeparser_plot.py timeseries_ls_12345.pb --thread 12345
"""

import sys
import os
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Add the current directory to path to import timeseries_parser
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

try:
    from timeseries_parser import TimeSeriesParser
except ImportError as e:
    print(f"Error: Could not import timeseries_parser", file=sys.stderr)
    print(f"Make sure timeseries_parser.py is in the same directory", file=sys.stderr)
    sys.exit(1)


def plot_timeseries(pb_file, output_file=None, filter_thread=None):
    """
    Create a time series plot from protobuf data

    Args:
        pb_file (str): Path to .pb protobuf file
        output_file (str): Output image file path (None = show plot)
        filter_thread (int): Optional thread ID to filter
    """
    # Parse the protobuf file
    print(f"Parsing {pb_file}...")
    parser = TimeSeriesParser(pb_file)

    # Get the data
    data = parser.to_dict()
    samples = data['samples']
    metadata = data['metadata']

    if not samples:
        print("Error: No samples found in the protobuf file", file=sys.stderr)
        sys.exit(1)

    # Filter by thread if requested
    if filter_thread is not None:
        samples = [s for s in samples if s['thread_id'] == filter_thread]
        if not samples:
            print(f"Error: No samples found for thread {filter_thread}", file=sys.stderr)
            sys.exit(1)

    # Extract data for plotting
    window_numbers = [s['window_number'] for s in samples]
    read_counts = [s['read_count'] for s in samples]
    write_counts = [s['write_count'] for s in samples]
    wss_exact = [s['wss_exact'] for s in samples]
    wss_approx = [s['wss_approx'] for s in samples]

    # Extract read size histograms
    read_sizes = {
        '1': [s['read_size_histogram']['1'] for s in samples],
        '2': [s['read_size_histogram']['2'] for s in samples],
        '4': [s['read_size_histogram']['4'] for s in samples],
        '8': [s['read_size_histogram']['8'] for s in samples],
        '16': [s['read_size_histogram']['16'] for s in samples],
        '32': [s['read_size_histogram']['32'] for s in samples],
        '64': [s['read_size_histogram']['64'] for s in samples],
        'other': [s['read_size_histogram']['other'] for s in samples]
    }

    # Extract write size histograms
    write_sizes = {
        '1': [s['write_size_histogram']['1'] for s in samples],
        '2': [s['write_size_histogram']['2'] for s in samples],
        '4': [s['write_size_histogram']['4'] for s in samples],
        '8': [s['write_size_histogram']['8'] for s in samples],
        '16': [s['write_size_histogram']['16'] for s in samples],
        '32': [s['write_size_histogram']['32'] for s in samples],
        '64': [s['write_size_histogram']['64'] for s in samples],
        'other': [s['write_size_histogram']['other'] for s in samples]
    }

    # Create subplots (5 rows, 1 column)
    fig, axes = plt.subplots(5, 1, figsize=(12, 12))

    # Get sampling window info for legend
    sample_window_refs = metadata['sample_window_refs']
    legend_label = f'Sampling Window: {sample_window_refs:,} refs'

    # Plot 1: Read Count (with size breakdown)
    # Define colors for different sizes
    size_colors = {
        '1': '#1f77b4',   # blue
        '2': '#ff7f0e',   # orange
        '4': '#2ca02c',   # green
        '8': '#d62728',   # red
        '16': '#9467bd',  # purple
        '32': '#8c564b',  # brown
        '64': '#e377c2',  # pink
        'other': '#7f7f7f'  # gray
    }

    # Plot total read count (thicker line)
    axes[0].plot(window_numbers, read_counts, linewidth=2.5,
                 label='Total Reads', color='black', alpha=0.8, zorder=10)

    # Plot each read size (only if non-zero)
    for size_label, counts in read_sizes.items():
        if max(counts) > 0:  # Only plot if there are non-zero values
            axes[0].plot(window_numbers, counts, linewidth=1.5, marker='.',
                        markersize=3, label=f'{size_label}B reads',
                        color=size_colors[size_label], alpha=0.7)

    axes[0].set_ylabel('Read Count', fontsize=11, fontweight='bold')
    axes[0].set_title(f'Time Series Data - {legend_label}', fontsize=14, fontweight='bold', pad=15)
    axes[0].grid(True, alpha=0.3, linestyle='--')
    axes[0].legend(loc='best', framealpha=0.9, fontsize=8, ncol=2)

    # Plot 2: Write Count (with size breakdown)
    # Plot total write count (thicker line)
    axes[1].plot(window_numbers, write_counts, linewidth=2.5,
                 label='Total Writes', color='black', alpha=0.8, zorder=10)

    # Plot each write size (only if non-zero)
    for size_label, counts in write_sizes.items():
        if max(counts) > 0:  # Only plot if there are non-zero values
            axes[1].plot(window_numbers, counts, linewidth=1.5, marker='.',
                        markersize=3, label=f'{size_label}B writes',
                        color=size_colors[size_label], alpha=0.7)

    axes[1].set_ylabel('Write Count', fontsize=11, fontweight='bold')
    axes[1].grid(True, alpha=0.3, linestyle='--')
    axes[1].legend(loc='best', framealpha=0.9, fontsize=8, ncol=2)

    # Plot 3: WSS Exact
    axes[2].plot(window_numbers, wss_exact, marker='^', markersize=4, linewidth=2,
                 label=legend_label, color='tab:green', alpha=0.7)
    axes[2].set_ylabel('WSS Exact', fontsize=11, fontweight='bold')
    axes[2].grid(True, alpha=0.3, linestyle='--')
    axes[2].legend(loc='best', framealpha=0.9, fontsize=9)

    # Plot 4: WSS Approx
    axes[3].plot(window_numbers, wss_approx, marker='d', markersize=4, linewidth=2,
                 label=legend_label, color='tab:red', alpha=0.7)
    axes[3].set_ylabel('WSS Approx', fontsize=11, fontweight='bold')
    axes[3].grid(True, alpha=0.3, linestyle='--')
    axes[3].legend(loc='best', framealpha=0.9, fontsize=9)

    # Plot 5: WSS Absolute Error (|wss_exact - wss_approx|)
    wss_abs_error = [abs(exact - approx) for exact, approx in zip(wss_exact, wss_approx)]
    axes[4].plot(window_numbers, wss_abs_error, marker='x', markersize=5, linewidth=2,
                 label=legend_label, color='tab:purple', alpha=0.7)
    axes[4].set_ylabel('WSS Abs Error', fontsize=11, fontweight='bold')
    axes[4].set_xlabel('Window Index', fontsize=12, fontweight='bold')
    axes[4].grid(True, alpha=0.3, linestyle='--')
    axes[4].legend(loc='best', framealpha=0.9, fontsize=9)

    # Add metadata as text box on the first subplot
    info_text = f"PID: {metadata['pid']} | Profiler: {metadata['profiler']} | "
    info_text += f"Threads: {metadata['num_threads']} | Samples: {len(samples)}"

    fig.text(0.5, 0.995, info_text, ha='center', va='top',
             fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    # Save or show the plot
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_file}")
    else:
        plt.show()

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Create time series plots from protobuf data files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument('input', help='Input .pb protobuf file')
    parser.add_argument('--output', '-o', help='Output image file (default: show plot)')
    parser.add_argument('--thread', type=int, help='Filter by thread ID')

    args = parser.parse_args()

    try:
        plot_timeseries(args.input, args.output, args.thread)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
