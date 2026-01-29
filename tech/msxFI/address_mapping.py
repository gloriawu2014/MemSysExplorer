#!/usr/bin/env python3
"""
Physical address mapping for eDRAM memory hierarchy.
Supports configurable organization and refresh group tracking.
"""

from typing import Tuple, Dict, Optional
from dataclasses import dataclass
import math


@dataclass
class MemoryConfig:
    """Memory organization configuration.

    Hierarchy: Bank Grid (rows x cols) -> Mat -> Subarray -> Cells -> Block
    """
    # Bank organization
    bank_rows: int
    bank_cols: int

    # Mat organization
    mats_per_bank: int

    # Subarray organization
    subarrays_per_mat: int
    subarray_rows: int
    subarray_cols: int

    # Block organization
    block_size_bytes: int
    cells_per_block: Optional[int] = None

    # Refresh group configuration (physical activation groups)
    banks_per_refresh_group: int = 2  # Default: 2 banks refresh together
    mats_per_bank_in_refresh_group: Optional[int] = None  # Default: all mats in bank

    def __post_init__(self):
        if self.cells_per_block is None:
            self.cells_per_block = self.block_size_bytes * 8

        # Set default for mats_per_bank_in_refresh_group
        if self.mats_per_bank_in_refresh_group is None:
            self.mats_per_bank_in_refresh_group = self.mats_per_bank

        # Calculate derived parameters
        self.row_size_bytes = self.subarray_cols // 8
        self.cache_lines_per_row = self.row_size_bytes // self.block_size_bytes

        # Validate that row can hold at least one cache line
        if self.row_size_bytes < self.block_size_bytes:
            raise ValueError(
                f"Row size ({self.row_size_bytes}B) is smaller than block size ({self.block_size_bytes}B). "
                f"Need at least {self.block_size_bytes * 8} columns, got {self.subarray_cols}."
            )

        # Validate all parameters are positive
        params = [
            ('bank_rows', self.bank_rows),
            ('bank_cols', self.bank_cols),
            ('mats_per_bank', self.mats_per_bank),
            ('subarrays_per_mat', self.subarrays_per_mat),
            ('subarray_rows', self.subarray_rows),
            ('subarray_cols', self.subarray_cols),
            ('block_size_bytes', self.block_size_bytes),
        ]

        for name, value in params:
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
            if not self._is_power_of_2(value):
                raise ValueError(f"{name} must be power of 2, got {value}")

        # Validate refresh group parameters
        if self.banks_per_refresh_group <= 0:
            raise ValueError(f"banks_per_refresh_group must be positive, got {self.banks_per_refresh_group}")
        if not self._is_power_of_2(self.banks_per_refresh_group):
            raise ValueError(f"banks_per_refresh_group must be power of 2, got {self.banks_per_refresh_group}")

        if self.mats_per_bank_in_refresh_group <= 0 or self.mats_per_bank_in_refresh_group > self.mats_per_bank:
            raise ValueError(f"mats_per_bank_in_refresh_group must be between 1 and {self.mats_per_bank}, got {self.mats_per_bank_in_refresh_group}")
        if not self._is_power_of_2(self.mats_per_bank_in_refresh_group):
            raise ValueError(f"mats_per_bank_in_refresh_group must be power of 2, got {self.mats_per_bank_in_refresh_group}")

    @staticmethod
    def _is_power_of_2(n: int) -> bool:
        return n > 0 and (n & (n - 1)) == 0

    def get_total_capacity_bytes(self) -> int:
        total_cells = (self.bank_rows * self.bank_cols *
                      self.mats_per_bank * self.subarrays_per_mat *
                      self.subarray_rows * self.subarray_cols)
        return total_cells // 8

    def get_total_subarrays(self) -> int:
        return (self.bank_rows * self.bank_cols *
                self.mats_per_bank * self.subarrays_per_mat)

    def get_total_banks(self) -> int:
        return self.bank_rows * self.bank_cols


class AddressMapper:
    """Address mapper for hierarchical memory organization.

    Address bits (LSB to MSB): [block_offset | subarray_col | subarray_row | subarray_idx | mat_idx | bank_col | bank_row]
    """

    def __init__(self, config: MemoryConfig, subarrays_per_activation_group: int = 8):
        self.config = config
        self.subarrays_per_group = subarrays_per_activation_group

        self._calculate_bit_widths()
        self._calculate_bit_shifts()
        self._calculate_masks()

    def _calculate_bit_widths(self):
        self.block_offset_bits = int(math.log2(self.config.block_size_bytes))

        # Only need column bits if there are multiple cache lines per row
        if self.config.cache_lines_per_row > 1:
            self.subarray_col_bits = int(math.log2(self.config.cache_lines_per_row))
        else:
            self.subarray_col_bits = 0  # No column selection needed

        self.subarray_row_bits = int(math.log2(self.config.subarray_rows))
        self.subarray_idx_bits = int(math.log2(self.config.subarrays_per_mat)) if self.config.subarrays_per_mat > 1 else 0
        self.mat_bits = int(math.log2(self.config.mats_per_bank)) if self.config.mats_per_bank > 1 else 0
        self.bank_col_bits = int(math.log2(self.config.bank_cols))
        self.bank_row_bits = int(math.log2(self.config.bank_rows))

    def _calculate_bit_shifts(self):
        self.block_offset_shift = 0
        self.subarray_col_shift = self.block_offset_shift + self.block_offset_bits
        self.subarray_row_shift = self.subarray_col_shift + self.subarray_col_bits
        self.subarray_idx_shift = self.subarray_row_shift + self.subarray_row_bits
        self.mat_shift = self.subarray_idx_shift + self.subarray_idx_bits
        self.bank_col_shift = self.mat_shift + self.mat_bits
        self.bank_row_shift = self.bank_col_shift + self.bank_col_bits

        self.total_address_bits = (self.bank_row_shift + self.bank_row_bits)

    def _calculate_masks(self):
        self.block_offset_mask = (1 << self.block_offset_bits) - 1
        self.subarray_col_mask = (1 << self.subarray_col_bits) - 1
        self.subarray_row_mask = (1 << self.subarray_row_bits) - 1
        self.subarray_idx_mask = (1 << self.subarray_idx_bits) - 1
        self.mat_mask = (1 << self.mat_bits) - 1
        self.bank_col_mask = (1 << self.bank_col_bits) - 1
        self.bank_row_mask = (1 << self.bank_row_bits) - 1

    def decode_address(self, address: int) -> Dict:
        components = {
            'address': hex(address),
            'block_offset': (address >> self.block_offset_shift) & self.block_offset_mask,
            'subarray_col': (address >> self.subarray_col_shift) & self.subarray_col_mask,
            'subarray_row': (address >> self.subarray_row_shift) & self.subarray_row_mask,
            'subarray_idx': (address >> self.subarray_idx_shift) & self.subarray_idx_mask,
            'mat_idx': (address >> self.mat_shift) & self.mat_mask,
            'bank_col': (address >> self.bank_col_shift) & self.bank_col_mask,
            'bank_row': (address >> self.bank_row_shift) & self.bank_row_mask,
        }

        gid, sid = self.get_ids(address)
        components['activation_group_id'] = gid
        components['subarray_id'] = sid

        return components

    def get_ids(self, address: int) -> Tuple[Tuple[int, int, int, int], int]:
        bank_col = (address >> self.bank_col_shift) & self.bank_col_mask
        bank_row = (address >> self.bank_row_shift) & self.bank_row_mask
        mat_idx = (address >> self.mat_shift) & self.mat_mask
        subarray_idx = (address >> self.subarray_idx_shift) & self.subarray_idx_mask
        subarray_row = (address >> self.subarray_row_shift) & self.subarray_row_mask

        bank_col_group = bank_col // self.config.banks_per_refresh_group
        mat_group = mat_idx // self.config.mats_per_bank_in_refresh_group
        activation_group_id = (bank_row, bank_col_group, mat_group, subarray_row)

        subarray_id = (
            bank_row * self.config.bank_cols * self.config.mats_per_bank * self.config.subarrays_per_mat +
            bank_col * self.config.mats_per_bank * self.config.subarrays_per_mat +
            mat_idx * self.config.subarrays_per_mat +
            subarray_idx
        )

        return activation_group_id, subarray_id

    def get_subarray_address_range(self, subarray_id: int) -> Tuple[int, int]:
        subarrays_per_bank = self.config.mats_per_bank * self.config.subarrays_per_mat
        subarrays_per_row = self.config.bank_cols * subarrays_per_bank

        bank_row = subarray_id // subarrays_per_row
        remainder = subarray_id % subarrays_per_row

        bank_col = remainder // subarrays_per_bank
        remainder = remainder % subarrays_per_bank

        mat_idx = remainder // self.config.subarrays_per_mat
        subarray_idx = remainder % self.config.subarrays_per_mat

        base_address = (
            (bank_row << self.bank_row_shift) |
            (bank_col << self.bank_col_shift) |
            (mat_idx << self.mat_shift) |
            (subarray_idx << self.subarray_idx_shift)
        )

        subarray_size = self.config.subarray_rows * self.config.row_size_bytes

        return base_address, base_address + subarray_size - 1

    def get_activation_info(self, address: int) -> Dict:
        components = self.decode_address(address)

        bank_id = (components['bank_row'], components['bank_col'])
        mat_id = components['mat_idx']
        subarray_id = components['subarray_id']
        row_id = components['subarray_row']

        cache_lines_per_row = self.config.cache_lines_per_row
        activated_bytes = self.config.row_size_bytes
        row_activation_overhead = cache_lines_per_row

        coactivated_subarrays = self.config.banks_per_refresh_group * self.config.mats_per_bank_in_refresh_group * self.config.subarrays_per_mat
        total_coactivated_cache_lines = coactivated_subarrays * cache_lines_per_row
        total_coactivated_bytes = coactivated_subarrays * activated_bytes

        return {
            'bank_id': bank_id,
            'mat_id': mat_id,
            'subarray_id': subarray_id,
            'row_id': row_id,
            'cache_lines_per_row': cache_lines_per_row,
            'activated_cache_lines': cache_lines_per_row,
            'activated_bytes': activated_bytes,
            'row_activation_overhead': row_activation_overhead,
            'coactivated_subarrays': coactivated_subarrays,
            'total_coactivated_cache_lines': total_coactivated_cache_lines,
            'total_coactivated_bytes': total_coactivated_bytes,
        }

    def get_row_address_range(self, address: int) -> Tuple[int, int]:
        components = self.decode_address(address)

        row_base = (
            (components['bank_row'] << self.bank_row_shift) |
            (components['bank_col'] << self.bank_col_shift) |
            (components['mat_idx'] << self.mat_shift) |
            (components['subarray_idx'] << self.subarray_idx_shift) |
            (components['subarray_row'] << self.subarray_row_shift)
        )

        row_size = self.config.row_size_bytes

        return row_base, row_base + row_size - 1

    def get_coactivated_subarrays(self, address: int) -> list:
        components = self.decode_address(address)

        bank_row = components['bank_row']
        bank_col = components['bank_col']
        mat_idx = components['mat_idx']
        subarray_idx = components['subarray_idx']
        row_id = components['subarray_row']

        bank_col_group_start = (bank_col // self.config.banks_per_refresh_group) * self.config.banks_per_refresh_group
        bank_cols_in_group = range(bank_col_group_start, bank_col_group_start + self.config.banks_per_refresh_group)

        mat_group_start = (mat_idx // self.config.mats_per_bank_in_refresh_group) * self.config.mats_per_bank_in_refresh_group
        mats_in_group = range(mat_group_start, mat_group_start + self.config.mats_per_bank_in_refresh_group)

        coactivated = []

        for bc in bank_cols_in_group:
            if bc >= self.config.bank_cols:
                continue
            for m in mats_in_group:
                if m >= self.config.mats_per_bank:
                    continue
                sid = (
                    bank_row * self.config.bank_cols * self.config.mats_per_bank * self.config.subarrays_per_mat +
                    bc * self.config.mats_per_bank * self.config.subarrays_per_mat +
                    m * self.config.subarrays_per_mat +
                    subarray_idx
                )
                coactivated.append((sid, row_id))

        return coactivated

    def print_config(self):
        print(f"Memory Organization:")
        print(f"  Banks: {self.config.bank_rows} × {self.config.bank_cols}")
        print(f"  Mats/Bank: {self.config.mats_per_bank}")
        print(f"  Subarrays/Mat: {self.config.subarrays_per_mat}")
        print(f"  Subarray: {self.config.subarray_rows} rows × {self.config.subarray_cols} bits")
        print(f"  Block size: {self.config.block_size_bytes} bytes")
        print(f"\nRow Configuration:")
        print(f"  Row width: {self.config.row_size_bytes} bytes")
        print(f"  Cache lines/row: {self.config.cache_lines_per_row}")
        print(f"\nRefresh Groups:")
        print(f"  Banks/group: {self.config.banks_per_refresh_group}")
        print(f"  Mats/bank: {self.config.mats_per_bank_in_refresh_group}")
        coactivated = self.config.banks_per_refresh_group * self.config.mats_per_bank_in_refresh_group * self.config.subarrays_per_mat
        print(f"  Co-activated subarrays: {coactivated}")
        print(f"\nAddress bits: {self.total_address_bits} total")


def get_default_config() -> MemoryConfig:
    return MemoryConfig(
        bank_rows=32,
        bank_cols=64,
        mats_per_bank=4,
        subarrays_per_mat=1,
        subarray_rows=16,
        subarray_cols=1024,
        block_size_bytes=64,
        banks_per_refresh_group=2,
        mats_per_bank_in_refresh_group=4
    )


if __name__ == '__main__':
    config = get_default_config()
    mapper = AddressMapper(config)
    mapper.print_config()

    print("\nAddress Decoding Examples:")
    for addr in [0x0, 0x1F1234A, 0x11234A0]:
        components = mapper.decode_address(addr)
        print(f"  {components['address']}: bank({components['bank_row']},{components['bank_col']}) "
              f"mat{components['mat_idx']} subarray{components['subarray_id']} row{components['subarray_row']}")

    print("\nCo-activated Subarrays (refresh group):")
    coactivated = mapper.get_coactivated_subarrays(0x127F01)
    print(f"  Address 0x127F01 co-activates {len(coactivated)} subarrays")
    for sid, row in coactivated[:8]:
        print(f"  Subarray {sid}, Row {row}")
