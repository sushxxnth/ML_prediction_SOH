"""
Multi-Source Battery Data Module

Supports loading and processing battery cycling data from multiple sources:
- NASA Battery Dataset (baseline)
- Sandia National Labs (temperature variations)
- CALCE University of Maryland (chemistry variations, driving profiles)
- Oxford Battery Degradation (dynamic urban profiles)
- TBSI Sunwoda (fast charging, EV conditions)
- McMaster University (multi-temperature, fast charging)

All datasets are normalized to a common format for unified training.
"""

from .nasa_set5 import (
    load_all_cells,
    save_cycle_tables,
    compute_labels,
    make_cycle_table,
    parse_mat_cell
)

__all__ = [
    'load_all_cells',
    'save_cycle_tables', 
    'compute_labels',
    'make_cycle_table',
    'parse_mat_cell'
]

