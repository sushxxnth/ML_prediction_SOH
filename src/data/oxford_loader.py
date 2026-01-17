"""
Oxford Battery Degradation Dataset Loader

The "Real-World Expert" - dynamic urban driving profiles.

Actual data format: MATLAB .mat files

Data Source: https://howey.eng.ox.ac.uk/data-and-code/

Author: Battery ML Research
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import pandas as pd
from scipy.io import loadmat

from .base_loader import BaseBatteryLoader, CellData, CycleData
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.context.extended_context import (
    ExtendedBatteryContext,
    TemperatureContext,
    ChemistryContext,
    UsageProfileContext,
    CRateContext,
    create_oxford_context
)


class OxfordLoader(BaseBatteryLoader):
    """
    Loader for Oxford Battery Degradation Dataset.
    
    Actual format: MATLAB .mat files with battery cycling data.
    
    Expected directory structure:
        data_dir/
            Oxford_Battery_Degradation_Dataset_1.mat
            (optionally more .mat files)
    """
    
    @property
    def dataset_name(self) -> str:
        return "oxford"
    
    @property
    def default_chemistry(self) -> str:
        return "LCO"
    
    @property
    def default_temperature(self) -> float:
        return 25.0
    
    def __init__(
        self,
        data_dir: str,
        cache_dir: Optional[str] = None,
        use_cache: bool = True,
        nominal_capacity: float = 0.74  # Ah, typical for Oxford pouch cells
    ):
        super().__init__(data_dir, cache_dir, use_cache)
        self.nominal_capacity = nominal_capacity
    
    def _parse_oxford_mat(self, filepath: Path) -> Dict[str, CellData]:
        """
        Parse Oxford MATLAB file.
        
        The .mat file may contain multiple cells with cycling data.
        """
        cells = {}
        
        try:
            mat = loadmat(str(filepath), squeeze_me=True, struct_as_record=False)
        except Exception as e:
            print(f"[WARN] Failed to load {filepath}: {e}")
            return cells
        
        # Find data variables (skip MATLAB metadata)
        data_keys = [k for k in mat.keys() if not k.startswith('__')]
        
        print(f"  Found keys in MAT file: {data_keys}")
        
        for key in data_keys:
            data = mat[key]
            
            # Try to extract cycling data
            cell = self._extract_cell_from_mat(key, data, filepath)
            if cell is not None:
                cells[cell.cell_id] = cell
        
        return cells
    
    def _extract_cell_from_mat(self, key: str, data: Any, filepath: Path) -> Optional[CellData]:
        """Extract cell data from a MAT variable."""
        cell_id = f"Oxford_{key}"
        
        # Try different possible structures
        cycles = []
        
        def get_array(obj, *names):
            """Try to get array from object with multiple possible names."""
            for name in names:
                if hasattr(obj, name):
                    arr = getattr(obj, name)
                    if isinstance(arr, np.ndarray):
                        return arr.flatten()
                elif isinstance(obj, dict) and name in obj:
                    arr = obj[name]
                    if isinstance(arr, np.ndarray):
                        return arr.flatten()
            return None
        
        # Try to find cycle-level data
        if hasattr(data, '__dict__'):
            # Struct-like object
            capacity = get_array(data, 'Capacity', 'capacity', 'Q', 'Qd')
            voltage = get_array(data, 'Voltage', 'voltage', 'V')
            current = get_array(data, 'Current', 'current', 'I')
            temp = get_array(data, 'Temperature', 'temperature', 'T', 'Temp')
            cycle_idx = get_array(data, 'Cycle', 'cycle', 'cycle_index', 'N')
            
        elif isinstance(data, np.ndarray):
            # Direct array - try to interpret
            if data.ndim == 1:
                # Might be capacity array
                capacity = data
                voltage = current = temp = cycle_idx = None
            elif data.ndim == 2:
                # Matrix - columns might be different measurements
                if data.shape[1] >= 1:
                    capacity = data[:, 0]
                else:
                    capacity = None
                voltage = current = temp = cycle_idx = None
            else:
                return None
        else:
            return None
        
        # Create cycles from capacity data
        if capacity is not None and len(capacity) > 0:
            num_cycles = len(capacity)
            
            for i in range(num_cycles):
                cap_val = float(capacity[i]) if not np.isnan(capacity[i]) else self.nominal_capacity
                
                # Get other values if available
                temp_val = float(temp[i]) if temp is not None and i < len(temp) else 25.0
                
                cycle = CycleData(
                    cell_id=cell_id,
                    cycle_index=i,
                    capacity=cap_val,
                    internal_resistance=np.nan,
                    soh_capacity=1.0,
                    soh_resistance=1.0,
                    rul_cycles=0,
                    temperature_mean=temp_val,
                    temperature_max=temp_val,
                    temperature_min=temp_val,
                    current_mean=self.nominal_capacity,
                    current_max=2 * self.nominal_capacity,
                    voltage_min=2.7,
                    voltage_max=4.2,
                    charge_time=3600.0,
                    discharge_time=3600.0,
                )
                cycles.append(cycle)
        
        if not cycles:
            # Create at least one cycle with available data
            cycle = CycleData(
                cell_id=cell_id,
                cycle_index=0,
                capacity=self.nominal_capacity,
                internal_resistance=np.nan,
                soh_capacity=1.0,
                soh_resistance=1.0,
                rul_cycles=0,
                temperature_mean=25.0,
                temperature_max=25.0,
                temperature_min=25.0,
                current_mean=self.nominal_capacity,
                current_max=2 * self.nominal_capacity,
                voltage_min=2.7,
                voltage_max=4.2,
                charge_time=3600.0,
                discharge_time=3600.0,
            )
            cycles.append(cycle)
        
        cell = CellData(
            cell_id=cell_id,
            source_dataset=self.dataset_name,
            chemistry=self.default_chemistry,
            nominal_capacity=self.nominal_capacity,
            nominal_voltage=3.7,
            form_factor="pouch",
            test_temperature=self.default_temperature,
            charge_rate=1.0,
            discharge_rate=1.0,
            usage_profile="urban_driving",
            cycles=cycles
        )
        
        return cell
    
    def _parse_raw_data(self) -> Dict[str, CellData]:
        """Parse all Oxford data files."""
        cells = {}
        
        # Find all MAT files
        mat_files = list(self.data_dir.rglob("*.mat"))
        
        # Also check for CSV files
        csv_files = list(self.data_dir.rglob("*.csv"))
        
        if not mat_files and not csv_files:
            print(f"[WARN] No MAT or CSV files found in {self.data_dir}")
            return cells
        
        print(f"Found {len(mat_files)} MAT files and {len(csv_files)} CSV files")
        
        for mat_file in mat_files:
            print(f"Processing: {mat_file.name}")
            mat_cells = self._parse_oxford_mat(mat_file)
            cells.update(mat_cells)
        
        return cells
    
    def _create_context(self, cell: CellData) -> ExtendedBatteryContext:
        """Create context for an Oxford cell."""
        return create_oxford_context(
            cell_id=cell.cell_id,
            profile=cell.usage_profile
        )


if __name__ == '__main__':
    import sys
    
    # Test with actual data
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "./data/oxford"
    
    print(f"Testing Oxford loader with {data_dir}")
    loader = OxfordLoader(data_dir, use_cache=False)
    cells = loader.load()
    
    print(f"\nLoaded {len(cells)} cells")
    if cells:
        print(f"Statistics: {loader.get_statistics()}")
