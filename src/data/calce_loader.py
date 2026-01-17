"""
CALCE Battery Data Loader (University of Maryland)

The "Chemistry & Profile Expert" - multiple chemistries and driving profiles.

Actual data format: Tab-separated TXT files with columns:
- Time, mV, mA, Temperature, Capacity, etc.

Data Source: https://calce.umd.edu/battery-data

Author: Battery ML Research
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import pandas as pd

from .base_loader import BaseBatteryLoader, CellData, CycleData
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.context.extended_context import (
    ExtendedBatteryContext,
    TemperatureContext,
    ChemistryContext,
    UsageProfileContext,
    CRateContext,
    create_calce_context
)


class CALCELoader(BaseBatteryLoader):
    """
    Loader for CALCE battery datasets.
    
    Supports CALCE sub-datasets:
    - CS2: Prismatic LCO cells
    - CX2: Cylindrical cells
    
    Actual directory structure:
        data_dir/
            CS2_data/CS2_data/type_1/CS2_33/CS2_33/*.txt
            CX2_data/...
    
    TXT Format (tab-separated):
        Time, Status code, mV, mA, Temperature, Capacity, ...
    """
    
    # Sub-dataset configurations
    SUBDATASETS = {
        'CS2': {
            'chemistry': 'LCO',
            'form_factor': 'prismatic',
            'nominal_capacity': 1.1,  # Ah
            'nominal_voltage': 3.7,
        },
        'CX2': {
            'chemistry': 'LCO',
            'form_factor': 'cylindrical',
            'nominal_capacity': 2.0,
            'nominal_voltage': 3.7,
        },
    }
    
    @property
    def dataset_name(self) -> str:
        return "calce"
    
    @property
    def default_chemistry(self) -> str:
        return "LCO"
    
    @property
    def default_temperature(self) -> float:
        return 25.0
    
    def _identify_subdataset(self, filepath: Path) -> str:
        """Identify which sub-dataset a file belongs to."""
        path_str = str(filepath).upper()
        if 'CS2' in path_str:
            return 'CS2'
        elif 'CX2' in path_str:
            return 'CX2'
        return 'CS2'  # Default
    
    def _extract_cell_id(self, filepath: Path) -> str:
        """Extract cell ID from filepath."""
        # Path like: .../CS2_data/type_1/CS2_33/CS2_33/CS2_33_1_28_10.txt
        parts = filepath.stem.split('_')
        if len(parts) >= 2:
            return f"{parts[0]}_{parts[1]}"  # e.g., CS2_33
        return filepath.stem
    
    def _parse_calce_txt(self, filepath: Path) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Parse a CALCE TXT file.
        
        Returns:
            Tuple of (DataFrame with cycle data, metadata dict)
        """
        metadata = {}
        
        try:
            # Read tab-separated file
            df = pd.read_csv(filepath, sep='\t', encoding='utf-8', on_bad_lines='skip')
        except Exception as e:
            print(f"[WARN] Failed to read {filepath}: {e}")
            return pd.DataFrame(), metadata
        
        # Normalize column names
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
        
        # Extract metadata
        subdataset = self._identify_subdataset(filepath)
        cell_id = self._extract_cell_id(filepath)
        config = self.SUBDATASETS.get(subdataset, self.SUBDATASETS['CS2'])
        
        metadata = {
            'subdataset': subdataset,
            'cell_id': cell_id,
            'chemistry': config['chemistry'],
            'form_factor': config['form_factor'],
            'nominal_capacity': config['nominal_capacity'],
            'filepath': str(filepath)
        }
        
        return df, metadata
    
    def _aggregate_cell_data(self, txt_files: List[Path]) -> Optional[CellData]:
        """
        Aggregate multiple TXT files for a single cell into CellData.
        """
        if not txt_files:
            return None
        
        # Parse first file to get metadata
        _, metadata = self._parse_calce_txt(txt_files[0])
        if not metadata:
            return None
        
        cell_id = metadata['cell_id']
        subdataset = metadata['subdataset']
        config = self.SUBDATASETS.get(subdataset, self.SUBDATASETS['CS2'])
        
        # Aggregate all files for this cell
        all_data = []
        for txt_file in txt_files:
            df, _ = self._parse_calce_txt(txt_file)
            if not df.empty:
                all_data.append(df)
        
        if not all_data:
            return None
        
        # Combine all data
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Map columns
        col_map = {
            'mv': 'voltage',
            'ma': 'current',
            'temperature': 'temp',
            'capacity': 'capacity',
            'time': 'time',
            'pgm_cycle': 'cycle'
        }
        
        for old_col, new_col in col_map.items():
            if old_col in combined_df.columns:
                combined_df[new_col] = combined_df[old_col]
        
        # Convert units
        if 'voltage' in combined_df.columns:
            combined_df['voltage'] = combined_df['voltage'] / 1000.0  # mV to V
        if 'current' in combined_df.columns:
            combined_df['current'] = combined_df['current'] / 1000.0  # mA to A
        
        # Group by cycle if available, otherwise by file
        cycles = []
        
        if 'cycle' in combined_df.columns:
            cycle_groups = combined_df.groupby('cycle')
        else:
            # Treat each file as a cycle
            cycle_groups = [(i, all_data[i]) for i in range(len(all_data))]
        
        cycle_idx = 0
        for cycle_key, group in cycle_groups:
            if isinstance(group, pd.DataFrame) and len(group) > 0:
                # Compute cycle statistics
                voltage = group.get('voltage', pd.Series([3.7])).values
                current = group.get('current', pd.Series([0.5])).values
                temp = group.get('temp', pd.Series([25.0])).values
                capacity = group.get('capacity', pd.Series([config['nominal_capacity']])).values
                
                # Get discharge capacity (when current < 0)
                discharge_mask = current < 0 if len(current) > 0 else np.array([False])
                if discharge_mask.any():
                    discharge_capacity = np.max(np.abs(capacity[discharge_mask])) if capacity[discharge_mask].size > 0 else config['nominal_capacity']
                else:
                    discharge_capacity = config['nominal_capacity']
                
                cycle = CycleData(
                    cell_id=f"CALCE_{cell_id}",
                    cycle_index=cycle_idx,
                    capacity=float(discharge_capacity) if not np.isnan(discharge_capacity) else config['nominal_capacity'],
                    internal_resistance=np.nan,
                    soh_capacity=1.0,
                    soh_resistance=1.0,
                    rul_cycles=0,
                    temperature_mean=float(np.nanmean(temp)) if len(temp) > 0 else 25.0,
                    temperature_max=float(np.nanmax(temp)) if len(temp) > 0 else 25.0,
                    temperature_min=float(np.nanmin(temp)) if len(temp) > 0 else 25.0,
                    current_mean=float(np.nanmean(np.abs(current))) if len(current) > 0 else 0.5,
                    current_max=float(np.nanmax(np.abs(current))) if len(current) > 0 else 1.0,
                    voltage_min=float(np.nanmin(voltage)) if len(voltage) > 0 else 2.5,
                    voltage_max=float(np.nanmax(voltage)) if len(voltage) > 0 else 4.2,
                    charge_time=3600.0,
                    discharge_time=3600.0,
                )
                cycles.append(cycle)
                cycle_idx += 1
        
        if not cycles:
            return None
        
        cell = CellData(
            cell_id=f"CALCE_{cell_id}",
            source_dataset=self.dataset_name,
            chemistry=config['chemistry'],
            nominal_capacity=config['nominal_capacity'],
            nominal_voltage=config['nominal_voltage'],
            form_factor=config['form_factor'],
            test_temperature=25.0,
            charge_rate=0.5,
            discharge_rate=0.5,
            usage_profile="constant_current",
            cycles=cycles
        )
        
        return cell
    
    def _parse_raw_data(self) -> Dict[str, CellData]:
        """Parse all CALCE data files."""
        cells = {}
        
        # Find all TXT files
        txt_files = list(self.data_dir.rglob("*.txt"))
        
        if not txt_files:
            print(f"[WARN] No TXT files found in {self.data_dir}")
            print("Expected structure: data/calce/CS2_data/CS2_data/type_*/CS2_*/CS2_*/*.txt")
            return cells
        
        print(f"Found {len(txt_files)} TXT files in CALCE dataset")
        
        # Group files by cell ID
        cell_files: Dict[str, List[Path]] = {}
        for txt_file in txt_files:
            cell_id = self._extract_cell_id(txt_file)
            if cell_id not in cell_files:
                cell_files[cell_id] = []
            cell_files[cell_id].append(txt_file)
        
        print(f"Found {len(cell_files)} unique cells")
        
        # Process each cell
        for cell_id, files in cell_files.items():
            cell = self._aggregate_cell_data(files)
            if cell is not None:
                cells[cell.cell_id] = cell
        
        return cells
    
    def _create_context(self, cell: CellData) -> ExtendedBatteryContext:
        """Create context for a CALCE cell."""
        return create_calce_context(
            cell_id=cell.cell_id,
            chemistry=cell.chemistry,
            profile=cell.usage_profile,
            temperature_c=cell.test_temperature
        )
    
    def get_cells_by_type(self, cell_type: str) -> List[CellData]:
        """Get cells of a specific type (CS2, CX2)."""
        if not self._loaded:
            self.load()
        
        cell_type = cell_type.upper()
        return [
            cell for cell in self.cells.values()
            if cell_type in cell.cell_id.upper()
        ]


if __name__ == '__main__':
    import sys
    
    # Test with actual data
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "./data/calce"
    
    print(f"Testing CALCE loader with {data_dir}")
    loader = CALCELoader(data_dir, use_cache=False)
    cells = loader.load()
    
    print(f"\nLoaded {len(cells)} cells")
    if cells:
        print(f"Statistics: {loader.get_statistics()}")
        
        # Show sample cell
        sample_cell = list(cells.values())[0]
        print(f"\nSample cell: {sample_cell.cell_id}")
        print(f"  Cycles: {len(sample_cell.cycles)}")
        print(f"  Chemistry: {sample_cell.chemistry}")
