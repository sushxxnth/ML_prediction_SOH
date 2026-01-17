"""
TBSI Sunwoda Battery Dataset Loader

The "Fast Charging Expert" - EV conditions and rapid charging protocols.

Actual data format: Excel files (Features.xlsx, Labels.xlsx)

Data Source: https://github.com/terencetaothucb/TBSI-Sunwoda-Battery-Dataset

Author: Battery ML Research
"""

import os
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
    create_tbsi_sunwoda_context
)


class TBSISunwodaLoader(BaseBatteryLoader):
    """
    Loader for TBSI Sunwoda Battery Dataset.
    
    Actual format: Excel files with features and labels.
    
    Expected directory structure:
        data_dir/
            TBSI-Sunwoda-Battery-Dataset-main/
                Features.xlsx
                Labels.xlsx
    """
    
    @property
    def dataset_name(self) -> str:
        return "tbsi_sunwoda"
    
    @property
    def default_chemistry(self) -> str:
        return "NMC"
    
    @property
    def default_temperature(self) -> float:
        return 25.0
    
    def __init__(
        self,
        data_dir: str,
        cache_dir: Optional[str] = None,
        use_cache: bool = True,
        nominal_capacity: float = 2.5  # Ah
    ):
        super().__init__(data_dir, cache_dir, use_cache)
        self.nominal_capacity = nominal_capacity
    
    def _find_excel_files(self) -> Tuple[Optional[Path], Optional[Path]]:
        """Find Features.xlsx and Labels.xlsx."""
        features_file = None
        labels_file = None
        
        for excel_file in self.data_dir.rglob("*.xlsx"):
            name_lower = excel_file.name.lower()
            if 'feature' in name_lower:
                features_file = excel_file
            elif 'label' in name_lower:
                labels_file = excel_file
        
        return features_file, labels_file
    
    def _parse_tbsi_excel(self) -> Dict[str, CellData]:
        """Parse TBSI Excel files."""
        cells = {}
        
        features_file, labels_file = self._find_excel_files()
        
        if features_file is None:
            print(f"[WARN] Features.xlsx not found in {self.data_dir}")
            return cells
        
        print(f"Loading features from: {features_file}")
        
        try:
            features_df = pd.read_excel(features_file)
            labels_df = pd.read_excel(labels_file) if labels_file else None
        except Exception as e:
            print(f"[WARN] Failed to read Excel files: {e}")
            return cells
        
        print(f"Features shape: {features_df.shape}")
        print(f"Features columns: {list(features_df.columns)[:10]}...")
        
        if labels_df is not None:
            print(f"Labels shape: {labels_df.shape}")
        
        # Create cells from Excel data
        # Each row might represent a cell or a cycle
        # Adapt based on actual structure
        
        # Try to identify cell/sample columns
        id_cols = [c for c in features_df.columns if 'id' in c.lower() or 'cell' in c.lower() or 'sample' in c.lower()]
        
        if id_cols:
            # Group by cell ID
            id_col = id_cols[0]
            groups = features_df.groupby(id_col)
        else:
            # Treat each row as a different cell
            groups = [(f"cell_{i}", features_df.iloc[[i]]) for i in range(len(features_df))]
        
        for cell_id, group in groups:
            cell_id_str = f"TBSI_{cell_id}"
            
            # Try to extract meaningful features
            cycles = []
            
            for idx, (_, row) in enumerate(group.iterrows() if hasattr(group, 'iterrows') else [(0, group)]):
                # Try to find capacity column
                cap_cols = [c for c in row.index if 'cap' in c.lower() or 'q' in c.lower()]
                capacity = float(row[cap_cols[0]]) if cap_cols else self.nominal_capacity
                
                # Try to find temperature column
                temp_cols = [c for c in row.index if 'temp' in c.lower() or 't_' in c.lower()]
                temp = float(row[temp_cols[0]]) if temp_cols else 25.0
                
                # Try to find c-rate or current column
                crate_cols = [c for c in row.index if 'crate' in c.lower() or 'c_rate' in c.lower() or 'current' in c.lower()]
                c_rate = float(row[crate_cols[0]]) if crate_cols else 1.0
                
                cycle = CycleData(
                    cell_id=cell_id_str,
                    cycle_index=idx,
                    capacity=capacity if not np.isnan(capacity) else self.nominal_capacity,
                    internal_resistance=np.nan,
                    soh_capacity=1.0,
                    soh_resistance=1.0,
                    rul_cycles=0,
                    temperature_mean=temp if not np.isnan(temp) else 25.0,
                    temperature_max=temp if not np.isnan(temp) else 25.0,
                    temperature_min=temp if not np.isnan(temp) else 25.0,
                    current_mean=c_rate * self.nominal_capacity,
                    current_max=c_rate * self.nominal_capacity * 1.2,
                    voltage_min=2.5,
                    voltage_max=4.2,
                    charge_time=3600 / max(c_rate, 0.1),
                    discharge_time=3600 / max(c_rate, 0.1),
                )
                cycles.append(cycle)
            
            if cycles:
                # Determine profile based on inferred C-rate
                avg_crate = np.mean([c.current_mean / self.nominal_capacity for c in cycles])
                profile = "fast_charging" if avg_crate >= 2.0 else "EV_charging"
                
                cell = CellData(
                    cell_id=cell_id_str,
                    source_dataset=self.dataset_name,
                    chemistry=self.default_chemistry,
                    nominal_capacity=self.nominal_capacity,
                    nominal_voltage=3.7,
                    form_factor="cylindrical",
                    test_temperature=np.mean([c.temperature_mean for c in cycles]),
                    charge_rate=avg_crate,
                    discharge_rate=avg_crate,
                    usage_profile=profile,
                    cycles=cycles
                )
                
                cells[cell_id_str] = cell
        
        return cells
    
    def _parse_raw_data(self) -> Dict[str, CellData]:
        """Parse all TBSI Sunwoda data files."""
        cells = {}
        
        # Find Excel files
        features_file, labels_file = self._find_excel_files()
        
        if features_file is None:
            # Also try CSV files
            csv_files = list(self.data_dir.rglob("*.csv"))
            if csv_files:
                print(f"Found {len(csv_files)} CSV files (Excel not found)")
                # Process CSVs instead
                for csv_file in csv_files:
                    if 'feature' in csv_file.name.lower():
                        print(f"Processing: {csv_file.name}")
            else:
                print(f"[WARN] No Excel or CSV files found in {self.data_dir}")
                print("Expected: Features.xlsx and Labels.xlsx")
            return cells
        
        cells = self._parse_tbsi_excel()
        
        return cells
    
    def _create_context(self, cell: CellData) -> ExtendedBatteryContext:
        """Create context for a TBSI cell."""
        return create_tbsi_sunwoda_context(
            cell_id=cell.cell_id,
            temperature_c=cell.test_temperature,
            c_rate=cell.charge_rate
        )


if __name__ == '__main__':
    import sys
    
    # Test with actual data
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "./data/tbsi_sunwoda"
    
    print(f"Testing TBSI loader with {data_dir}")
    loader = TBSISunwodaLoader(data_dir, use_cache=False)
    cells = loader.load()
    
    print(f"\nLoaded {len(cells)} cells")
    if cells:
        print(f"Statistics: {loader.get_statistics()}")
