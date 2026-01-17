"""
Storage Degradation Dataset Loader (PLN Dataset)

Parses storage degradation data with SOC, Temperature, and Storage Period.
This dataset contains calendar aging (storage) data, not cycling data.

Key Features:
- SOC levels: 0%, 50%, 100%
- Temperatures: -40°C, -5°C, 25°C, 50°C
- Storage periods: 3W, 3M, 6M
- Capacity measurements after storage periods

Author: Battery ML Research
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
import pandas as pd

from .base_loader import BaseBatteryLoader, CellData, CycleData

# Context helpers
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.context.extended_context import (
    ExtendedBatteryContext,
    TemperatureContext,
    ChemistryContext,
    UsageProfileContext,
    CRateContext,
    normalize_temperature,
    normalize_crate
)


class StorageDegradationLoader(BaseBatteryLoader):
    """
    Loader for storage degradation data (calendar aging).
    
    This dataset tracks capacity loss during storage at different:
    - SOC levels (0%, 50%, 100%)
    - Temperatures (-40°C to 50°C)
    - Storage periods (3W, 3M, 6M)
    """

    @property
    def dataset_name(self) -> str:
        return "storage_degradation_pln"

    @property
    def default_chemistry(self) -> str:
        # PLN cells - typically LCO or NMC, defaulting to LCO
        return "LCO"

    @property
    def default_temperature(self) -> float:
        return 25.0

    def __init__(
        self,
        data_dir: str,
        cache_dir: Optional[str] = None,
        use_cache: bool = True,
        nominal_capacity: float = 1.5  # Ah (typical for PLN cells)
    ):
        super().__init__(data_dir, cache_dir, use_cache)
        self.nominal_capacity = nominal_capacity

    def _parse_raw_data(self) -> Dict[str, CellData]:
        """
        Parse storage degradation data.
        
        Supports both:
        1. Augmented CSV (preferred): processed/PLN_augmented.csv  
        2. Original Excel: PLN_Number_SOC_Temp_StoragePeriod.xlsx
        """
        cells = {}
        
        # Try augmented CSV first
        augmented_csv = self.data_dir / 'processed' / 'PLN_augmented.csv'
        if augmented_csv.exists():
            print(f"Loading augmented storage data from {augmented_csv}")
            df = pd.read_csv(augmented_csv)
        else:
            # Fall back to original Excel
            excel_file = None
            for ext in ['.xlsx', '.xls']:
                potential = list(self.data_dir.glob(f'*{ext}'))
                if potential:
                    excel_file = potential[0]
                    break
            
            if excel_file is None:
                raise FileNotFoundError(f"No Excel or CSV file found in {self.data_dir}")
            
            print(f"Loading storage degradation data from {excel_file}")
            df = pd.read_excel(excel_file)
        
        # Clean column names
        df.columns = df.columns.str.strip()
        
        # Expected columns
        required_cols = ['SOC', 'TEMP', 'Time', 'Discharge Capacity']
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}. Found: {df.columns.tolist()}")
        
        # CRITICAL FIX: The data structure shows each (PLN, SOC, TEMP) has only ONE measurement
        # We need to reconstruct degradation trajectories by:
        # 1. Grouping by (SOC, TEMP) to get all cells at that condition
        # 2. Using Time periods (3W, 3M, 6M) as "cycles" to show degradation over time
        # 3. Each cell (PLN) contributes one measurement per time period
        
        # Filter out rows with missing data
        df = df.dropna(subset=['SOC', 'TEMP', 'Time', 'Discharge Capacity'])
        
        # Group by (SOC, TEMP) - all cells at same condition
        # Then create "synthetic" cells that track average capacity over time
        grouping_cols = ['SOC', 'TEMP']
        
        for (group_key, group_df) in df.groupby(grouping_cols):
            # Get SOC and temperature from group key
            soc_val, temp_val = group_key
            
            # CRITICAL FIX: Group by Time period to get average capacity at each time
            # This creates a degradation trajectory: 3W -> 3M -> 6M
            time_mapping = {'3W': 0.75, '3M': 3.0, '6M': 6.0}  # months
            
            # Get average capacity at each time period (across all PLN replicates)
            time_periods = ['3W', '3M', '6M']
            cycles = []
            initial_capacity = None
            
            for time_period in time_periods:
                time_data = group_df[group_df['Time'] == time_period]
                
                if len(time_data) == 0:
                    continue  # Skip if no data for this time period
                
                # Use average capacity across replicates at this time period
                avg_capacity = float(time_data['Discharge Capacity'].mean())
                std_capacity = float(time_data['Discharge Capacity'].std())
                
                # First time period is initial capacity
                if initial_capacity is None:
                    initial_capacity = avg_capacity
                
                # Compute SOH (relative to initial)
                soh = avg_capacity / initial_capacity if initial_capacity > 0 else 1.0
                
                # Cycle index based on time period order
                cycle_idx = len(cycles)
                
                # RUL: remaining storage periods
                remaining_periods = len([t for t in time_periods if time_periods.index(time_period) < time_periods.index(t) and t in group_df['Time'].values])
                rul_cycles = max(0, remaining_periods * 10)
                
                # Create cell ID for this condition
                cell_id = f"Storage_SOC{soc_val:.0f}_T{temp_val:.0f}C"
                
                cycle = CycleData(
                    cell_id=cell_id,
                    cycle_index=cycle_idx,
                    capacity=avg_capacity,
                    internal_resistance=np.nan,
                    soh_capacity=soh,
                    soh_resistance=np.nan,
                    rul_cycles=rul_cycles,
                    temperature_mean=temp_val,
                    temperature_max=temp_val,
                    temperature_min=temp_val,
                    current_mean=0.0,
                    current_max=0.0,
                    voltage_min=np.nan,
                    voltage_max=np.nan,
                    charge_time=0.0,
                    discharge_time=0.0
                )
                cycles.append(cycle)
            
            # Only create cell if we have at least 2 time periods (degradation trajectory)
            if len(cycles) < 2:
                continue  # Skip cells with insufficient data
            
            # Create cell data (one cell per SOC/TEMP condition with full trajectory)
            cell = CellData(
                cell_id=cell_id,
                source_dataset="Storage_Degradation_PLN",
                chemistry=self.default_chemistry,
                nominal_capacity=self.nominal_capacity,
                nominal_voltage=3.7,
                form_factor="unknown",
                test_temperature=temp_val,
                charge_rate=0.0,
                discharge_rate=0.0,
                usage_profile="storage",
                cycles=cycles,
                total_cycles=len(cycles),
                eol_cycle=len(cycles) - 1,
                initial_capacity=initial_capacity if initial_capacity else self.nominal_capacity,
                final_capacity=cycles[-1].capacity if cycles else np.nan
            )
            
            cells[cell_id] = cell
        
        print(f"Parsed {len(cells)} storage degradation cells")
        return cells

    def _create_context(
        self,
        cell: CellData
    ) -> ExtendedBatteryContext:
        """
        Create context for storage degradation cell.
        
        Key: Extract SOC from cell_id or metadata
        """
        # Extract SOC from cell_id (format: PLN_X_SOC{Y}_T{Z}C or Storage_SOC{Y}_T{Z}C)
        soc_pct = 50.0  # Default
        if 'SOC' in cell.cell_id:
            try:
                # Extract SOC value from cell_id
                parts = cell.cell_id.split('_')
                for part in parts:
                    if part.startswith('SOC'):
                        soc_pct = float(part.replace('SOC', ''))
                        break
            except:
                pass
        
        # Get temperature from cell
        temp_celsius = cell.test_temperature if cell.test_temperature else 25.0
        chemistry_str = cell.chemistry if cell.chemistry else "LCO"
        
        # Create context
        context = ExtendedBatteryContext(
            temperature=TemperatureContext.from_celsius(temp_celsius),
            chemistry=ChemistryContext.from_string(chemistry_str),
            usage_profile=UsageProfileContext.STORAGE,  # Calendar aging
            c_rate=CRateContext.VERY_SLOW,  # Storage = very slow (0C)
            temperature_celsius=temp_celsius,
            c_rate_value=0.0,  # Storage = no C-rate
            soc_pct=soc_pct,  # Store SOC in context
            source_dataset="Storage_Degradation_PLN",
            cell_id=cell.cell_id,
            additional_info={
                'storage_periods': len(cell.cycles)
            }
        )
        
        return context

