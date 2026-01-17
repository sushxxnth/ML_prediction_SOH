"""
Panasonic 18650PF Battery Dataset Loader

Dataset from University of Wisconsin-Madison (Dr. Phillip Kollmeyer)
Temperatures: -20°C, -10°C, 0°C, 10°C, 25°C
Profiles: Real-world EV drive cycles (US06, HWFET, UDDS, LA92, NN)
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.io import loadmat

from src.data.base_loader import BaseBatteryLoader, CellData, CycleData
from src.context.extended_context import (
    ExtendedBatteryContext,
    TemperatureContext,
    ChemistryContext,
    UsageProfileContext,
    CRateContext,
    normalize_temperature,
    normalize_crate,
    chemistry_to_id
)


class Panasonic18650PFLoader(BaseBatteryLoader):
    """Loader for Panasonic 18650PF dataset with sub-zero temperatures."""
    
    def __init__(self, data_dir: str, cache_dir: Optional[str] = None, use_cache: bool = True):
        super().__init__(data_dir, cache_dir, use_cache)
        self.dataset_name_str = "panasonic_18650pf"
        self.nominal_capacity = 2.9  # Ah (from readme: 2.9Ah Panasonic 18650PF)
        
    @property
    def dataset_name(self) -> str:
        return self.dataset_name_str
    
    @property
    def default_chemistry(self) -> str:
        return "NMC"  # Panasonic 18650PF is typically NMC
    
    @property
    def default_temperature(self) -> float:
        return 25.0
    
    def _extract_temperature_from_path(self, file_path: Path) -> float:
        """Extract temperature from file path or folder name."""
        path_str = str(file_path)
        
        # Check for temperature in path (e.g., "-20degC", "25degC")
        temp_patterns = [
            (r'(-?\d+)degC', lambda m: float(m.group(1))),
            (r'n(\d+)degC', lambda m: -float(m.group(1))),  # "n20degC" = -20°C
        ]
        
        for pattern, converter in temp_patterns:
            match = re.search(pattern, path_str)
            if match:
                return converter(match)
        
        # Default to 25°C if not found
        return 25.0
    
    def _identify_cycles(self, df: pd.DataFrame) -> List[Tuple[int, int]]:
        """
        Identify cycles in the time-series data.
        Cycles are typically separated by Ah resets (Ah goes back to 0) or charge phases.
        """
        cycles = []
        ah = df['Ah'].values
        current = df['Current'].values
        
        # Find Ah resets (Ah goes to 0 or very low, indicating start of new cycle)
        ah_resets = np.where(ah < 0.01)[0]
        
        if len(ah_resets) == 0:
            # No resets found, treat entire sequence as one cycle
            if len(df) > 100:  # Only if we have enough data
                cycles.append((0, len(df)))
            return cycles
        
        # Add start and end indices
        start_idx = 0
        for reset_idx in ah_resets:
            if reset_idx > start_idx + 100:  # Minimum cycle length
                cycles.append((start_idx, reset_idx))
            start_idx = reset_idx
        
        # Add final cycle
        if start_idx < len(df) - 100:
            cycles.append((start_idx, len(df)))
        
        # If no cycles found, split into chunks based on discharge phases
        if len(cycles) == 0:
            # Find discharge phases (negative current)
            discharge_mask = current < -0.1
            if discharge_mask.sum() > 100:
                # Split into chunks of ~1000 points
                chunk_size = 1000
                for i in range(0, len(df), chunk_size):
                    if i + chunk_size < len(df):
                        cycles.append((i, i + chunk_size))
        
        return cycles if len(cycles) > 0 else [(0, len(df))]
    
    def _parse_raw_data(self) -> Dict[str, CellData]:
        """Parse Panasonic 18650PF dataset from .mat files."""
        all_cells_data: Dict[str, CellData] = {}
        data_dir = Path(self.data_dir)
        
        # Temperature folders to process (excluding "Trise" variants for now)
        temp_folders = ['-20degC', '-10degC', '0degC', '10degC', '25degC']
        
        for temp_folder in temp_folders:
            temp_dir = data_dir / temp_folder
            if not temp_dir.exists():
                continue
            
            # Extract temperature
            temp_match = re.search(r'(-?\d+)degC', temp_folder)
            temp_c = float(temp_match.group(1)) if temp_match else 25.0
            
            # Process Drive cycles folder (most relevant for SOH/RUL)
            drive_cycles_dir = temp_dir / 'Drive cycles'
            if not drive_cycles_dir.exists():
                continue
            
            print(f"Parsing Panasonic 18650PF from {temp_folder} (temp={temp_c}°C)")
            
            # Process each .mat file in Drive cycles
            for mat_file in sorted(drive_cycles_dir.glob('*.mat')):
                try:
                    # Load .mat file
                    mat = loadmat(str(mat_file), squeeze_me=True, struct_as_record=False)
                    
                    if 'meas' not in mat:
                        continue
                    
                    meas = mat['meas']
                    
                    # Extract data columns
                    voltage = np.asarray(meas.Voltage).flatten()
                    current = np.asarray(meas.Current).flatten()
                    ah = np.asarray(meas.Ah).flatten()
                    time = np.asarray(meas.Time).flatten()
                    battery_temp = np.asarray(meas.Battery_Temp_degC).flatten()
                    chamber_temp = np.asarray(meas.Chamber_Temp_degC).flatten()
                    
                    # Create DataFrame
                    df = pd.DataFrame({
                        'Voltage': voltage,
                        'Current': current,
                        'Ah': ah,
                        'Time': time,
                        'Battery_Temp': battery_temp,
                        'Chamber_Temp': chamber_temp
                    })
                    
                    # Identify cycles
                    cycle_indices = self._identify_cycles(df)
                    
                    if len(cycle_indices) == 0:
                        continue
                    
                    # Extract cell ID from filename
                    cell_id_base = mat_file.stem
                    # Create unique cell ID with temperature
                    cell_id = f"Panasonic_{temp_c}C_{cell_id_base}"
                    
                    cycles_data: List[CycleData] = []
                    initial_capacity = None
                    
                    for cycle_idx, (start_idx, end_idx) in enumerate(cycle_indices):
                        cycle_df = df.iloc[start_idx:end_idx].copy()
                        
                        if len(cycle_df) < 10:
                            continue
                        
                        # Calculate capacity from Ah (discharge capacity)
                        # Capacity is the maximum Ah reached during discharge
                        cycle_ah = cycle_df['Ah'].values
                        discharge_mask = cycle_df['Current'].values < -0.1
                        
                        if discharge_mask.sum() > 0:
                            discharge_ah = cycle_ah[discharge_mask]
                            capacity_ah = np.max(discharge_ah) - np.min(discharge_ah) if len(discharge_ah) > 0 else np.nan
                        else:
                            # Fallback: use Ah range
                            capacity_ah = np.max(cycle_ah) - np.min(cycle_ah) if len(cycle_ah) > 0 else np.nan
                        
                        if np.isnan(capacity_ah) or capacity_ah < 0.1:
                            continue
                        
                        # Set initial capacity from first cycle
                        if initial_capacity is None:
                            initial_capacity = capacity_ah
                        
                        # Calculate SOH
                        soh_capacity = capacity_ah / initial_capacity if initial_capacity > 0 else 1.0
                        
                        # Calculate average metrics
                        temp_mean = float(np.nanmean(cycle_df['Battery_Temp'].values))
                        temp_max = float(np.nanmax(cycle_df['Battery_Temp'].values))
                        temp_min = float(np.nanmin(cycle_df['Battery_Temp'].values))
                        current_mean = float(np.nanmean(cycle_df['Current'].values))
                        current_max = float(np.nanmax(np.abs(cycle_df['Current'].values)))
                        voltage_min = float(np.nanmin(cycle_df['Voltage'].values))
                        voltage_max = float(np.nanmax(cycle_df['Voltage'].values))
                        
                        # Calculate charge/discharge rates (C-rate)
                        charge_rate = abs(current_mean) / self.nominal_capacity if current_mean > 0 else 0.0
                        discharge_rate = abs(current_mean) / self.nominal_capacity if current_mean < 0 else 0.0
                        
                        # Time duration
                        time_duration = float(cycle_df['Time'].iloc[-1] - cycle_df['Time'].iloc[0])
                        
                        cycle_data = CycleData(
                            cell_id=cell_id,
                            cycle_index=cycle_idx,
                            capacity=capacity_ah,
                            internal_resistance=np.nan,  # Not available in this dataset
                            soh_capacity=soh_capacity,
                            soh_resistance=np.nan,
                            rul_cycles=0,  # Will be computed later
                            temperature_mean=temp_mean,
                            temperature_max=temp_max,
                            temperature_min=temp_min,
                            current_mean=current_mean,
                            current_max=current_max,
                            voltage_min=voltage_min,
                            voltage_max=voltage_max,
                            charge_time=time_duration if current_mean > 0 else 0.0,
                            discharge_time=time_duration if current_mean < 0 else 0.0,
                            context=None  # Will be set at cell level
                        )
                        cycles_data.append(cycle_data)
                    
                    if len(cycles_data) > 0:
                        # Create context
                        avg_temp = temp_c
                        avg_charge_rate = np.mean([c.current_mean for c in cycles_data if c.current_mean > 0]) / self.nominal_capacity if any(c.current_mean > 0 for c in cycles_data) else 0.0
                        avg_discharge_rate = np.mean([abs(c.current_mean) for c in cycles_data if c.current_mean < 0]) / self.nominal_capacity if any(c.current_mean < 0 for c in cycles_data) else 1.0
                        
                        cell = CellData(
                            cell_id=cell_id,
                            source_dataset=self.dataset_name,
                            chemistry=self.default_chemistry,
                            nominal_capacity=self.nominal_capacity,
                            nominal_voltage=3.7,
                            form_factor="18650",
                            test_temperature=avg_temp,
                            charge_rate=avg_charge_rate,
                            discharge_rate=avg_discharge_rate,
                            usage_profile="drive_cycle",  # Real-world drive cycles
                            cycles=cycles_data,
                            context=None  # Will be set by base_loader.load()
                        )
                        
                        cell.compute_labels()
                        all_cells_data[cell_id] = cell
                        print(f"  Loaded {cell_id}: {len(cycles_data)} cycles, temp={avg_temp}°C")
                
                except Exception as e:
                    print(f"[WARN] Failed to parse {mat_file}: {e}")
                    import traceback
                    traceback.print_exc()
        
        return all_cells_data
    
    def _create_context(self, cell: CellData) -> ExtendedBatteryContext:
        """Create context for a Panasonic cell."""
        return ExtendedBatteryContext(
            cell_id=cell.cell_id,
            source_dataset=self.dataset_name,
            temperature_celsius=cell.test_temperature,
            c_rate_value=max(cell.charge_rate, cell.discharge_rate),  # Use max C-rate
            chemistry=ChemistryContext.from_string(cell.chemistry),
            usage_profile=UsageProfileContext.from_string(cell.usage_profile),
            temperature=TemperatureContext.from_celsius(cell.test_temperature),
            c_rate=CRateContext.from_c_rate(max(cell.charge_rate, cell.discharge_rate))
        )

