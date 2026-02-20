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
    
    def __init__(self, data_dir: str, cache_dir: Optional[str] = None, use_cache: bool = True,
                 min_capacity_ah: float = 0.01):
        super().__init__(data_dir, cache_dir, use_cache)
        self.dataset_name_str = "panasonic_18650pf"
        self.nominal_capacity = 2.9  # Ah (from readme: 2.9Ah Panasonic 18650PF)
        self.min_capacity_ah = min_capacity_ah
        
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
        
        For Panasonic data, Ah can be negative during discharge and positive during charge.
        We identify cycles by detecting rising edges in the Ah signal (transitions from
        discharge to charge, or Ah resets).
        """
        cycles = []
        ah = df['Ah'].values
        current = df['Current'].values
        
        # Detect cycle boundaries by finding rising edges in Ah
        # A rising edge occurs when Ah goes from decreasing to increasing (charge start)
        # or when there's a significant drop (reset to beginning of new cycle)
        
        ah_diff = np.diff(ah)
        
        # Find potential cycle starts:
        # 1. Large negative jumps (resets): Ah drops by > 0.5 Ah
        # 2. Sign changes in derivative (transitions from discharge to charge)
        
        # Method 1: Detect large resets
        large_drops = np.where(ah_diff < -0.3)[0] + 1  # +1 because diff reduces length by 1
        
        # Method 2: Detect current sign changes (discharge to charge transitions)
        # Charge phases have positive current, discharge has negative
        current_changes = np.where(np.diff(np.sign(current)) > 0)[0] + 1
        
        # Combine both methods
        potential_starts = np.concatenate([large_drops, current_changes])
        potential_starts = np.unique(potential_starts)
        potential_starts = np.sort(potential_starts)
        
        # Filter out starts that are too close together (< 50 points)
        filtered_starts = [0]  # Always start from beginning
        for start in potential_starts:
            if start - filtered_starts[-1] > 50:
                filtered_starts.append(start)
        
        # Create cycle ranges
        for i in range(len(filtered_starts)):
            start_idx = filtered_starts[i]
            end_idx = filtered_starts[i + 1] if i + 1 < len(filtered_starts) else len(df)
            
            # Only include cycles with significant data (> 20 points)
            if end_idx - start_idx > 20:
                cycles.append((start_idx, end_idx))
        
        # Fallback: if no cycles found, treat entire sequence as one cycle
        if len(cycles) == 0 and len(df) > 100:
            cycles.append((0, len(df)))
        
        return cycles
    
    def _parse_raw_data(self) -> Dict[str, CellData]:
        """Parse Panasonic 18650PF dataset from .mat files."""
        all_cells_data: Dict[str, CellData] = {}
        data_dir = Path(self.data_dir)
        
        # Search both root and nested Panasonic data folder
        search_roots = [data_dir, data_dir / "Panasonic 18650PF Data"]
        
        # Temperature folders to process (excluding "Trise" variants for now)
        temp_folders = ['-20degC', '-10degC', '0degC', '10degC', '25degC']
        
        for root in search_roots:
            if not root.exists():
                continue
            for temp_folder in temp_folders:
                temp_dir = root / temp_folder
                if not temp_dir.exists():
                    continue
                
                # Extract temperature
                temp_match = re.search(r'(-?\d+)degC', temp_folder)
                temp_c = float(temp_match.group(1)) if temp_match else 25.0
                
                # Process Drive cycles folder (most relevant for SOH/RUL)
                drive_cycles_dir = temp_dir / 'Drive cycles'
                if not drive_cycles_dir.exists():
                    drive_cycles_dir = temp_dir / 'Drive Cycles'
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
                            
                            if np.isnan(capacity_ah) or capacity_ah < self.min_capacity_ah:
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
