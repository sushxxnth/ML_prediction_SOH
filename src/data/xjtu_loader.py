"""
XJTU Battery Dataset Loader

Loads the XJTU battery cycling dataset from Xi'an Jiaotong University.
This dataset contains cycling data from multiple batches of batteries
tested at different C-rates.

Dataset structure:
- Batch-1: 2C discharge rate (8 cells)
- Batch-2: 1C discharge rate (8 cells)
- Batch-3: 1.5C discharge rate (8 cells)

Each .mat file contains:
- 'data': Cycle-by-cycle time-series data (voltage, current, temperature, etc.)
- 'summary': Cycle-level summary statistics (capacity, power, etc.)

Author: Battery ML Research
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
import scipy.io

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.base_loader import BaseBatteryLoader, CellData, CycleData
from src.context.extended_context import (
    ExtendedBatteryContext,
    TemperatureContext,
    ChemistryContext,
    UsageProfileContext,
    CRateContext
)


class XJTULoader(BaseBatteryLoader):
    """
    Loader for XJTU Battery Dataset.
    
    Dataset characteristics:
    - Chemistry: LiCoO2 (LCO)
    - Form factor: 18650
    - Nominal capacity: ~2.0 Ah
    - Nominal voltage: 3.7V
    - Temperature: Room temperature (~25°C)
    - Test profiles: Constant current cycling
    """
    
    @property
    def dataset_name(self) -> str:
        return "xjtu"
    
    @property
    def default_chemistry(self) -> str:
        return "LCO"
    
    @property
    def default_temperature(self) -> float:
        return 25.0  # Room temperature
    
    def _parse_raw_data(self) -> Dict[str, CellData]:
        """
        Parse XJTU .mat files.
        
        Returns:
            Dictionary mapping cell_id to CellData
        """
        cells = {}
        
        # Process each batch
        for batch_num in [1, 2, 3]:
            batch_dir = self.data_dir / f"Batch-{batch_num}"
            
            if not batch_dir.exists():
                print(f"[WARN] Batch-{batch_num} directory not found: {batch_dir}")
                continue
            
            # Determine discharge rate based on batch
            discharge_rates = {1: 2.0, 2: 1.0, 3: 1.5}
            discharge_rate = discharge_rates[batch_num]
            
            # Process each battery in the batch
            mat_files = sorted(batch_dir.glob("*.mat"))
            
            for mat_file in mat_files:
                try:
                    cell_id = f"XJTU_Batch{batch_num}_{mat_file.stem}"
                    print(f"  Processing {cell_id}...")
                    
                    cell = self._parse_cell_file(
                        mat_file,
                        cell_id,
                        batch_num,
                        discharge_rate
                    )
                    
                    if cell and len(cell.cycles) > 0:
                        cells[cell_id] = cell
                    
                except Exception as e:
                    print(f"[ERROR] Failed to process {mat_file}: {e}")
                    continue
        
        return cells
    
    def _parse_cell_file(
        self,
        mat_file: Path,
        cell_id: str,
        batch_num: int,
        discharge_rate: float
    ) -> Optional[CellData]:
        """
        Parse a single XJTU .mat file.
        
        Args:
            mat_file: Path to .mat file
            cell_id: Cell identifier
            batch_num: Batch number (1, 2, or 3)
            discharge_rate: Discharge C-rate
            
        Returns:
            CellData instance or None if parsing fails
        """
        # Load MATLAB file
        mat_data = scipy.io.loadmat(str(mat_file))
        
        # Extract summary data
        summary = mat_data['summary'][0, 0]
        
        # Get cycle-level summary statistics
        discharge_capacities = summary['discharge_capacity_Ah'].flatten()
        charge_capacities = summary['charge_capacity_Ah'].flatten()
        discharge_voltages = summary['discharge_median_voltage'].flatten()
        charge_voltages = summary['charge_median_voltage'].flatten()
        
        # Get cycle life
        cycle_life = int(summary['cycle_life'][0, 0])
        
        # Extract time-series data for each cycle
        data_array = mat_data['data'][0]
        
        # Create cell metadata
        cell = CellData(
            cell_id=cell_id,
            source_dataset="xjtu",
            chemistry="LCO",
            nominal_capacity=2.0,  # Approximate nominal capacity
            nominal_voltage=3.7,
            form_factor="18650",
            test_temperature=25.0,  # Room temperature
            charge_rate=0.5,  # Standard charge rate
            discharge_rate=discharge_rate,
            usage_profile="constant_current"
        )
        
        # Process each cycle
        for cycle_idx in range(len(data_array)):
            try:
                cycle_data = data_array[cycle_idx]
                
                # Extract time-series data
                voltage = cycle_data['voltage_V'].flatten()
                current = cycle_data['current_A'].flatten()
                temperature = cycle_data['temperature_C'].flatten()
                time_min = cycle_data['relative_time_min'].flatten()
                
                # Get capacity from summary
                discharge_capacity = discharge_capacities[cycle_idx]
                
                # Skip if capacity is invalid
                if np.isnan(discharge_capacity) or discharge_capacity <= 0:
                    continue
                
                # Compute cycle statistics
                temp_mean = np.mean(temperature) if len(temperature) > 0 else 25.0
                temp_max = np.max(temperature) if len(temperature) > 0 else 25.0
                temp_min = np.min(temperature) if len(temperature) > 0 else 25.0
                
                # Separate charge and discharge phases
                charge_mask = current > 0
                discharge_mask = current < 0
                
                # Compute current statistics (discharge phase)
                discharge_currents = np.abs(current[discharge_mask])
                current_mean = np.mean(discharge_currents) if len(discharge_currents) > 0 else 0
                current_max = np.max(discharge_currents) if len(discharge_currents) > 0 else 0
                
                # Voltage statistics
                voltage_min = np.min(voltage) if len(voltage) > 0 else 2.5
                voltage_max = np.max(voltage) if len(voltage) > 0 else 4.2
                
                # Time statistics (convert minutes to seconds)
                charge_time = np.sum(np.diff(time_min[charge_mask])) * 60 if np.sum(charge_mask) > 1 else 0
                discharge_time = np.sum(np.diff(time_min[discharge_mask])) * 60 if np.sum(discharge_mask) > 1 else 0
                
                # Estimate internal resistance (simplified)
                # IR = ΔV / I during discharge
                if len(discharge_currents) > 0 and current_mean > 0:
                    voltage_drop = voltage_max - np.mean(voltage[discharge_mask])
                    internal_resistance = voltage_drop / current_mean
                else:
                    internal_resistance = np.nan
                
                # Create cycle data
                cycle = CycleData(
                    cell_id=cell_id,
                    cycle_index=cycle_idx,
                    capacity=discharge_capacity,
                    internal_resistance=internal_resistance,
                    soh_capacity=1.0,  # Will be computed later
                    soh_resistance=1.0,  # Will be computed later
                    rul_cycles=0,  # Will be computed later
                    temperature_mean=temp_mean,
                    temperature_max=temp_max,
                    temperature_min=temp_min,
                    current_mean=current_mean,
                    current_max=current_max,
                    voltage_min=voltage_min,
                    voltage_max=voltage_max,
                    charge_time=charge_time,
                    discharge_time=discharge_time
                )
                
                cell.cycles.append(cycle)
                
            except Exception as e:
                print(f"[WARN] Failed to process cycle {cycle_idx} for {cell_id}: {e}")
                continue
        
        return cell
    
    def _create_context(self, cell: CellData) -> ExtendedBatteryContext:
        """
        Create context for a XJTU cell.
        
        Args:
            cell: CellData instance
            
        Returns:
            ExtendedBatteryContext instance
        """
        return ExtendedBatteryContext(
            temperature=TemperatureContext.from_celsius(cell.test_temperature),
            chemistry=ChemistryContext.LCO,
            usage_profile=UsageProfileContext.CONSTANT_CURRENT,
            c_rate=CRateContext.from_c_rate(cell.discharge_rate)
        )


if __name__ == '__main__':
    # Test the loader
    import sys
    
    data_dir = Path(__file__).parent.parent.parent / "data" / "new_datasets" / "XJTU" / "Battery Dataset"
    
    print(f"Loading XJTU dataset from: {data_dir}")
    
    loader = XJTULoader(
        data_dir=str(data_dir),
        use_cache=True
    )
    
    cells = loader.load()
    
    print(f"\n{'='*60}")
    print(f"XJTU Dataset Summary")
    print(f"{'='*60}")
    print(f"Total cells: {len(cells)}")
    
    for cell_id, cell in cells.items():
        print(f"\n{cell_id}:")
        print(f"  Cycles: {len(cell.cycles)}")
        print(f"  Discharge rate: {cell.discharge_rate}C")
        print(f"  Initial capacity: {cell.initial_capacity:.3f} Ah")
        print(f"  Final capacity: {cell.final_capacity:.3f} Ah")
        print(f"  Capacity fade: {(1 - cell.final_capacity/cell.initial_capacity)*100:.1f}%")
    
    # Get statistics
    stats = loader.get_statistics()
    print(f"\n{'='*60}")
    print(f"Dataset Statistics")
    print(f"{'='*60}")
    for key, value in stats.items():
        print(f"{key}: {value}")
