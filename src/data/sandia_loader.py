"""
Sandia National Labs Battery Data Loader

The "Temperature King" - comprehensive temperature variation testing.

Key Features:
- Multiple chemistries: LCO, LFP, NCA, NMC
- Temperature range: 5°C to 45°C
- Multiple C-rates: 0.5C to 3C
- Long-term cycling and degradation data

Data Source: https://www.batteryarchive.org/

Author: Battery ML Research
"""

import os
import re
import glob
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
    create_sandia_context
)


class SandiaLoader(BaseBatteryLoader):
    """
    Loader for Sandia National Labs battery datasets.
    
    Supports various Sandia studies from BatteryArchive.org:
    - SNL_18650_LCO: LCO cells at various temperatures
    - SNL_18650_LFP: LFP cells
    - SNL_18650_NCA: NCA cells
    - SNL_18650_NMC: NMC cells
    
    Expected directory structure:
        data_dir/
            SNL_18650_LCO/
                cell_001_25C_1C.csv
                cell_002_35C_2C.csv
                ...
            SNL_18650_NMC/
                ...
            metadata.csv (optional)
    
    CSV Format (BatteryArchive standard):
        cycle_index, discharge_capacity, charge_capacity, 
        internal_resistance, temperature, voltage_min, voltage_max,
        current_max, time_charge, time_discharge
    """
    
    # Chemistry mapping from folder names
    CHEMISTRY_MAP = {
        'LCO': ChemistryContext.LCO,
        'LFP': ChemistryContext.LFP,
        'NCA': ChemistryContext.NCA,
        'NMC': ChemistryContext.NMC,
        'NMC111': ChemistryContext.NMC_111,
        'NMC523': ChemistryContext.NMC_523,
        'NMC622': ChemistryContext.NMC_622,
        'NMC811': ChemistryContext.NMC_811,
    }
    
    # Temperature extraction patterns
    TEMP_PATTERNS = [
        r'(\d+)C',           # "25C" -> 25
        r'(\d+)deg',         # "25deg" -> 25
        r'T(\d+)',           # "T25" -> 25
        r'_(\d+)_',          # "_25_" -> 25
    ]
    
    # C-rate extraction patterns
    CRATE_PATTERNS = [
        r'(\d+\.?\d*)C',     # "2C" or "1.5C"
        r'C(\d+\.?\d*)',     # "C2" format
        r'(\d+\.?\d*)A',     # Current in Amps (need nominal capacity)
    ]
    
    @property
    def dataset_name(self) -> str:
        return "sandia"
    
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
        nominal_capacity: float = 2.5  # Ah, typical 18650
    ):
        super().__init__(data_dir, cache_dir, use_cache)
        self.nominal_capacity = nominal_capacity
    
    def _extract_temperature(self, filename: str) -> float:
        """Extract temperature from filename."""
        for pattern in self.TEMP_PATTERNS:
            match = re.search(pattern, filename, re.IGNORECASE)
            if match:
                return float(match.group(1))
        return self.default_temperature
    
    def _extract_crate(self, filename: str) -> float:
        """Extract C-rate from filename."""
        for pattern in self.CRATE_PATTERNS:
            match = re.search(pattern, filename, re.IGNORECASE)
            if match:
                return float(match.group(1))
        return 1.0  # Default 1C
    
    def _extract_chemistry(self, path: str) -> str:
        """Extract chemistry from path."""
        path_str = str(path).upper()
        for chem in self.CHEMISTRY_MAP.keys():
            if chem in path_str:
                return chem
        return self.default_chemistry
    
    def _parse_sandia_csv(self, filepath: Path) -> Tuple[List[CycleData], Dict[str, Any]]:
        """
        Parse a Sandia CSV file.
        
        Returns:
            Tuple of (list of CycleData, metadata dict)
        """
        cycles = []
        metadata = {}
        
        try:
            df = pd.read_csv(filepath)
        except Exception as e:
            print(f"[WARN] Failed to read {filepath}: {e}")
            return cycles, metadata
        
        # Normalize column names
        df.columns = df.columns.str.lower().str.strip()
        df.columns = df.columns.str.replace(' ', '_')
        
        # Extract metadata from filename
        filename = filepath.stem
        temperature = self._extract_temperature(filename)
        c_rate = self._extract_crate(filename)
        chemistry = self._extract_chemistry(str(filepath))
        
        metadata = {
            'temperature': temperature,
            'c_rate': c_rate,
            'chemistry': chemistry,
            'filename': filename
        }
        
        # Map columns (handle different naming conventions)
        col_maps = {
            'cycle_index': ['cycle_index', 'cycle', 'cycle_number', 'idx'],
            'capacity': ['discharge_capacity', 'capacity', 'cap_discharge', 'qd'],
            'ir': ['internal_resistance', 'ir', 'dcir', 'resistance'],
            'temp': ['temperature', 'temp', 'cell_temp', 'ambient_temp'],
            'v_min': ['voltage_min', 'v_min', 'discharge_voltage_min'],
            'v_max': ['voltage_max', 'v_max', 'charge_voltage_max'],
            'i_max': ['current_max', 'i_max', 'max_current'],
            't_charge': ['time_charge', 'charge_time', 'charge_duration'],
            't_discharge': ['time_discharge', 'discharge_time', 'discharge_duration'],
        }
        
        def get_col(name: str) -> Optional[str]:
            for col in col_maps.get(name, [name]):
                if col in df.columns:
                    return col
            return None
        
        # Parse cycles
        cycle_col = get_col('cycle_index')
        if cycle_col is None:
            # If no cycle column, assume sequential
            df['_cycle_idx'] = range(len(df))
            cycle_col = '_cycle_idx'
        
        for _, row in df.iterrows():
            try:
                cycle = CycleData(
                    cell_id=filename,
                    cycle_index=int(row.get(cycle_col, 0)),
                    capacity=float(row.get(get_col('capacity'), np.nan) or np.nan),
                    internal_resistance=float(row.get(get_col('ir'), np.nan) or np.nan),
                    soh_capacity=1.0,  # Will be computed later
                    soh_resistance=1.0,
                    rul_cycles=0,
                    temperature_mean=float(row.get(get_col('temp'), temperature) or temperature),
                    temperature_max=float(row.get(get_col('temp'), temperature) or temperature),
                    temperature_min=float(row.get(get_col('temp'), temperature) or temperature),
                    current_mean=c_rate * self.nominal_capacity,
                    current_max=float(row.get(get_col('i_max'), c_rate * self.nominal_capacity) or c_rate * self.nominal_capacity),
                    voltage_min=float(row.get(get_col('v_min'), 2.5) or 2.5),
                    voltage_max=float(row.get(get_col('v_max'), 4.2) or 4.2),
                    charge_time=float(row.get(get_col('t_charge'), 3600) or 3600),
                    discharge_time=float(row.get(get_col('t_discharge'), 3600) or 3600),
                )
                cycles.append(cycle)
            except Exception as e:
                continue
        
        return cycles, metadata
    
    def _parse_raw_data(self) -> Dict[str, CellData]:
        """Parse all Sandia data files."""
        cells = {}
        
        # Find all CSV files
        csv_files = list(self.data_dir.rglob("*.csv"))
        
        if not csv_files:
            print(f"[WARN] No CSV files found in {self.data_dir}")
            print("Please download Sandia data from https://www.batteryarchive.org/")
            return cells
        
        print(f"Found {len(csv_files)} CSV files in {self.data_dir}")
        
        for filepath in csv_files:
            # Skip metadata files
            if 'metadata' in filepath.name.lower():
                continue
            
            cycles, metadata = self._parse_sandia_csv(filepath)
            
            if not cycles:
                continue
            
            cell_id = f"SNL_{filepath.stem}"
            chemistry = metadata.get('chemistry', self.default_chemistry)
            temperature = metadata.get('temperature', self.default_temperature)
            c_rate = metadata.get('c_rate', 1.0)
            
            cell = CellData(
                cell_id=cell_id,
                source_dataset=self.dataset_name,
                chemistry=chemistry,
                nominal_capacity=self.nominal_capacity,
                nominal_voltage=3.7,
                form_factor="18650",
                test_temperature=temperature,
                charge_rate=c_rate,
                discharge_rate=c_rate,
                usage_profile="constant_current",
                cycles=cycles
            )
            
            cells[cell_id] = cell
        
        return cells
    
    def _create_context(self, cell: CellData) -> ExtendedBatteryContext:
        """Create context for a Sandia cell."""
        return create_sandia_context(
            cell_id=cell.cell_id,
            temperature_c=cell.test_temperature,
            chemistry=cell.chemistry,
            c_rate=cell.discharge_rate
        )
    
    def get_cells_by_temperature(self, temperature: float, tolerance: float = 5.0) -> List[CellData]:
        """Get cells tested at a specific temperature."""
        if not self._loaded:
            self.load()
        
        return [
            cell for cell in self.cells.values()
            if abs(cell.test_temperature - temperature) <= tolerance
        ]
    
    def get_cells_by_chemistry(self, chemistry: str) -> List[CellData]:
        """Get cells of a specific chemistry."""
        if not self._loaded:
            self.load()
        
        chemistry = chemistry.upper()
        return [
            cell for cell in self.cells.values()
            if chemistry in cell.chemistry.upper()
        ]
    
    def get_temperature_distribution(self) -> Dict[float, int]:
        """Get distribution of cells by temperature."""
        if not self._loaded:
            self.load()
        
        dist = {}
        for cell in self.cells.values():
            temp = round(cell.test_temperature)
            dist[temp] = dist.get(temp, 0) + 1
        
        return dict(sorted(dist.items()))


def create_synthetic_sandia_data(output_dir: str, num_cells: int = 20):
    """
    Create synthetic Sandia-like data for testing.
    
    This generates realistic-looking battery data with:
    - Multiple temperatures (15°C, 25°C, 35°C, 45°C)
    - Multiple C-rates (0.5C, 1C, 2C)
    - Realistic degradation patterns
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    temperatures = [15, 25, 35, 45]
    c_rates = [0.5, 1.0, 2.0]
    chemistries = ['LCO', 'NMC', 'LFP']
    
    nominal_capacity = 2.5  # Ah
    
    for i in range(num_cells):
        # Random conditions
        temp = np.random.choice(temperatures)
        c_rate = np.random.choice(c_rates)
        chemistry = np.random.choice(chemistries)
        
        # Generate degradation curve
        # Higher temp and C-rate = faster degradation
        temp_factor = 1 + 0.02 * (temp - 25)  # 2% faster per degree above 25
        crate_factor = 1 + 0.1 * (c_rate - 1)  # 10% faster per C above 1C
        degradation_rate = 0.0005 * temp_factor * crate_factor
        
        num_cycles = np.random.randint(500, 1500)
        
        cycles = []
        for cycle in range(num_cycles):
            # Capacity fade with noise
            capacity = nominal_capacity * (1 - degradation_rate * cycle) + np.random.normal(0, 0.01)
            capacity = max(0.5, capacity)
            
            # IR growth
            ir = 0.05 * (1 + 0.001 * cycle * temp_factor) + np.random.normal(0, 0.001)
            
            cycles.append({
                'cycle_index': cycle,
                'discharge_capacity': capacity,
                'internal_resistance': ir,
                'temperature': temp + np.random.normal(0, 1),
                'voltage_min': 2.5 + np.random.normal(0, 0.05),
                'voltage_max': 4.2 + np.random.normal(0, 0.02),
                'current_max': c_rate * nominal_capacity,
                'time_charge': 3600 / c_rate,
                'time_discharge': 3600 / c_rate,
            })
        
        # Save to CSV
        df = pd.DataFrame(cycles)
        filename = f"cell_{i:03d}_{chemistry}_{temp}C_{c_rate}C.csv"
        df.to_csv(output_path / filename, index=False)
    
    print(f"Created {num_cells} synthetic Sandia cells in {output_dir}")


if __name__ == '__main__':
    # Test with synthetic data
    test_dir = Path("/tmp/sandia_test")
    create_synthetic_sandia_data(str(test_dir), num_cells=10)
    
    # Load and verify
    loader = SandiaLoader(str(test_dir), use_cache=False)
    cells = loader.load()
    
    print(f"\nLoaded {len(cells)} cells")
    print(f"Statistics: {loader.get_statistics()}")
    print(f"Temperature distribution: {loader.get_temperature_distribution()}")
    
    # Test filtering
    hot_cells = loader.get_cells_by_temperature(45.0)
    print(f"\nCells at 45°C: {len(hot_cells)}")
    
    nmc_cells = loader.get_cells_by_chemistry('NMC')
    print(f"NMC cells: {len(nmc_cells)}")

