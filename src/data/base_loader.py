"""
Base Battery Data Loader

Abstract base class for all battery dataset loaders.
Provides a unified interface for loading, processing, and
converting battery data to the common training format.

Author: Battery ML Research
"""

import os
import json
import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Generator
from pathlib import Path
import numpy as np
import pandas as pd

# Import context system
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.context.extended_context import (
    ExtendedBatteryContext,
    TemperatureContext,
    ChemistryContext,
    UsageProfileContext,
    CRateContext
)


@dataclass
class CycleData:
    """
    Standardized cycle-level data format.
    
    All datasets are converted to this format for unified training.
    """
    # Identifiers
    cell_id: str
    cycle_index: int
    
    # Core measurements
    capacity: float              # Ah
    internal_resistance: float   # Ohms (can be np.nan)
    
    # SOH/RUL labels
    soh_capacity: float         # Capacity-based SOH (0-1)
    soh_resistance: float       # Resistance-based SOH (0-1)
    rul_cycles: int             # Remaining useful life in cycles
    
    # Per-cycle statistics
    temperature_mean: float     # Mean temperature during cycle
    temperature_max: float      # Max temperature
    temperature_min: float      # Min temperature
    
    current_mean: float         # Mean discharge current
    current_max: float          # Max current (for C-rate estimation)
    
    voltage_min: float          # Minimum voltage (cutoff)
    voltage_max: float          # Maximum voltage (charged)
    
    charge_time: float          # Charging duration (seconds)
    discharge_time: float       # Discharge duration (seconds)
    
    # Context
    context: ExtendedBatteryContext = None
    
    # Optional: raw time-series data
    time_series: Optional[pd.DataFrame] = None
    
    def to_feature_vector(self) -> np.ndarray:
        """Convert to feature vector for model input."""
        return np.array([
            self.cycle_index,
            self.capacity,
            self.internal_resistance if not np.isnan(self.internal_resistance) else 0,
            self.temperature_mean,
            self.current_mean,
            self.voltage_min,
            self.voltage_max,
            self.charge_time / 3600,  # Convert to hours
            self.discharge_time / 3600,
        ], dtype=np.float32)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'cell_id': self.cell_id,
            'cycle_index': self.cycle_index,
            'capacity': self.capacity,
            'internal_resistance': self.internal_resistance,
            'soh_capacity': self.soh_capacity,
            'soh_resistance': self.soh_resistance,
            'rul_cycles': self.rul_cycles,
            'temperature_mean': self.temperature_mean,
            'temperature_max': self.temperature_max,
            'temperature_min': self.temperature_min,
            'current_mean': self.current_mean,
            'current_max': self.current_max,
            'voltage_min': self.voltage_min,
            'voltage_max': self.voltage_max,
            'charge_time': self.charge_time,
            'discharge_time': self.discharge_time,
            'context': self.context.to_dict() if self.context else None
        }


@dataclass
class CellData:
    """
    Complete data for a single battery cell.
    """
    cell_id: str
    source_dataset: str
    
    # Cell metadata
    chemistry: str
    nominal_capacity: float  # Ah
    nominal_voltage: float   # V
    form_factor: str         # 18650, pouch, prismatic, coin
    
    # Test conditions
    test_temperature: float  # °C
    charge_rate: float       # C-rate
    discharge_rate: float    # C-rate
    usage_profile: str
    
    # Cycle data
    cycles: List[CycleData] = field(default_factory=list)
    
    # Computed metadata
    total_cycles: int = 0
    eol_cycle: int = 0
    initial_capacity: float = 0
    final_capacity: float = 0
    
    # Context
    context: ExtendedBatteryContext = None
    
    def compute_labels(self, eol_capacity_frac: float = 0.8, eol_ir_mult: float = 1.7):
        """Compute SOH and RUL labels for all cycles."""
        if not self.cycles:
            return
        
        # Get initial values
        capacities = [c.capacity for c in self.cycles if not np.isnan(c.capacity)]
        resistances = [c.internal_resistance for c in self.cycles 
                      if not np.isnan(c.internal_resistance)]
        
        if capacities:
            self.initial_capacity = np.median(capacities[:5])
            self.final_capacity = capacities[-1] if capacities else np.nan
        
        initial_ir = np.median(resistances[:5]) if len(resistances) >= 5 else (
            resistances[0] if resistances else np.nan
        )
        
        # Determine EoL
        eol_cap_threshold = eol_capacity_frac * self.initial_capacity
        eol_ir_threshold = eol_ir_mult * initial_ir if not np.isnan(initial_ir) else np.inf
        
        self.eol_cycle = len(self.cycles) - 1
        for i, cycle in enumerate(self.cycles):
            if cycle.capacity <= eol_cap_threshold:
                self.eol_cycle = i
                break
            if not np.isnan(cycle.internal_resistance) and cycle.internal_resistance >= eol_ir_threshold:
                self.eol_cycle = min(self.eol_cycle, i)
        
        self.total_cycles = len(self.cycles)
        
        # Compute SOH and RUL for each cycle
        for i, cycle in enumerate(self.cycles):
            # Capacity-based SOH
            if self.initial_capacity > 0:
                cycle.soh_capacity = cycle.capacity / self.initial_capacity
            else:
                cycle.soh_capacity = 1.0
            
            # Resistance-based SOH
            if not np.isnan(initial_ir) and not np.isnan(cycle.internal_resistance):
                cycle.soh_resistance = initial_ir / cycle.internal_resistance
            else:
                cycle.soh_resistance = np.nan
            
            # RUL
            cycle.rul_cycles = max(0, self.eol_cycle - i)
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert all cycles to DataFrame."""
        rows = [c.to_dict() for c in self.cycles]
        df = pd.DataFrame(rows)
        df['cell_id'] = self.cell_id
        df['source_dataset'] = self.source_dataset
        return df
    
    def get_feature_matrix(self) -> np.ndarray:
        """Get feature matrix for all cycles."""
        return np.stack([c.to_feature_vector() for c in self.cycles])


class BaseBatteryLoader(ABC):
    """
    Abstract base class for battery dataset loaders.
    
    Subclasses must implement:
    - _parse_raw_data(): Parse raw data files
    - _create_context(): Create context for a cell
    
    Provides:
    - Unified loading interface
    - Caching support
    - Data validation
    - Export to standard format
    """
    
    def __init__(
        self,
        data_dir: str,
        cache_dir: Optional[str] = None,
        use_cache: bool = True
    ):
        """
        Args:
            data_dir: Directory containing raw data files
            cache_dir: Directory for caching processed data
            use_cache: Whether to use cached data if available
        """
        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir) if cache_dir else self.data_dir / "processed"
        self.use_cache = use_cache
        
        self.cells: Dict[str, CellData] = {}
        self._loaded = False
    
    @property
    @abstractmethod
    def dataset_name(self) -> str:
        """Return the dataset name identifier."""
        pass
    
    @property
    @abstractmethod
    def default_chemistry(self) -> str:
        """Return the default chemistry for this dataset."""
        pass
    
    @property
    @abstractmethod
    def default_temperature(self) -> float:
        """Return the default test temperature for this dataset."""
        pass
    
    @abstractmethod
    def _parse_raw_data(self) -> Dict[str, CellData]:
        """
        Parse raw data files and return cell data.
        
        Returns:
            Dictionary mapping cell_id to CellData
        """
        pass
    
    @abstractmethod
    def _create_context(self, cell: CellData) -> ExtendedBatteryContext:
        """
        Create context for a cell based on its metadata.
        
        Args:
            cell: CellData instance
            
        Returns:
            ExtendedBatteryContext instance
        """
        pass
    
    def load(self, force_reload: bool = False) -> Dict[str, CellData]:
        """
        Load the dataset.
        
        Args:
            force_reload: If True, ignore cache and reload from raw files
            
        Returns:
            Dictionary mapping cell_id to CellData
        """
        if self._loaded and not force_reload:
            return self.cells
        
        # Check cache
        cache_path = self.cache_dir / f"{self.dataset_name}_processed.json"
        
        if self.use_cache and cache_path.exists() and not force_reload:
            print(f"Loading {self.dataset_name} from cache: {cache_path}")
            self.cells = self._load_from_cache(cache_path)
        else:
            print(f"Parsing {self.dataset_name} from raw files: {self.data_dir}")
            self.cells = self._parse_raw_data()
            
            # Compute labels and contexts
            for cell_id, cell in self.cells.items():
                cell.compute_labels()
                cell.context = self._create_context(cell)
                
                # Propagate context to cycles
                for cycle in cell.cycles:
                    cycle.context = cell.context
            
            # Save to cache
            if self.use_cache:
                self._save_to_cache(cache_path)
        
        self._loaded = True
        print(f"Loaded {len(self.cells)} cells from {self.dataset_name}")
        
        return self.cells
    
    def _save_to_cache(self, cache_path: Path):
        """Save processed data to cache."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        cache_data = {
            'dataset_name': self.dataset_name,
            'num_cells': len(self.cells),
            'cells': {}
        }
        
        for cell_id, cell in self.cells.items():
            cache_data['cells'][cell_id] = {
                'cell_id': cell.cell_id,
                'source_dataset': cell.source_dataset,
                'chemistry': cell.chemistry,
                'nominal_capacity': cell.nominal_capacity,
                'nominal_voltage': cell.nominal_voltage,
                'form_factor': cell.form_factor,
                'test_temperature': cell.test_temperature,
                'charge_rate': cell.charge_rate,
                'discharge_rate': cell.discharge_rate,
                'usage_profile': cell.usage_profile,
                'total_cycles': cell.total_cycles,
                'eol_cycle': cell.eol_cycle,
                'initial_capacity': cell.initial_capacity,
                'final_capacity': cell.final_capacity,
                'context': cell.context.to_dict() if cell.context else None,
                'cycles': [c.to_dict() for c in cell.cycles]
            }
        
        with open(cache_path, 'w') as f:
            json.dump(cache_data, f, indent=2, default=str)
        
        print(f"Cached {self.dataset_name} to {cache_path}")
    
    def _load_from_cache(self, cache_path: Path) -> Dict[str, CellData]:
        """Load processed data from cache."""
        with open(cache_path, 'r') as f:
            cache_data = json.load(f)
        
        cells = {}
        for cell_id, cell_dict in cache_data['cells'].items():
            # Reconstruct CellData
            cell = CellData(
                cell_id=cell_dict['cell_id'],
                source_dataset=cell_dict['source_dataset'],
                chemistry=cell_dict['chemistry'],
                nominal_capacity=cell_dict['nominal_capacity'],
                nominal_voltage=cell_dict['nominal_voltage'],
                form_factor=cell_dict['form_factor'],
                test_temperature=cell_dict['test_temperature'],
                charge_rate=cell_dict['charge_rate'],
                discharge_rate=cell_dict['discharge_rate'],
                usage_profile=cell_dict['usage_profile'],
                total_cycles=cell_dict['total_cycles'],
                eol_cycle=cell_dict['eol_cycle'],
                initial_capacity=cell_dict['initial_capacity'],
                final_capacity=cell_dict['final_capacity']
            )
            
            # Reconstruct context
            if cell_dict.get('context'):
                cell.context = ExtendedBatteryContext.from_dict(cell_dict['context'])
            
            # Reconstruct cycles
            for cycle_dict in cell_dict['cycles']:
                ctx = None
                if cycle_dict.get('context'):
                    ctx = ExtendedBatteryContext.from_dict(cycle_dict['context'])
                
                cycle = CycleData(
                    cell_id=cycle_dict['cell_id'],
                    cycle_index=cycle_dict['cycle_index'],
                    capacity=cycle_dict['capacity'],
                    internal_resistance=cycle_dict['internal_resistance'],
                    soh_capacity=cycle_dict['soh_capacity'],
                    soh_resistance=cycle_dict['soh_resistance'],
                    rul_cycles=cycle_dict['rul_cycles'],
                    temperature_mean=cycle_dict['temperature_mean'],
                    temperature_max=cycle_dict['temperature_max'],
                    temperature_min=cycle_dict['temperature_min'],
                    current_mean=cycle_dict['current_mean'],
                    current_max=cycle_dict['current_max'],
                    voltage_min=cycle_dict['voltage_min'],
                    voltage_max=cycle_dict['voltage_max'],
                    charge_time=cycle_dict['charge_time'],
                    discharge_time=cycle_dict['discharge_time'],
                    context=ctx
                )
                cell.cycles.append(cycle)
            
            cells[cell_id] = cell
        
        return cells
    
    def get_all_cycles(self) -> List[CycleData]:
        """Get all cycles from all cells as a flat list."""
        if not self._loaded:
            self.load()
        
        all_cycles = []
        for cell in self.cells.values():
            all_cycles.extend(cell.cycles)
        
        return all_cycles
    
    def get_combined_dataframe(self) -> pd.DataFrame:
        """Get all data as a single DataFrame."""
        if not self._loaded:
            self.load()
        
        dfs = [cell.to_dataframe() for cell in self.cells.values()]
        return pd.concat(dfs, ignore_index=True)
    
    def iter_cells(self) -> Generator[CellData, None, None]:
        """Iterate over all cells."""
        if not self._loaded:
            self.load()
        
        for cell in self.cells.values():
            yield cell
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        if not self._loaded:
            self.load()
        
        all_cycles = self.get_all_cycles()
        
        temperatures = [c.temperature_mean for c in all_cycles if not np.isnan(c.temperature_mean)]
        capacities = [c.capacity for c in all_cycles if not np.isnan(c.capacity)]
        
        return {
            'dataset_name': self.dataset_name,
            'num_cells': len(self.cells),
            'total_cycles': len(all_cycles),
            'chemistries': list(set(c.chemistry for c in self.cells.values())),
            'temperature_range': (min(temperatures), max(temperatures)) if temperatures else (None, None),
            'capacity_range': (min(capacities), max(capacities)) if capacities else (None, None),
            'avg_cycles_per_cell': len(all_cycles) / len(self.cells) if self.cells else 0,
        }
    
    def export_to_csv(self, output_path: str):
        """Export dataset to CSV."""
        df = self.get_combined_dataframe()
        df.to_csv(output_path, index=False)
        print(f"Exported {len(df)} cycles to {output_path}")


def validate_cycle_data(cycle: CycleData) -> List[str]:
    """Validate cycle data and return list of issues."""
    issues = []
    
    if np.isnan(cycle.capacity) or cycle.capacity <= 0:
        issues.append(f"Invalid capacity: {cycle.capacity}")
    
    if cycle.soh_capacity < 0 or cycle.soh_capacity > 1.5:
        issues.append(f"SOH out of range: {cycle.soh_capacity}")
    
    if cycle.rul_cycles < 0:
        issues.append(f"Negative RUL: {cycle.rul_cycles}")
    
    if cycle.voltage_min > cycle.voltage_max:
        issues.append(f"Voltage range inverted: {cycle.voltage_min} > {cycle.voltage_max}")
    
    return issues


if __name__ == '__main__':
    print("Base Battery Loader - Module Test")
    print("This is an abstract base class. Import and extend for specific datasets.")

