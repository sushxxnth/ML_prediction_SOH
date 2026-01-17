"""
Multi-Modal Battery Data Loader

Combines Capacity and EIS Impedance data for SOC-aware degradation modeling.
This is the core data pipeline for the 4 novel contributions:
1. SOC-aware degradation modeling
2. Unified cycling + storage degradation  
3. Multi-modal feature fusion
4. Extreme temperature generalization

Author: Battery ML Research
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .eis_impedance_loader import EISImpedanceLoader, EISSpectrum, EISCell
from .storage_degradation_loader import StorageDegradationLoader
from .base_loader import CycleData, CellData


@dataclass
class MultiModalSample:
    """
    A single sample combining capacity and EIS data.
    
    Used for multi-modal fusion learning.
    """
    # Identifiers
    cell_id: str
    sample_idx: int
    
    # Condition context (5D)
    temperature: float  # Celsius
    soc: float  # Percentage (0, 50, 100)
    storage_period: str  # '3W', '3M', '6M'
    chemistry: str  # 'LCO', 'NMC', etc.
    usage_profile: str  # 'storage', 'cycling'
    
    # Capacity features
    capacity: float  # Ah
    soh_capacity: float  # 0-1
    capacity_fade_rate: Optional[float] = None  # Ah/month
    
    # EIS features (extracted)
    eis_features: Optional[Dict[str, float]] = None
    
    # EIS raw spectrum (for deep learning)
    eis_spectrum: Optional[np.ndarray] = None  # Shape: (n_freq, 2)
    
    # Labels
    rul_months: Optional[float] = None
    
    def to_5d_context(self) -> np.ndarray:
        """
        Convert to 5D context vector for model input.
        
        Returns:
            Array of shape (5,): [temp_norm, soc_norm, chemistry_id, usage_id, storage_months]
        """
        # Normalize temperature: -40 to 50 -> 0 to 1
        temp_norm = (self.temperature + 40) / 90.0
        
        # Normalize SOC: 0 to 100 -> 0 to 1
        soc_norm = self.soc / 100.0
        
        # Chemistry encoding
        chemistry_map = {'LCO': 0, 'NMC': 1, 'LFP': 2, 'NCA': 3}
        chemistry_id = chemistry_map.get(self.chemistry, 0)
        
        # Usage profile encoding
        usage_map = {'storage': 0, 'cycling': 1, 'mixed': 2}
        usage_id = usage_map.get(self.usage_profile, 0)
        
        # Storage period to months
        period_map = {'3W': 0.75, '3M': 3.0, '6M': 6.0}
        storage_months = period_map.get(self.storage_period, 0.0) / 6.0  # Normalize by max
        
        return np.array([temp_norm, soc_norm, chemistry_id / 3.0, usage_id / 2.0, storage_months])
    
    def get_capacity_features(self) -> np.ndarray:
        """Get capacity-related features."""
        return np.array([
            self.capacity,
            self.soh_capacity,
            self.capacity_fade_rate if self.capacity_fade_rate else 0.0
        ])
    
    def get_eis_feature_vector(self) -> Optional[np.ndarray]:
        """Get EIS features as array."""
        if self.eis_features is None:
            return None
        
        # Select key features in consistent order
        feature_keys = [
            'R0', 'Rct_estimate', 'warburg_slope',
            'z_real_mean', 'z_imag_mean', 'z_imag_min',
            'z_real_1Hz', 'z_imag_1Hz',
            'z_real_100Hz', 'z_imag_100Hz'
        ]
        
        return np.array([self.eis_features.get(k, 0.0) for k in feature_keys])


class MultiModalLoader:
    """
    Loader that combines Capacity and EIS data for multi-modal learning.
    
    This enables:
    - SOC-aware degradation modeling (Contribution 1)
    - Multi-modal feature fusion (Contribution 3)
    - Extreme temperature coverage (Contribution 4)
    """
    
    def __init__(
        self,
        data_dir: str,
        cache_dir: Optional[str] = None,
        chemistry: str = "LCO"
    ):
        """
        Args:
            data_dir: Root directory containing PLN data folders
            cache_dir: Directory for caching
            chemistry: Battery chemistry type
        """
        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir) if cache_dir else self.data_dir / 'cache'
        self.chemistry = chemistry
        
        # Sub-loaders
        self.eis_loader: Optional[EISImpedanceLoader] = None
        self.capacity_loader: Optional[StorageDegradationLoader] = None
        
        # Combined samples
        self.samples: List[MultiModalSample] = []
        
    def load(self, force_reload: bool = False) -> List[MultiModalSample]:
        """Load and combine all data sources."""
        cache_path = self.cache_dir / 'multimodal_samples.pkl'
        
        if not force_reload and cache_path.exists():
            print(f"Loading multi-modal data from cache: {cache_path}")
            import pickle
            with open(cache_path, 'rb') as f:
                self.samples = pickle.load(f)
            return self.samples
        
        # Load EIS data
        print("Loading EIS impedance data...")
        self.eis_loader = EISImpedanceLoader(str(self.data_dir), str(self.cache_dir))
        self.eis_loader.load(force_reload=force_reload)
        
        # Load capacity data (from PLN metadata Excel)
        print("Loading capacity data...")
        pln_metadata_dir = self.data_dir / 'PLN_Number_SOC_Temp_StoragePeriod'
        if pln_metadata_dir.exists():
            self.capacity_loader = StorageDegradationLoader(
                str(pln_metadata_dir),
                cache_dir=str(self.cache_dir),
                use_cache=not force_reload
            )
            self.capacity_loader.load(force_reload=force_reload)
        
        # Combine data sources
        self._create_multimodal_samples()
        
        # Cache results
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        import pickle
        with open(cache_path, 'wb') as f:
            pickle.dump(self.samples, f)
        print(f"Cached {len(self.samples)} multi-modal samples to: {cache_path}")
        
        return self.samples
    
    def _create_multimodal_samples(self):
        """Match EIS spectra with capacity data to create multi-modal samples."""
        self.samples = []
        
        # Create samples from EIS data (primary source)
        for cell_key, eis_cell in self.eis_loader.cells.items():
            for i, spectrum in enumerate(eis_cell.spectra):
                sample = MultiModalSample(
                    cell_id=spectrum.cell_id,
                    sample_idx=i,
                    temperature=spectrum.temperature,
                    soc=spectrum.soc,
                    storage_period=spectrum.storage_period,
                    chemistry=self.chemistry,
                    usage_profile='storage',
                    capacity=np.nan,  # Will be filled if available
                    soh_capacity=np.nan,
                    eis_features=spectrum.features,
                    eis_spectrum=spectrum.to_array()
                )
                self.samples.append(sample)
        
        # Match with capacity data if available
        if self.capacity_loader is not None:
            self._match_capacity_data()
        
        print(f"Created {len(self.samples)} multi-modal samples")
    
    def _match_capacity_data(self):
        """Match EIS samples with capacity measurements."""
        # Get capacity data as DataFrame
        capacity_df = self.capacity_loader.get_combined_dataframe()
        
        if capacity_df is None or len(capacity_df) == 0:
            print("No capacity data available for matching")
            return
        
        # Create lookup from capacity data
        # Group by temperature and SOC for approximate matching
        for sample in self.samples:
            # Find matching capacity measurements
            mask = (
                (capacity_df['temperature_mean'].abs() - abs(sample.temperature)).abs() < 5
            )
            matching = capacity_df[mask]
            
            if len(matching) > 0:
                # Use average capacity for this condition
                sample.capacity = float(matching['capacity'].mean())
                sample.soh_capacity = float(matching['soh_capacity'].mean())
    
    def get_dataframe(self) -> pd.DataFrame:
        """Get all samples as a DataFrame."""
        records = []
        for s in self.samples:
            record = {
                'cell_id': s.cell_id,
                'sample_idx': s.sample_idx,
                'temperature': s.temperature,
                'soc': s.soc,
                'storage_period': s.storage_period,
                'chemistry': s.chemistry,
                'capacity': s.capacity,
                'soh_capacity': s.soh_capacity,
            }
            # Add EIS features
            if s.eis_features:
                for k, v in s.eis_features.items():
                    record[f'eis_{k}'] = v
            records.append(record)
        return pd.DataFrame(records)
    
    def get_train_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get training data arrays.
        
        Returns:
            context: 5D context vectors (n_samples, 5)
            eis_features: EIS feature vectors (n_samples, n_eis_features)
            spectra: Raw EIS spectra (n_samples, n_freq, 2)
        """
        contexts = []
        eis_features = []
        spectra = []
        
        for sample in self.samples:
            contexts.append(sample.to_5d_context())
            
            eis_feat = sample.get_eis_feature_vector()
            if eis_feat is not None:
                eis_features.append(eis_feat)
            else:
                eis_features.append(np.zeros(10))
            
            if sample.eis_spectrum is not None:
                spectra.append(sample.eis_spectrum)
            else:
                spectra.append(np.zeros((35, 2)))
        
        return np.array(contexts), np.array(eis_features), np.array(spectra)
    
    def get_statistics(self) -> Dict:
        """Get dataset statistics."""
        df = self.get_dataframe()
        return {
            'total_samples': len(self.samples),
            'temperatures': sorted(df['temperature'].unique().tolist()),
            'soc_levels': sorted(df['soc'].unique().tolist()),
            'storage_periods': df['storage_period'].unique().tolist(),
            'cells': df['cell_id'].nunique(),
            'samples_per_temp': df.groupby('temperature').size().to_dict(),
            'samples_per_soc': df.groupby('soc').size().to_dict(),
        }
    
    def get_soc_grouped_data(self) -> Dict[float, List[MultiModalSample]]:
        """Group samples by SOC level for SOC-aware analysis."""
        grouped = {}
        for sample in self.samples:
            if sample.soc not in grouped:
                grouped[sample.soc] = []
            grouped[sample.soc].append(sample)
        return grouped
    
    def get_temperature_grouped_data(self) -> Dict[float, List[MultiModalSample]]:
        """Group samples by temperature for cross-temperature analysis."""
        grouped = {}
        for sample in self.samples:
            if sample.temperature not in grouped:
                grouped[sample.temperature] = []
            grouped[sample.temperature].append(sample)
        return grouped


def load_multimodal_data(data_dir: str, cache_dir: Optional[str] = None) -> MultiModalLoader:
    """Convenience function to load multi-modal PLN data."""
    loader = MultiModalLoader(data_dir, cache_dir)
    loader.load()
    return loader


if __name__ == "__main__":
    import sys
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    
    loader = load_multimodal_data(data_dir)
    print("\n=== Multi-Modal Data Statistics ===")
    stats = loader.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n=== 5D Context Sample ===")
    if loader.samples:
        sample = loader.samples[0]
        print(f"  Sample: {sample.cell_id}")
        print(f"  5D Context: {sample.to_5d_context()}")
        print(f"  EIS Features: {sample.get_eis_feature_vector()}")
