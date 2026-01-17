"""
EIS Impedance Data Loader for PLN Storage Degradation Dataset

Parses Electrochemical Impedance Spectroscopy (EIS) data from CSV files.
This loader enables multi-modal learning (Capacity + Impedance).

Key Features:
- Parses EIS spectra (35 frequency points, 0.0125 Hz to 1638 Hz)
- Extracts equivalent circuit parameters (R0, Rct, Warburg)
- Supports 4 temperatures: -40°C, -5°C, 25°C, 50°C
- Supports 3 SOC levels: 0%, 50%, 100%
- Supports multiple storage periods (3W, 3M, 6M)

Author: Battery ML Research
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import optimize


@dataclass
class EISSpectrum:
    """Single EIS measurement spectrum."""
    cell_id: str
    temperature: float
    soc: float
    storage_period: str  # '3W', '3M', '6M'
    measurement_date: str
    
    # Raw spectrum data
    frequency: np.ndarray  # Hz
    z_real: np.ndarray  # Ohm
    z_imag: np.ndarray  # Ohm (negative for capacitive)
    z_magnitude: np.ndarray  # Ohm
    z_phase: np.ndarray  # degrees
    
    # Extracted features (computed lazily)
    _features: Optional[Dict] = field(default=None, repr=False)
    
    @property
    def features(self) -> Dict[str, float]:
        """Extract EIS features for ML models."""
        if self._features is None:
            self._features = self._extract_features()
        return self._features
    
    def _extract_features(self) -> Dict[str, float]:
        """Extract equivalent circuit and statistical features."""
        features = {}
        
        # R0: High-frequency resistance (intercept with real axis)
        # Approximated by Z_real at highest frequency
        features['R0'] = float(self.z_real[-1])  # Highest freq first if sorted
        
        # Rct: Charge transfer resistance
        # Approximated by semicircle diameter in mid-frequency range
        max_z_imag_idx = np.argmin(self.z_imag)  # Most negative
        features['Rct_estimate'] = float(self.z_real[max_z_imag_idx] - features['R0'])
        
        # Warburg features (low frequency slope)
        # In Nyquist plot, Warburg shows as 45° line at low freq
        low_freq_mask = self.frequency < 1.0  # Below 1 Hz
        if np.sum(low_freq_mask) >= 2:
            low_z_real = self.z_real[low_freq_mask]
            low_z_imag = self.z_imag[low_freq_mask]
            # Warburg slope: dZ_imag/dZ_real
            if len(low_z_real) > 1:
                try:
                    slope = np.polyfit(low_z_real, low_z_imag, 1)[0]
                    features['warburg_slope'] = float(slope)
                except:
                    features['warburg_slope'] = np.nan
            else:
                features['warburg_slope'] = np.nan
        else:
            features['warburg_slope'] = np.nan
        
        # Statistical features
        features['z_real_mean'] = float(np.mean(self.z_real))
        features['z_real_std'] = float(np.std(self.z_real))
        features['z_imag_mean'] = float(np.mean(self.z_imag))
        features['z_imag_min'] = float(np.min(self.z_imag))  # Most capacitive
        features['z_mag_max'] = float(np.max(self.z_magnitude))
        features['z_mag_min'] = float(np.min(self.z_magnitude))
        features['phase_min'] = float(np.min(self.z_phase))
        features['phase_max'] = float(np.max(self.z_phase))
        
        # Frequency-specific features (at characteristic frequencies)
        for target_freq in [0.1, 1.0, 10.0, 100.0, 1000.0]:
            idx = np.argmin(np.abs(self.frequency - target_freq))
            features[f'z_real_{target_freq:.0f}Hz'] = float(self.z_real[idx])
            features[f'z_imag_{target_freq:.0f}Hz'] = float(self.z_imag[idx])
        
        return features
    
    def to_array(self) -> np.ndarray:
        """Convert spectrum to fixed-size array for neural networks."""
        # Stack [z_real, z_imag] as (n_freq, 2) array
        return np.stack([self.z_real, self.z_imag], axis=-1)


@dataclass 
class EISCell:
    """Collection of EIS spectra for a single cell over time."""
    cell_id: str
    temperature: float
    soc: float
    spectra: List[EISSpectrum] = field(default_factory=list)
    
    @property
    def n_measurements(self) -> int:
        return len(self.spectra)
    
    def get_feature_matrix(self) -> pd.DataFrame:
        """Get features for all measurements as DataFrame."""
        records = []
        for i, spec in enumerate(self.spectra):
            record = spec.features.copy()
            record['measurement_idx'] = i
            record['storage_period'] = spec.storage_period
            record['date'] = spec.measurement_date
            records.append(record)
        return pd.DataFrame(records)


class EISImpedanceLoader:
    """
    Loader for EIS impedance data from PLN storage degradation dataset.
    
    Directory structure:
        Impedance_{temp}C/
            3W/
                {temp}C_3W_{idx}/
                    {date}_{temp}C_{soc}SOC_{cell_id}.csv
            3M/
                ...
    """
    
    TEMPERATURE_DIRS = {
        'Impedance_-40C': -40.0,
        'Impedance_-5C': -5.0,
        'Impedance_25C': 25.0,
        'Impedance_50C': 50.0,
    }
    
    SOC_PATTERN = re.compile(r'(\d+)SOC')
    CELL_PATTERN = re.compile(r'PLN(\d+)')
    DATE_PATTERN = re.compile(r'^(\d+_\d+_\d+)')
    
    def __init__(self, data_dir: str, cache_dir: Optional[str] = None):
        """
        Args:
            data_dir: Root directory containing Impedance_* folders
            cache_dir: Directory for caching processed data
        """
        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir) if cache_dir else self.data_dir / 'cache'
        self.cells: Dict[str, EISCell] = {}
        self._spectra_list: List[EISSpectrum] = []
        
    def load(self, force_reload: bool = False) -> Dict[str, EISCell]:
        """Load all EIS data."""
        cache_path = self.cache_dir / 'eis_data.pkl'
        
        if not force_reload and cache_path.exists():
            print(f"Loading EIS data from cache: {cache_path}")
            import pickle
            with open(cache_path, 'rb') as f:
                data = pickle.load(f)
                self.cells = data['cells']
                self._spectra_list = data['spectra']
                return self.cells
        
        print("Parsing EIS data from CSV files...")
        self._parse_all()
        
        # Cache the results
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        import pickle
        with open(cache_path, 'wb') as f:
            pickle.dump({'cells': self.cells, 'spectra': self._spectra_list}, f)
        print(f"Cached EIS data to: {cache_path}")
        
        return self.cells
    
    def _parse_all(self):
        """Parse all temperature directories."""
        for temp_dir_name, temperature in self.TEMPERATURE_DIRS.items():
            temp_dir = self.data_dir / temp_dir_name
            if temp_dir.exists():
                self._parse_temperature_dir(temp_dir, temperature)
        
        print(f"Loaded {len(self._spectra_list)} EIS spectra from {len(self.cells)} cells")
    
    def _parse_temperature_dir(self, temp_dir: Path, temperature: float):
        """Parse all storage periods within a temperature directory."""
        for storage_period in ['3W', '3M', '6M']:
            period_dir = temp_dir / storage_period
            if period_dir.exists():
                self._parse_period_dir(period_dir, temperature, storage_period)
    
    def _parse_period_dir(self, period_dir: Path, temperature: float, storage_period: str):
        """Parse all measurement folders within a storage period."""
        for measurement_dir in period_dir.iterdir():
            if measurement_dir.is_dir():
                self._parse_measurement_dir(measurement_dir, temperature, storage_period)
    
    def _parse_measurement_dir(self, measurement_dir: Path, temperature: float, storage_period: str):
        """Parse all CSV files in a measurement directory."""
        for csv_file in measurement_dir.glob('*.csv'):
            try:
                spectrum = self._parse_csv(csv_file, temperature, storage_period)
                if spectrum is not None:
                    self._spectra_list.append(spectrum)
                    
                    # Add to cell collection
                    # Create unique cell key: PLN{id}_SOC{soc}_T{temp}
                    cell_key = f"{spectrum.cell_id}_SOC{spectrum.soc:.0f}_T{spectrum.temperature:.0f}"
                    
                    if cell_key not in self.cells:
                        self.cells[cell_key] = EISCell(
                            cell_id=spectrum.cell_id,
                            temperature=spectrum.temperature,
                            soc=spectrum.soc
                        )
                    self.cells[cell_key].spectra.append(spectrum)
            except Exception as e:
                print(f"Error parsing {csv_file}: {e}")
    
    def _parse_csv(self, csv_file: Path, temperature: float, storage_period: str) -> Optional[EISSpectrum]:
        """Parse a single EIS CSV file."""
        filename = csv_file.name
        
        # Extract metadata from filename
        # Format: {date}_{temp}C_{soc}SOC_{cell_id}.csv
        soc_match = self.SOC_PATTERN.search(filename)
        cell_match = self.CELL_PATTERN.search(filename)
        date_match = self.DATE_PATTERN.search(filename)
        
        if not soc_match or not cell_match:
            return None
        
        soc = float(soc_match.group(1))
        cell_id = f"PLN{cell_match.group(1)}"
        measurement_date = date_match.group(1) if date_match else ""
        
        # Read CSV (no header)
        # Format: frequency, z_real, z_imag, z_magnitude, z_phase
        data = pd.read_csv(csv_file, header=None)
        
        if data.shape[1] != 5:
            print(f"Unexpected columns in {csv_file}: {data.shape[1]}")
            return None
        
        data.columns = ['frequency', 'z_real', 'z_imag', 'z_magnitude', 'z_phase']
        
        # Sort by frequency (ascending)
        data = data.sort_values('frequency')
        
        return EISSpectrum(
            cell_id=cell_id,
            temperature=temperature,
            soc=soc,
            storage_period=storage_period,
            measurement_date=measurement_date,
            frequency=data['frequency'].values,
            z_real=data['z_real'].values,
            z_imag=data['z_imag'].values,
            z_magnitude=data['z_magnitude'].values,
            z_phase=data['z_phase'].values
        )
    
    def get_all_features(self) -> pd.DataFrame:
        """Get extracted features for all spectra as a DataFrame."""
        records = []
        for spec in self._spectra_list:
            record = spec.features.copy()
            record['cell_id'] = spec.cell_id
            record['temperature'] = spec.temperature
            record['soc'] = spec.soc
            record['storage_period'] = spec.storage_period
            record['date'] = spec.measurement_date
            records.append(record)
        return pd.DataFrame(records)
    
    def get_spectra_array(self, n_freq: int = 35) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Get all spectra as a 3D array for deep learning.
        
        Returns:
            spectra: Array of shape (n_samples, n_freq, 2) with [z_real, z_imag]
            metadata: DataFrame with cell_id, temperature, soc, storage_period
        """
        spectra_list = []
        metadata_records = []
        
        for spec in self._spectra_list:
            arr = spec.to_array()
            if len(arr) == n_freq:
                spectra_list.append(arr)
                metadata_records.append({
                    'cell_id': spec.cell_id,
                    'temperature': spec.temperature,
                    'soc': spec.soc,
                    'storage_period': spec.storage_period,
                    'date': spec.measurement_date
                })
        
        return np.array(spectra_list), pd.DataFrame(metadata_records)
    
    def get_statistics(self) -> Dict:
        """Get dataset statistics."""
        df = self.get_all_features()
        return {
            'total_spectra': len(self._spectra_list),
            'unique_cells': len(self.cells),
            'temperatures': sorted(df['temperature'].unique().tolist()),
            'soc_levels': sorted(df['soc'].unique().tolist()),
            'storage_periods': df['storage_period'].unique().tolist(),
            'cells_per_temperature': df.groupby('temperature')['cell_id'].nunique().to_dict(),
        }


def load_pln_eis_data(data_dir: str, cache_dir: Optional[str] = None) -> EISImpedanceLoader:
    """Convenience function to load PLN EIS data."""
    loader = EISImpedanceLoader(data_dir, cache_dir)
    loader.load()
    return loader


if __name__ == "__main__":
    # Test the loader
    import sys
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    
    loader = load_pln_eis_data(data_dir)
    print("\n=== EIS Data Statistics ===")
    stats = loader.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n=== Sample Features ===")
    features_df = loader.get_all_features()
    print(features_df.head())
