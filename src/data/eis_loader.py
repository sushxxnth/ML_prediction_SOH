"""
EIS Impedance Data Loader for Multi-Modal Fusion.

Loads EIS spectra from Impedance_* directories and matches to PLN cells.
"""

import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class EISSpectrum:
    """Single EIS measurement."""
    cell_id: str
    temperature: float
    soc: float
    storage_period: str  # "3W", "3M", etc.
    frequencies: np.ndarray  # (N,)
    z_real: np.ndarray      # (N,)
    z_imag: np.ndarray      # (N,)
    z_magnitude: np.ndarray # (N,)
    z_phase: np.ndarray     # (N,)
    
    def to_feature_vector(self, n_points: int = 34) -> np.ndarray:
        """Convert to fixed-size feature vector for model input."""
        # Stack Z_real and Z_imag (most informative for degradation)
        # Shape: (n_points, 2) -> flatten to (2*n_points,)
        z_real = self._resample(self.z_real, n_points)
        z_imag = self._resample(self.z_imag, n_points)
        
        # Normalize
        z_real = (z_real - z_real.mean()) / (z_real.std() + 1e-8)
        z_imag = (z_imag - z_imag.mean()) / (z_imag.std() + 1e-8)
        
        return np.concatenate([z_real, z_imag]).astype(np.float32)
    
    def to_nyquist_array(self, n_points: int = 34) -> np.ndarray:
        """Return (n_points, 4) array for CNN input."""
        z_real = self._resample(self.z_real, n_points)
        z_imag = self._resample(self.z_imag, n_points)
        z_mag = self._resample(self.z_magnitude, n_points)
        z_phase = self._resample(self.z_phase, n_points)
        
        return np.stack([z_real, z_imag, z_mag, z_phase], axis=-1).astype(np.float32)
    
    def _resample(self, arr: np.ndarray, n_points: int) -> np.ndarray:
        """Resample to fixed number of points."""
        if len(arr) == n_points:
            return arr
        indices = np.linspace(0, len(arr) - 1, n_points).astype(int)
        return arr[indices]


class EISLoader:
    """Load EIS impedance spectra from directory structure."""
    
    def __init__(self, data_root: str):
        self.data_root = Path(data_root)
        self.spectra: List[EISSpectrum] = []
        self.by_cell: Dict[str, List[EISSpectrum]] = {}
    
    def load(self) -> List[EISSpectrum]:
        """Load all EIS spectra."""
        impedance_dirs = list(self.data_root.glob("Impedance_*"))
        
        if not impedance_dirs:
            print(f"[WARN] No Impedance_* directories found in {self.data_root}")
            return []
        
        for temp_dir in impedance_dirs:
            temp = self._parse_temperature(temp_dir.name)
            
            # Find all CSV files
            for csv_file in temp_dir.rglob("*.csv"):
                try:
                    spectrum = self._load_spectrum(csv_file, temp)
                    if spectrum:
                        self.spectra.append(spectrum)
                        
                        # Index by cell
                        if spectrum.cell_id not in self.by_cell:
                            self.by_cell[spectrum.cell_id] = []
                        self.by_cell[spectrum.cell_id].append(spectrum)
                except Exception as e:
                    print(f"[WARN] Failed to load {csv_file}: {e}")
        
        print(f"Loaded {len(self.spectra)} EIS spectra from {len(self.by_cell)} cells")
        return self.spectra
    
    def _parse_temperature(self, dir_name: str) -> float:
        """Extract temperature from directory name like 'Impedance_25C' or 'Impedance_-40C'."""
        match = re.search(r'Impedance_(-?\d+)C', dir_name)
        if match:
            return float(match.group(1))
        return 25.0
    
    def _load_spectrum(self, csv_path: Path, temperature: float) -> Optional[EISSpectrum]:
        """Load single spectrum from CSV."""
        # Parse filename: "3_18_2015_25C_50SOC_PLN52.csv"
        filename = csv_path.stem
        
        # Extract cell ID (PLN##)
        cell_match = re.search(r'PLN(\d+)', filename)
        if not cell_match:
            return None
        cell_id = f"PLN{cell_match.group(1)}"
        
        # Extract SOC
        soc_match = re.search(r'(\d+)SOC', filename)
        soc = float(soc_match.group(1)) if soc_match else 50.0
        
        # Extract storage period from path (e.g., "3W", "3M")
        storage_period = "3W"
        for part in csv_path.parts:
            if part in ["3W", "3M", "6M"]:
                storage_period = part
                break
        
        # Load data (no header)
        try:
            data = np.loadtxt(csv_path, delimiter=',')
        except:
            data = pd.read_csv(csv_path, header=None).values
        
        if data.shape[1] < 5:
            return None
        
        return EISSpectrum(
            cell_id=cell_id,
            temperature=temperature,
            soc=soc,
            storage_period=storage_period,
            frequencies=data[:, 0],
            z_real=data[:, 1],
            z_imag=data[:, 2],
            z_magnitude=data[:, 3],
            z_phase=data[:, 4]
        )
    
    def get_for_cell(self, cell_id: str) -> List[EISSpectrum]:
        """Get all spectra for a cell."""
        # Handle different naming conventions
        if cell_id.startswith("PLN"):
            return self.by_cell.get(cell_id, [])
        
        # Try to extract PLN number from cell_id
        match = re.search(r'(\d+)', cell_id)
        if match:
            pln_id = f"PLN{match.group(1)}"
            return self.by_cell.get(pln_id, [])
        
        return []
    
    def get_statistics(self) -> Dict:
        """Get dataset statistics."""
        if not self.spectra:
            return {}
        
        temps = [s.temperature for s in self.spectra]
        socs = [s.soc for s in self.spectra]
        
        return {
            'n_spectra': len(self.spectra),
            'n_cells': len(self.by_cell),
            'temperatures': sorted(set(temps)),
            'socs': sorted(set(socs)),
            'freq_points': len(self.spectra[0].frequencies) if self.spectra else 0
        }


def extract_eis_features(spectrum: EISSpectrum) -> np.ndarray:
    """
    Extract physics-based features from EIS spectrum.
    
    Returns 8D feature vector:
    - R_ohmic: High-frequency intercept (electrolyte resistance)
    - R_ct: Charge transfer resistance (semi-circle diameter)
    - Z_warburg: Low-frequency impedance (diffusion)
    - Phase_max: Maximum phase angle
    - Freq_peak: Frequency at maximum -Z_imag
    - Z_real_slope: Slope of real part vs frequency
    - Z_imag_min: Minimum imaginary impedance
    - Nyquist_area: Approximate area under Nyquist curve
    """
    z_real = spectrum.z_real
    z_imag = spectrum.z_imag
    freq = spectrum.frequencies
    
    # R_ohmic: Z_real at highest frequency
    r_ohmic = z_real[0] if len(z_real) > 0 else 0
    
    # R_ct: Approximate as max Z_real - min Z_real
    r_ct = np.max(z_real) - np.min(z_real) if len(z_real) > 0 else 0
    
    # Z_warburg: Z_real at lowest frequency
    z_warburg = z_real[-1] if len(z_real) > 0 else 0
    
    # Phase_max
    phase_max = np.max(np.abs(spectrum.z_phase)) if len(spectrum.z_phase) > 0 else 0
    
    # Freq_peak: frequency at minimum Z_imag (most negative)
    if len(z_imag) > 0:
        peak_idx = np.argmin(z_imag)
        freq_peak = np.log10(freq[peak_idx] + 1e-6)
    else:
        freq_peak = 0
    
    # Z_real slope (high to low freq)
    if len(z_real) > 1:
        z_real_slope = (z_real[-1] - z_real[0]) / (len(z_real) - 1)
    else:
        z_real_slope = 0
    
    # Z_imag min
    z_imag_min = np.min(z_imag) if len(z_imag) > 0 else 0
    
    # Approximate Nyquist area (trapezoidal)
    if len(z_real) > 1 and len(z_imag) > 1:
        nyquist_area = np.trapz(-z_imag, z_real)
    else:
        nyquist_area = 0
    
    return np.array([
        r_ohmic,
        r_ct,
        z_warburg,
        phase_max,
        freq_peak,
        z_real_slope,
        z_imag_min,
        nyquist_area
    ], dtype=np.float32)


if __name__ == '__main__':
    # Test the loader
    loader = EISLoader('.')
    spectra = loader.load()
    
    print("\nStatistics:")
    stats = loader.get_statistics()
    for k, v in stats.items():
        print(f"  {k}: {v}")
    
    if spectra:
        print(f"\nSample spectrum: {spectra[0].cell_id}")
        print(f"  Frequencies: {spectra[0].frequencies[:5]}...")
        print(f"  Z_real: {spectra[0].z_real[:5]}...")
        
        features = extract_eis_features(spectra[0])
        print(f"  Extracted features (8D): {features}")
