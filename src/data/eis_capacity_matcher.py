"""
EIS-Capacity Data Matcher

Joins EIS impedance spectra with capacity measurements using:
- PLN cell number
- Temperature
- SOC
- Storage period (3W, 3M)

Uses PLN_Number_SOC_Temp_StoragePeriod.xlsx as the mapping file.
"""

import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from dataclasses import dataclass

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.eis_loader import EISLoader, EISSpectrum, extract_eis_features


@dataclass
class MatchedSample:
    """Sample with matched EIS and capacity data."""
    cell_id: str
    pln_number: int
    temperature: float
    soc: float
    storage_period: str
    
    # EIS data
    eis_spectrum: np.ndarray      # (34, 4)
    eis_features: np.ndarray      # (8,) extracted features
    
    # Capacity data (from mapping file)
    discharge_capacity: float     # Ah
    
    # Derived labels
    soh: float                    # Capacity / Initial capacity
    rul_normalized: float         # Remaining life fraction


class EISCapacityMatcher:
    """Match EIS spectra with capacity measurements."""
    
    def __init__(self, data_root: str):
        self.data_root = Path(data_root)
        self.eis_loader = EISLoader(data_root)
        self.capacity_df = None
        self.matched_samples: List[MatchedSample] = []
    
    def load_capacity_mapping(self) -> pd.DataFrame:
        """Load the PLN mapping file with capacity measurements."""
        mapping_file = self.data_root / 'PLN_Number_SOC_Temp_StoragePeriod' / 'PLN_Number_SOC_Temp_StoragePeriod.xlsx'
        
        if not mapping_file.exists():
            raise FileNotFoundError(f"Mapping file not found: {mapping_file}")
        
        df = pd.read_excel(mapping_file)
        
        # Clean column names
        df.columns = ['PLN', 'SOC', 'TEMP', 'Time', 'Discharge_Capacity', 'Notes']
        
        # Filter out bad cells
        df = df[df['Notes'] != 'bad'].copy()
        df = df.dropna(subset=['Discharge_Capacity'])
        
        # Normalize storage period
        df['Time'] = df['Time'].astype(str).str.strip().str.upper()
        
        print(f"Loaded capacity data for {len(df)} PLN cells")
        self.capacity_df = df
        return df
    
    def match_samples(self) -> List[MatchedSample]:
        """Match EIS spectra with capacity measurements."""
        
        # Load both data sources
        self.eis_loader.load()
        if self.capacity_df is None:
            self.load_capacity_mapping()
        
        # Create lookup dictionary from capacity data
        # Key: (PLN_number, SOC, TEMP, Time)
        capacity_lookup = {}
        for _, row in self.capacity_df.iterrows():
            key = (int(row['PLN']), float(row['SOC']), float(row['TEMP']), str(row['Time']))
            capacity_lookup[key] = row['Discharge_Capacity']
        
        # Calculate initial capacity for SOH normalization (per PLN cell)
        initial_capacities = {}
        for _, row in self.capacity_df.iterrows():
            pln = int(row['PLN'])
            cap = row['Discharge_Capacity']
            if pln not in initial_capacities:
                initial_capacities[pln] = cap
            else:
                initial_capacities[pln] = max(initial_capacities[pln], cap)
        
        # Match EIS spectra
        matched = []
        unmatched = 0
        
        for spectrum in self.eis_loader.spectra:
            # Extract PLN number
            pln_match = re.search(r'PLN(\d+)', spectrum.cell_id)
            if not pln_match:
                unmatched += 1
                continue
            
            pln_number = int(pln_match.group(1))
            
            # Normalize storage period
            storage_period = spectrum.storage_period.upper().strip()
            
            # Create lookup key
            key = (pln_number, spectrum.soc, spectrum.temperature, storage_period)
            
            # Try to find matching capacity
            if key not in capacity_lookup:
                # Try without exact temperature match (closest)
                found = False
                for temp_offset in [0, 5, -5, 10, -10]:
                    alt_key = (pln_number, spectrum.soc, spectrum.temperature + temp_offset, storage_period)
                    if alt_key in capacity_lookup:
                        key = alt_key
                        found = True
                        break
                
                if not found:
                    unmatched += 1
                    continue
            
            discharge_capacity = capacity_lookup[key]
            initial_capacity = initial_capacities.get(pln_number, discharge_capacity)
            
            # Calculate SOH
            soh = discharge_capacity / initial_capacity if initial_capacity > 0 else 1.0
            soh = np.clip(soh, 0.5, 1.2)
            
            # Calculate RUL (based on position in storage sequence)
            # For 3W storage, assume early life; for 3M, assume later
            period_to_rul = {'3W': 0.8, '3M': 0.4, '6M': 0.2}
            rul_normalized = period_to_rul.get(storage_period, 0.5)
            
            sample = MatchedSample(
                cell_id=spectrum.cell_id,
                pln_number=pln_number,
                temperature=spectrum.temperature,
                soc=spectrum.soc,
                storage_period=storage_period,
                eis_spectrum=spectrum.to_nyquist_array(),
                eis_features=extract_eis_features(spectrum),
                discharge_capacity=float(discharge_capacity),
                soh=float(soh),
                rul_normalized=float(rul_normalized)
            )
            matched.append(sample)
        
        self.matched_samples = matched
        print(f"Matched {len(matched)} samples ({unmatched} unmatched)")
        return matched
    
    def get_statistics(self) -> Dict:
        """Get dataset statistics."""
        if not self.matched_samples:
            return {}
        
        temps = [s.temperature for s in self.matched_samples]
        socs = [s.soc for s in self.matched_samples]
        sohs = [s.soh for s in self.matched_samples]
        caps = [s.discharge_capacity for s in self.matched_samples]
        
        return {
            'n_samples': len(self.matched_samples),
            'n_cells': len(set(s.pln_number for s in self.matched_samples)),
            'temperatures': sorted(set(temps)),
            'socs': sorted(set(socs)),
            'soh_range': (min(sohs), max(sohs)),
            'capacity_range': (min(caps), max(caps))
        }
    
    def create_feature_matrix(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Create feature matrices for training.
        
        Returns:
            capacity_features: (N, 20) - capacity + context features
            eis_spectra: (N, 34, 4) - EIS Nyquist data
            soh_labels: (N,)
            rul_labels: (N,)
        """
        n = len(self.matched_samples)
        
        capacity_features = np.zeros((n, 20), dtype=np.float32)
        eis_spectra = np.zeros((n, 34, 4), dtype=np.float32)
        soh_labels = np.zeros(n, dtype=np.float32)
        rul_labels = np.zeros(n, dtype=np.float32)
        
        for i, sample in enumerate(self.matched_samples):
            # Capacity features from EIS + context
            eis_feat = sample.eis_features  # 8D
            
            # Build 20D feature vector
            feat = np.zeros(20, dtype=np.float32)
            feat[:8] = eis_feat                                    # EIS-derived features
            feat[8] = sample.discharge_capacity / 2.0              # Normalized capacity
            feat[9] = sample.temperature / 100                     # Normalized temp
            feat[10] = sample.soc / 100                            # Normalized SOC
            feat[11] = {'3W': 0.2, '3M': 0.5, '6M': 0.8}.get(sample.storage_period, 0.5)
            feat[12] = sample.soh                                  # SOH as feature
            feat[13:20] = np.random.randn(7) * 0.05                # Placeholder for lithium features
            
            capacity_features[i] = feat
            eis_spectra[i] = sample.eis_spectrum
            soh_labels[i] = sample.soh
            rul_labels[i] = sample.rul_normalized
        
        return capacity_features, eis_spectra, soh_labels, rul_labels


if __name__ == '__main__':
    matcher = EISCapacityMatcher('.')
    
    print("Loading and matching data...")
    samples = matcher.match_samples()
    
    print("\nStatistics:")
    stats = matcher.get_statistics()
    for k, v in stats.items():
        print(f"  {k}: {v}")
    
    if samples:
        print("\nSample matched record:")
        s = samples[0]
        print(f"  Cell: {s.cell_id}")
        print(f"  PLN: {s.pln_number}")
        print(f"  Temp: {s.temperature}°C, SOC: {s.soc}%")
        print(f"  Storage: {s.storage_period}")
        print(f"  Capacity: {s.discharge_capacity:.4f} Ah")
        print(f"  SOH: {s.soh:.4f}")
        
        # Create feature matrices
        cap_feat, eis_spec, soh, rul = matcher.create_feature_matrix()
        print(f"\nFeature matrices:")
        print(f"  Capacity features: {cap_feat.shape}")
        print(f"  EIS spectra: {eis_spec.shape}")
        print(f"  SOH labels: {soh.shape}, range [{soh.min():.3f}, {soh.max():.3f}]")
