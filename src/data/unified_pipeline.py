"""
Unified Multi-Dataset Battery Data Pipeline

Combines data from multiple sources into a unified training pipeline:
- NASA (baseline, room temp)
- Sandia (temperature variations)
- CALCE (chemistry + profiles)
- Oxford (urban driving)
- TBSI Sunwoda (fast charging)

Key Features:
- Common data format across all datasets
- Rich context encoding for each sample
- Stratified sampling by context
- Cross-dataset validation splits

Author: Battery ML Research
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union, Generator
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split, StratifiedKFold

# Import loaders
from .base_loader import CellData, CycleData, BaseBatteryLoader
from .sandia_loader import SandiaLoader
from .calce_loader import CALCELoader
from .oxford_loader import OxfordLoader
from .tbsi_loader import TBSISunwodaLoader
from .randomwalk_loader import RandomizedBatteryLoader
from .panasonic_loader import Panasonic18650PFLoader
from .storage_degradation_loader import StorageDegradationLoader
from .xjtu_loader import XJTULoader
from .dataset_registry import get_dataset_info, list_datasets

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.context.extended_context import (
    ExtendedBatteryContext,
    TemperatureContext,
    ChemistryContext,
    UsageProfileContext,
    CRateContext,
    TOTAL_HYBRID_DIM,
    batch_contexts_to_tensor,
    create_nasa_context,
    normalize_temperature,
    normalize_crate,
    chemistry_to_id
)
from src.data.lithium_inventory_integration import (
    extract_lithium_features_for_cell,
    augment_cycle_with_lithium_features,
    get_augmented_feature_dim
)


@dataclass
class UnifiedSample:
    """
    A single sample from the unified dataset.
    
    Contains features, labels, and rich context for training.
    """
    # Identifiers
    cell_id: str
    cycle_idx: int
    source_dataset: str
    
    # Features (can be extended)
    features: np.ndarray  # Shape: (feature_dim,)
    
    # Labels
    soh: float           # State of Health (0-1)
    rul: int             # Remaining Useful Life (cycles) - absolute
    
    # Context
    context: ExtendedBatteryContext
    context_vector: np.ndarray  # Condensed numeric context (e.g., temp, charge/discharge rates)
    chem_id: int = 0            # Chemistry id for embedding
    
    # Normalized RUL (with defaults)
    rul_normalized: float = 0.0  # RUL as fraction of EOL (0-1)
    eol_cycle: int = 100  # End-of-life cycle number
    
    # Optional: sequence data for time-series models
    sequence: Optional[np.ndarray] = None  # Shape: (seq_len, feature_dim)
    
    def to_tensor_dict(self, device: str = 'cpu') -> Dict[str, torch.Tensor]:
        """Convert to tensor dictionary for model input."""
        d = {
            'features': torch.tensor(self.features, dtype=torch.float32, device=device),
            'soh': torch.tensor(self.soh, dtype=torch.float32, device=device),
            'rul': torch.tensor(self.rul, dtype=torch.float32, device=device),
            'rul_normalized': torch.tensor(self.rul_normalized, dtype=torch.float32, device=device),
            'eol_cycle': torch.tensor(self.eol_cycle, dtype=torch.float32, device=device),
            'context': torch.tensor(self.context_vector, dtype=torch.float32, device=device),
            'chem_id': torch.tensor(self.chem_id, dtype=torch.long, device=device),
        }
        if self.sequence is not None:
            d['sequence'] = torch.tensor(self.sequence, dtype=torch.float32, device=device)
        return d


class UnifiedBatteryDataset(Dataset):
    """
    PyTorch Dataset for unified multi-source battery data.
    
    Combines data from multiple sources with consistent formatting
    and rich context encoding.
    """
    
    def __init__(
        self,
        samples: List[UnifiedSample],
        sequence_length: int = 50,
        transform: Optional[callable] = None
    ):
        """
        Args:
            samples: List of UnifiedSample instances
            sequence_length: Length of sequences for time-series models
            transform: Optional transform function
        """
        self.samples = samples
        self.sequence_length = sequence_length
        self.transform = transform
        
        # Build index for efficient access
        self._build_index()
    
    def _build_index(self):
        """Build indices for stratified access."""
        self.by_dataset = {}
        self.by_temperature = {}
        self.by_chemistry = {}
        self.by_profile = {}
        
        for i, sample in enumerate(self.samples):
            # By dataset
            ds = sample.source_dataset
            if ds not in self.by_dataset:
                self.by_dataset[ds] = []
            self.by_dataset[ds].append(i)
            
            # By temperature
            temp = sample.context.temperature.name
            if temp not in self.by_temperature:
                self.by_temperature[temp] = []
            self.by_temperature[temp].append(i)
            
            # By chemistry
            chem = sample.context.chemistry.name
            if chem not in self.by_chemistry:
                self.by_chemistry[chem] = []
            self.by_chemistry[chem].append(i)
            
            # By profile
            profile = sample.context.usage_profile.name
            if profile not in self.by_profile:
                self.by_profile[profile] = []
            self.by_profile[profile].append(i)
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        
        item = sample.to_tensor_dict()
        
        if self.transform:
            item = self.transform(item)
        
        return item
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        sohs = [s.soh for s in self.samples]
        ruls = [s.rul for s in self.samples]
        
        return {
            'total_samples': len(self.samples),
            'by_dataset': {k: len(v) for k, v in self.by_dataset.items()},
            'by_temperature': {k: len(v) for k, v in self.by_temperature.items()},
            'by_chemistry': {k: len(v) for k, v in self.by_chemistry.items()},
            'by_profile': {k: len(v) for k, v in self.by_profile.items()},
            'soh_mean': float(np.mean(sohs)),
            'soh_std': float(np.std(sohs)),
            'rul_mean': float(np.mean(ruls)),
            'rul_std': float(np.std(ruls)),
            'unique_cells': len(set(s.cell_id for s in self.samples)),
        }


class UnifiedDataPipeline:
    """
    Main pipeline for loading and processing multi-source battery data.
    
    Usage:
        >>> pipeline = UnifiedDataPipeline(data_root='./data')
        >>> pipeline.load_datasets(['nasa', 'sandia', 'calce'])
        >>> train_ds, val_ds, test_ds = pipeline.create_splits()
    """
    
    # Loader registry
    LOADERS = {
        'sandia': SandiaLoader,
        'calce': CALCELoader,
        'oxford': OxfordLoader,
        'tbsi_sunwoda': TBSISunwodaLoader,
        'randomized_battery_usage': RandomizedBatteryLoader,
        'randomized': RandomizedBatteryLoader,
        'randomwalk': RandomizedBatteryLoader,
        'panasonic_18650pf': Panasonic18650PFLoader,
        'panasonic': Panasonic18650PFLoader,
        'storage_degradation': StorageDegradationLoader,
        'storage': StorageDegradationLoader,
        'storage_pln': StorageDegradationLoader,
        'xjtu': XJTULoader,
    }
    
    def __init__(
        self,
        data_root: str,
        cache_dir: Optional[str] = None,
        use_cache: bool = True,
        sequence_length: int = 50,
        feature_dim: int = 20,  # Updated: 9 original + 11 lithium inventory
        use_lithium_features: bool = True
    ):
        """
        Args:
            data_root: Root directory containing dataset subdirectories
            cache_dir: Directory for caching processed data
            use_cache: Whether to use cached data
            sequence_length: Length of sequences for time-series models
            feature_dim: Number of features per sample (20 with lithium features, 9 without)
            use_lithium_features: Whether to augment features with lithium inventory features
        """
        self.data_root = Path(data_root)
        self.cache_dir = Path(cache_dir) if cache_dir else self.data_root / "unified_cache"
        self.use_cache = use_cache
        self.sequence_length = sequence_length
        self.use_lithium_features = use_lithium_features
        self.feature_dim = get_augmented_feature_dim() if use_lithium_features else feature_dim
        
        self.loaders: Dict[str, BaseBatteryLoader] = {}
        self.cells: Dict[str, CellData] = {}
        self.samples: List[UnifiedSample] = []
        
        self._loaded = False
    
    def load_datasets(
        self,
        datasets: List[str],
        create_synthetic_if_missing: bool = True
    ) -> 'UnifiedDataPipeline':
        """
        Load specified datasets.
        
        Args:
            datasets: List of dataset names to load
            create_synthetic_if_missing: Create synthetic data if real data not found
        
        Returns:
            self for chaining
        """
        for ds_name in datasets:
            ds_name = ds_name.lower()
            
            if ds_name == 'nasa':
                # NASA uses custom loader (existing code)
                self._load_nasa()
                continue
            
            if ds_name not in self.LOADERS:
                print(f"[WARN] Unknown dataset: {ds_name}. Available: {list(self.LOADERS.keys())}")
                continue
            
            # Get data directory (with special cases)
            if ds_name == 'xjtu':
                ds_dir = self.data_root / "new_datasets" / "XJTU" / "Battery Dataset"
            else:
                ds_dir = self.data_root / ds_name
            
            if not ds_dir.exists() and create_synthetic_if_missing:
                print(f"Creating synthetic {ds_name} data...")
                ds_dir.mkdir(parents=True, exist_ok=True)
                self._create_synthetic(ds_name, str(ds_dir))

            
            if ds_dir.exists():
                # Create loader and load data
                loader_cls = self.LOADERS[ds_name]
                loader = loader_cls(
                    str(ds_dir),
                    cache_dir=str(self.cache_dir / ds_name),
                    use_cache=self.use_cache
                )
                
                cells = loader.load()
                self.loaders[ds_name] = loader
                self.cells.update(cells)
                
                print(f"Loaded {ds_name}: {len(cells)} cells")
            else:
                print(f"[WARN] No data found for {ds_name}")
        
        # Convert to unified samples
        self._create_samples()
        self._loaded = True
        
        return self
    
    def _load_nasa(self):
        """Load NASA dataset using existing code."""
        nasa_dir = self.data_root / "nasa_set5" / "raw"
        
        if not nasa_dir.exists():
            print(f"[WARN] NASA data not found at {nasa_dir}")
            return
        
        try:
            from src.data.nasa_set5 import load_all_cells
            
            nasa_cells = load_all_cells(str(nasa_dir))
            
            # Convert to our CellData format
            for cell_id, cell_data in nasa_cells.items():
                cycle_table = cell_data['cycle_table']
                meta = cell_data.get('meta', {})
                
                # Get ambient temperature
                ambient_temp = meta.get('ambient_temperature', 24.0)
                if isinstance(ambient_temp, str):
                    ambient_temp = 24.0
                
                cycles = []
                for _, row in cycle_table.iterrows():
                    cycle = CycleData(
                        cell_id=cell_id,
                        cycle_index=int(row.get('cycle_index', 0)),
                        capacity=float(row.get('Capacity', np.nan) or np.nan),
                        internal_resistance=float(row.get('IR', np.nan) or np.nan),
                        soh_capacity=float(row.get('SOH_Q', 1.0) or 1.0),
                        soh_resistance=float(row.get('SOH_R', np.nan) or np.nan),
                        rul_cycles=int(row.get('RUL_cycles', 0)),
                        temperature_mean=float(row.get('Temp_med', ambient_temp) or ambient_temp),
                        temperature_max=float(row.get('Temp_med', ambient_temp) or ambient_temp),
                        temperature_min=float(row.get('Temp_med', ambient_temp) or ambient_temp),
                        current_mean=float(row.get('Iabs_med_dis', 1.0) or 1.0),
                        current_max=float(row.get('Iabs_med_dis', 2.0) or 2.0),
                        voltage_min=2.5,
                        voltage_max=4.2,
                        charge_time=3600,
                        discharge_time=3600,
                    )
                    cycles.append(cycle)
                
                cell = CellData(
                    cell_id=f"NASA_{cell_id}",
                    source_dataset="nasa",
                    chemistry="LCO",
                    nominal_capacity=2.0,
                    nominal_voltage=3.7,
                    form_factor="18650",
                    test_temperature=ambient_temp,
                    charge_rate=1.0,
                    discharge_rate=1.0,
                    usage_profile="constant_current",
                    cycles=cycles
                )
                cell.compute_labels()
                cell.context = create_nasa_context(cell_id, ambient_temp)
                
                for cycle in cell.cycles:
                    cycle.context = cell.context
                
                self.cells[f"NASA_{cell_id}"] = cell
            
            print(f"Loaded NASA: {len(nasa_cells)} cells")
            
        except Exception as e:
            print(f"[WARN] Failed to load NASA data: {e}")
    
    def _create_synthetic(self, ds_name: str, output_dir: str):
        """Create synthetic data for testing."""
        # Synthetic data creation is optional - skip if not needed
        print(f"[INFO] Synthetic data creation skipped for {ds_name}")
    
    def _create_samples(self):
        """Convert cells to unified samples."""
        self.samples = []
        
        # Pre-compute lithium inventory features for all cells (for efficiency)
        lithium_features_cache = {}
        if self.use_lithium_features:
            print("Extracting lithium inventory features...")
            for cell_id, cell in self.cells.items():
                try:
                    lithium_features_cache[cell_id] = extract_lithium_features_for_cell(cell)
                except Exception as e:
                    print(f"Warning: Could not extract lithium features for {cell_id}: {e}")
                    lithium_features_cache[cell_id] = {}
        
        for cell in self.cells.values():
            # Get pre-computed lithium features for this cell
            cell_lithium_features = lithium_features_cache.get(cell.cell_id, {})
            
            for cycle in cell.cycles:
                # Create base feature vector
                base_features = cycle.to_feature_vector()
                
                # Augment with lithium inventory features if enabled
                if self.use_lithium_features:
                    try:
                        features = augment_cycle_with_lithium_features(
                            cycle, cell, cell_lithium_features
                        )
                    except Exception as e:
                        print(f"Warning: Could not augment features for {cell.cell_id} cycle {cycle.cycle_index}: {e}")
                        # Fallback: pad with zeros
                        li_features = np.zeros(11, dtype=np.float32)
                        features = np.concatenate([base_features, li_features])
                else:
                    features = base_features

                # Harmonize SOH scale: if SOH looks like a percent (>5), convert to ratio
                soh_val = cycle.soh_capacity
                if soh_val is None or np.isnan(soh_val):
                    soh_val = 0.9
                elif soh_val > 5:
                    soh_val = soh_val / 100.0
                # Clamp to a reasonable range
                soh_val = float(np.clip(soh_val, 0.0, 1.2))

                # RUL: derive from SOH trajectory if not available
                rul_val = cycle.rul_cycles
                if rul_val is None or np.isnan(rul_val) or rul_val == 0:
                    # Default to 100 for now; will be overwritten for TBSI below
                    rul_val = 100.0
                
                # Get context
                context = cycle.context or cell.context
                if context is None:
                    context = create_nasa_context(cell.cell_id, cell.test_temperature)

                # Build condensed numeric context + chemistry id
                # 5D context: [Temperature, ChargeRate, DischargeRate, SOC, UsageProfile]
                temp_norm = normalize_temperature(getattr(cell, 'test_temperature', 25.0))
                charge_norm = normalize_crate(getattr(cell, 'charge_rate', 1.0))
                discharge_norm = normalize_crate(getattr(cell, 'discharge_rate', getattr(cell, 'charge_rate', 1.0)))
                
                # Get SOC from context (if available)
                soc_pct = context.soc_pct if context and context.soc_pct is not None else 50.0  # Default 50%
                soc_norm = soc_pct / 100.0
                soc_norm = np.clip(soc_norm, 0.0, 1.0)
                
                # Get UsageProfile from context and normalize to [0, 1]
                # STORAGE = 0.0, cycling profiles = 0.5-1.0 based on intensity
                usage_profile = context.usage_profile if context and hasattr(context, 'usage_profile') else UsageProfileContext.CONSTANT_CURRENT
                if usage_profile == UsageProfileContext.STORAGE:
                    usage_profile_norm = 0.0  # Storage/calendar aging
                else:
                    # Normalize cycling profiles: 0.1 (eco) to 1.0 (aggressive)
                    profile_value = usage_profile.value
                    # Map: CONSTANT_CURRENT=0 -> 0.3, URBAN=1 -> 0.5, AGGRESSIVE=4 -> 1.0, etc.
                    if profile_value <= UsageProfileContext.CONSTANT_CURRENT.value:
                        usage_profile_norm = 0.3
                    elif profile_value <= UsageProfileContext.ECO_DRIVING.value:
                        usage_profile_norm = 0.4
                    elif profile_value <= UsageProfileContext.URBAN_DRIVING.value:
                        usage_profile_norm = 0.5
                    elif profile_value <= UsageProfileContext.MIXED_DRIVING.value:
                        usage_profile_norm = 0.7
                    else:
                        usage_profile_norm = min(1.0, 0.5 + (profile_value - UsageProfileContext.MIXED_DRIVING.value) * 0.1)
                
                chem_id = chemistry_to_id(context.chemistry if hasattr(context, 'chemistry') else ChemistryContext.LCO)
                context_vector = np.array([temp_norm, charge_norm, discharge_norm, soc_norm, usage_profile_norm], dtype=np.float32)
                
                # Estimate EOL for this cell (will be refined later)
                eol_cycle = len(cell.cycles)
                rul_normalized = float(rul_val) / max(eol_cycle, 1)
                
                sample = UnifiedSample(
                    cell_id=cell.cell_id,
                    cycle_idx=cycle.cycle_index,
                    source_dataset=cell.source_dataset,
                    features=features,
                    soh=soh_val,
                    rul=int(rul_val),
                    context=context,
                    context_vector=context_vector,
                    chem_id=chem_id,
                    rul_normalized=rul_normalized,
                    eol_cycle=eol_cycle
                )
                
                self.samples.append(sample)
        
        print(f"Created {len(self.samples)} unified samples from {len(self.cells)} cells")
        if self.use_lithium_features:
            print(f"  Features augmented with lithium inventory: {self.feature_dim}D (9 original + 11 lithium)")
        
        # Post-process: derive RUL for TBSI samples based on trajectory
        self._derive_tbsi_rul()
    
    def _derive_tbsi_rul(self):
        """
        Derive RUL and fix SOH for TBSI samples based on capacity trajectory.
        
        TBSI stores each cycle as a separate "cell", so we need to:
        1. Load Labels.xlsx to get actual capacity values
        2. Sort TBSI samples by their cell_id (which contains the row/cycle index)
        3. Compute SOH from capacity (normalized by first value)
        4. Find the EOL cycle (first where SOH < 0.8)
        5. Compute RUL = EOL_cycle - current_cycle for each sample
        """
        import pandas as pd
        
        # Get TBSI samples
        tbsi_samples = [(i, s) for i, s in enumerate(self.samples) if s.source_dataset == 'tbsi_sunwoda']
        
        if not tbsi_samples:
            return
        
        # Load Labels.xlsx directly to get capacity values
        labels_path = self.data_root / 'tbsi_sunwoda' / 'TBSI-Sunwoda-Battery-Dataset-main' / 'Labels.xlsx'
        if not labels_path.exists():
            print(f"[WARN] TBSI Labels.xlsx not found at {labels_path}")
            return
        
        labels_df = pd.read_excel(labels_path)
        capacities = labels_df.iloc[:, 0].values.astype(float)
        
        # Compute SOH (normalized capacity)
        cap0 = capacities[0]
        sohs = capacities / cap0
        
        # Find EOL (first cycle where SOH < 0.8)
        eol_idx = np.argmax(sohs < 0.8) if np.any(sohs < 0.8) else len(sohs)
        
        print(f"TBSI EOL detected at cycle {eol_idx} (SOH={sohs[eol_idx]:.3f})")
        
        # Sort samples by cell_id to match capacity order
        tbsi_samples_sorted = sorted(tbsi_samples, key=lambda x: int(x[1].cell_id.split('_')[-1]))
        
        # Update SOH and RUL for each sample
        for i, (orig_idx, sample) in enumerate(tbsi_samples_sorted):
            if i < len(sohs):
                derived_soh = float(sohs[i])
                derived_rul = max(0, eol_idx - i)
                derived_rul_norm = derived_rul / max(eol_idx, 1)  # Normalized to [0, 1]
            else:
                derived_soh = sample.soh
                derived_rul = 0
                derived_rul_norm = 0.0
            
            # Update the sample in-place
            self.samples[orig_idx] = UnifiedSample(
                cell_id=sample.cell_id,
                cycle_idx=sample.cycle_idx,
                source_dataset=sample.source_dataset,
                features=sample.features,
                soh=derived_soh,
                rul=int(derived_rul),
                context=sample.context,
                context_vector=sample.context_vector,
                chem_id=getattr(sample, 'chem_id', 0),
                rul_normalized=derived_rul_norm,
                eol_cycle=eol_idx
            )
        
        # Also update normalized RUL for non-TBSI samples
        self._normalize_all_rul()
        
        # Verify
        tbsi_updated = [s for s in self.samples if s.source_dataset == 'tbsi_sunwoda']
        tbsi_sohs = [s.soh for s in tbsi_updated]
        tbsi_ruls = [s.rul for s in tbsi_updated]
        tbsi_ruls_norm = [s.rul_normalized for s in tbsi_updated]
        print(f"TBSI SOH updated: min={min(tbsi_sohs):.3f}, max={max(tbsi_sohs):.3f}")
        print(f"TBSI RUL derived: min={min(tbsi_ruls):.0f}, max={max(tbsi_ruls):.0f}, mean={np.mean(tbsi_ruls):.1f}")
        print(f"TBSI RUL normalized: min={min(tbsi_ruls_norm):.3f}, max={max(tbsi_ruls_norm):.3f}")
    
    def _normalize_all_rul(self):
        """
        Compute normalized RUL for all samples based on their cell's EOL.
        
        For each cell, find the maximum RUL (which approximates EOL) and normalize.
        """
        # Group samples by cell_id
        from collections import defaultdict
        cell_samples = defaultdict(list)
        
        for i, sample in enumerate(self.samples):
            cell_samples[sample.cell_id].append((i, sample))
        
        # For each cell, find EOL and normalize RUL
        for cell_id, samples in cell_samples.items():
            # Skip TBSI (already normalized)
            if samples[0][1].source_dataset == 'tbsi_sunwoda':
                continue
            
            # Find max RUL as EOL estimate
            max_rul = max(s[1].rul for s in samples)
            eol_cycle = max_rul if max_rul > 0 else len(samples)
            
            # Update each sample
            for orig_idx, sample in samples:
                rul_norm = sample.rul / max(eol_cycle, 1)
                self.samples[orig_idx] = UnifiedSample(
                    cell_id=sample.cell_id,
                    cycle_idx=sample.cycle_idx,
                    source_dataset=sample.source_dataset,
                    features=sample.features,
                    soh=sample.soh,
                    rul=sample.rul,
                    context=sample.context,
                    context_vector=sample.context_vector,
                    chem_id=getattr(sample, 'chem_id', 0),
                    rul_normalized=rul_norm,
                    eol_cycle=eol_cycle
                )
        
        # Print stats
        all_rul_norm = [s.rul_normalized for s in self.samples]
        print(f"All RUL normalized: min={min(all_rul_norm):.3f}, max={max(all_rul_norm):.3f}, mean={np.mean(all_rul_norm):.3f}")
    
    def create_dataset(self) -> UnifiedBatteryDataset:
        """Create a PyTorch dataset from all loaded data."""
        if not self._loaded:
            raise RuntimeError("No data loaded. Call load_datasets() first.")
        
        return UnifiedBatteryDataset(self.samples, sequence_length=self.sequence_length)
    
    def create_splits(
        self,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        split_by_cell: bool = True,
        stratify_by: str = 'dataset',
        random_state: int = 42
    ) -> Tuple[UnifiedBatteryDataset, UnifiedBatteryDataset, UnifiedBatteryDataset]:
        """
        Create train/val/test splits.
        
        Args:
            train_ratio, val_ratio, test_ratio: Split ratios (must sum to 1)
            split_by_cell: If True, split by cell to prevent data leakage
            stratify_by: Stratification criterion ('dataset', 'temperature', 'chemistry')
            random_state: Random seed
        
        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        if not self._loaded:
            raise RuntimeError("No data loaded. Call load_datasets() first.")
        
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 0.001
        
        if split_by_cell:
            # Group samples by cell
            cell_samples = {}
            for i, sample in enumerate(self.samples):
                if sample.cell_id not in cell_samples:
                    cell_samples[sample.cell_id] = []
                cell_samples[sample.cell_id].append(i)
            
            cell_ids = list(cell_samples.keys())
            
            # Get stratification labels
            if stratify_by == 'dataset':
                labels = [self.cells[cid].source_dataset for cid in cell_ids]
            elif stratify_by == 'temperature':
                labels = [self.cells[cid].context.temperature.name for cid in cell_ids]
            elif stratify_by == 'chemistry':
                labels = [self.cells[cid].context.chemistry.name for cid in cell_ids]
            else:
                labels = None
            
            # Split cells
            train_cells, temp_cells = train_test_split(
                cell_ids,
                test_size=(val_ratio + test_ratio),
                stratify=labels,
                random_state=random_state
            )
            
            val_cells, test_cells = train_test_split(
                temp_cells,
                test_size=test_ratio / (val_ratio + test_ratio),
                random_state=random_state
            )
            
            # Get sample indices
            train_idx = [i for cid in train_cells for i in cell_samples[cid]]
            val_idx = [i for cid in val_cells for i in cell_samples[cid]]
            test_idx = [i for cid in test_cells for i in cell_samples[cid]]
        else:
            # Simple random split
            indices = list(range(len(self.samples)))
            train_idx, temp_idx = train_test_split(
                indices,
                test_size=(val_ratio + test_ratio),
                random_state=random_state
            )
            val_idx, test_idx = train_test_split(
                temp_idx,
                test_size=test_ratio / (val_ratio + test_ratio),
                random_state=random_state
            )
        
        # Create datasets
        train_samples = [self.samples[i] for i in train_idx]
        val_samples = [self.samples[i] for i in val_idx]
        test_samples = [self.samples[i] for i in test_idx]
        
        return (
            UnifiedBatteryDataset(train_samples, self.sequence_length),
            UnifiedBatteryDataset(val_samples, self.sequence_length),
            UnifiedBatteryDataset(test_samples, self.sequence_length)
        )
    
    def create_cross_dataset_splits(
        self,
        test_datasets: List[str]
    ) -> Tuple[UnifiedBatteryDataset, UnifiedBatteryDataset]:
        """
        Create splits for cross-dataset validation.
        
        Train on some datasets, test on others.
        
        Args:
            test_datasets: Datasets to use for testing only
        
        Returns:
            Tuple of (train_dataset, test_dataset)
        """
        if not self._loaded:
            raise RuntimeError("No data loaded. Call load_datasets() first.")
        
        test_datasets = [d.lower() for d in test_datasets]
        
        train_samples = []
        test_samples = []
        
        for sample in self.samples:
            if sample.source_dataset.lower() in test_datasets:
                test_samples.append(sample)
            else:
                train_samples.append(sample)
        
        print(f"Cross-dataset split: {len(train_samples)} train, {len(test_samples)} test")
        
        return (
            UnifiedBatteryDataset(train_samples, self.sequence_length),
            UnifiedBatteryDataset(test_samples, self.sequence_length)
        )
    
    def create_context_aware_splits(
        self,
        held_out_temperatures: Optional[List[str]] = None,
        held_out_chemistries: Optional[List[str]] = None,
        held_out_profiles: Optional[List[str]] = None
    ) -> Tuple[UnifiedBatteryDataset, UnifiedBatteryDataset]:
        """
        Create splits for context generalization testing.
        
        Hold out specific contexts for testing to evaluate generalization.
        
        Args:
            held_out_temperatures: Temperature contexts to hold out
            held_out_chemistries: Chemistry contexts to hold out
            held_out_profiles: Profile contexts to hold out
        
        Returns:
            Tuple of (train_dataset, test_dataset)
        """
        if not self._loaded:
            raise RuntimeError("No data loaded. Call load_datasets() first.")
        
        held_out_temps = [t.upper() for t in (held_out_temperatures or [])]
        held_out_chems = [c.upper() for c in (held_out_chemistries or [])]
        held_out_profs = [p.upper() for p in (held_out_profiles or [])]
        
        train_samples = []
        test_samples = []
        
        for sample in self.samples:
            is_held_out = (
                sample.context.temperature.name in held_out_temps or
                sample.context.chemistry.name in held_out_chems or
                sample.context.usage_profile.name in held_out_profs
            )
            
            if is_held_out:
                test_samples.append(sample)
            else:
                train_samples.append(sample)
        
        print(f"Context-aware split: {len(train_samples)} train, {len(test_samples)} test")
        if held_out_temps:
            print(f"  Held out temperatures: {held_out_temps}")
        if held_out_chems:
            print(f"  Held out chemistries: {held_out_chems}")
        if held_out_profs:
            print(f"  Held out profiles: {held_out_profs}")
        
        return (
            UnifiedBatteryDataset(train_samples, self.sequence_length),
            UnifiedBatteryDataset(test_samples, self.sequence_length)
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive pipeline statistics."""
        if not self._loaded:
            return {'status': 'not loaded'}
        
        ds = UnifiedBatteryDataset(self.samples)
        stats = ds.get_statistics()
        stats['num_cells'] = len(self.cells)
        stats['loaded_datasets'] = list(self.loaders.keys())
        
        return stats
    
    def save_processed(self, path: str):
        """Save processed data for fast loading."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save sample data
        samples_data = []
        for sample in self.samples:
            samples_data.append({
                'cell_id': sample.cell_id,
                'cycle_idx': sample.cycle_idx,
                'source_dataset': sample.source_dataset,
                'features': sample.features.tolist(),
                'soh': sample.soh,
                'rul': sample.rul,
                'context': sample.context.to_dict(),
                'context_vector': sample.context_vector.tolist(),
                'chem_id': sample.chem_id,
                'rul_normalized': sample.rul_normalized,
                'eol_cycle': sample.eol_cycle
            })
        
        with open(path / 'samples.json', 'w') as f:
            json.dump(samples_data, f)
        
        print(f"Saved {len(samples_data)} samples to {path}")
    
    def load_processed(self, path: str):
        """Load pre-processed data."""
        path = Path(path)
        
        with open(path / 'samples.json', 'r') as f:
            samples_data = json.load(f)
        
        self.samples = []
        for sd in samples_data:
            sample = UnifiedSample(
                cell_id=sd['cell_id'],
                cycle_idx=sd['cycle_idx'],
                source_dataset=sd['source_dataset'],
                features=np.array(sd['features'], dtype=np.float32),
                soh=sd['soh'],
                rul=sd['rul'],
                context=ExtendedBatteryContext.from_dict(sd['context']),
                context_vector=np.array(sd['context_vector'], dtype=np.float32),
                chem_id=sd.get('chem_id', 0),
                rul_normalized=sd.get('rul_normalized', 0.0),
                eol_cycle=sd.get('eol_cycle', 100)
            )
            self.samples.append(sample)
        
        self._loaded = True
        print(f"Loaded {len(self.samples)} samples from {path}")


# =============================================================================
# Convenience Functions
# =============================================================================

def create_multi_dataset_loaders(
    data_root: str,
    datasets: List[str],
    batch_size: int = 32,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create data loaders for multi-dataset training.
    
    Convenience function for quick setup.
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    pipeline = UnifiedDataPipeline(data_root)
    pipeline.load_datasets(datasets)
    
    train_ds, val_ds, test_ds = pipeline.create_splits(
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio
    )
    
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    
    return train_loader, val_loader, test_loader


# =============================================================================
# Testing
# =============================================================================

if __name__ == '__main__':
    print("="*60)
    print("Unified Data Pipeline - Test Suite")
    print("="*60)
    
    # Create test directory with synthetic data
    test_dir = Path("/tmp/unified_test")
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize pipeline
    pipeline = UnifiedDataPipeline(str(test_dir))
    
    # Load datasets (will create synthetic if missing)
    print("\n1. Loading datasets:")
    pipeline.load_datasets(['sandia', 'calce', 'oxford', 'tbsi_sunwoda'])
    
    # Print statistics
    print("\n2. Pipeline statistics:")
    stats = pipeline.get_statistics()
    print(f"   Total samples: {stats['total_samples']}")
    print(f"   By dataset: {stats['by_dataset']}")
    print(f"   By temperature: {stats['by_temperature']}")
    print(f"   By chemistry: {stats['by_chemistry']}")
    print(f"   By profile: {stats['by_profile']}")
    
    # Create standard splits
    print("\n3. Creating standard train/val/test splits:")
    train_ds, val_ds, test_ds = pipeline.create_splits()
    print(f"   Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
    
    # Create cross-dataset splits
    print("\n4. Creating cross-dataset splits (test on Oxford):")
    train_ds, test_ds = pipeline.create_cross_dataset_splits(['oxford'])
    print(f"   Train: {len(train_ds)}, Test: {len(test_ds)}")
    print(f"   Train datasets: {train_ds.get_statistics()['by_dataset']}")
    print(f"   Test datasets: {test_ds.get_statistics()['by_dataset']}")
    
    # Create context-aware splits
    print("\n5. Creating context-aware splits (test on HOT temperature):")
    train_ds, test_ds = pipeline.create_context_aware_splits(
        held_out_temperatures=['HOT']
    )
    print(f"   Train: {len(train_ds)}, Test: {len(test_ds)}")
    
    # Test data loading
    print("\n6. Testing DataLoader:")
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    batch = next(iter(train_loader))
    print(f"   Batch keys: {list(batch.keys())}")
    print(f"   Features shape: {batch['features'].shape}")
    print(f"   Context shape: {batch['context'].shape}")
    print(f"   SOH range: [{batch['soh'].min():.3f}, {batch['soh'].max():.3f}]")
    
    print("\n" + "="*60)
    print("All tests passed!")
    print("="*60)

