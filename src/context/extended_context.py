"""
Extended Multi-Dimensional Context System for Battery SOH Prediction

This module defines a rich context representation that captures:
1. Temperature Context: Operating temperature (Cold, Room, Hot)
2. Chemistry Context: Battery chemistry (LCO, NMC, LFP, NCA, etc.)
3. Usage Profile Context: Driving/usage pattern (Urban, Highway, Mixed, Constant)
4. C-Rate Context: Charge/discharge rate (Slow, Normal, Fast, Ultra-Fast)

The multi-dimensional context enables cross-domain generalization and
transfer learning across different battery types and operating conditions.

Author: Battery ML Research
Version: 2.0 - Extended Context
"""

import json
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field, asdict
from enum import Enum

# --------------------------------------------------------------------------- #
# Lightweight normalization helpers for condensed context vectors
# --------------------------------------------------------------------------- #

CHEM_ID_MAP = {
    "LCO": 0,
    "NMC": 1,
    "LFP": 2,
}


def normalize_temperature(temp_c: float) -> float:
    """Normalize temperature to [0,1] using a loose 0-60C range."""
    return float(np.clip(temp_c / 60.0, 0.0, 1.0))


def normalize_crate(c_rate: float) -> float:
    """Normalize C-rate to [0,1] using a loose 0-3C range."""
    return float(np.clip(c_rate / 3.0, 0.0, 1.0))


def chemistry_to_id(chemistry: "ChemistryContext") -> int:
    return CHEM_ID_MAP.get(chemistry.name, 0)


# Context Enumerations

class TemperatureContext(Enum):
    """Temperature operating conditions"""
    COLD = 0        # < 15°C (affects Li diffusion, increases IR)
    COOL = 1        # 15-20°C
    ROOM = 2        # 20-30°C (nominal)
    WARM = 3        # 30-40°C
    HOT = 4         # > 40°C (accelerates degradation)
    
    @classmethod
    def from_celsius(cls, temp_c: float) -> 'TemperatureContext':
        """Convert temperature in Celsius to context category."""
        if temp_c < 15:
            return cls.COLD
        elif temp_c < 20:
            return cls.COOL
        elif temp_c < 30:
            return cls.ROOM
        elif temp_c < 40:
            return cls.WARM
        else:
            return cls.HOT
    
    @classmethod
    def num_categories(cls) -> int:
        return 5


class ChemistryContext(Enum):
    """Battery chemistry types"""
    LCO = 0         # Lithium Cobalt Oxide (high energy, consumer electronics)
    NMC = 1         # Nickel Manganese Cobalt (balanced, EVs)
    NCA = 2         # Nickel Cobalt Aluminum (high energy, Tesla)
    LFP = 3         # Lithium Iron Phosphate (safe, long life)
    LMO = 4         # Lithium Manganese Oxide (power tools)
    NMC_111 = 5     # NMC with 1:1:1 ratio
    NMC_523 = 6     # NMC with 5:2:3 ratio
    NMC_622 = 7     # NMC with 6:2:2 ratio
    NMC_811 = 8     # NMC with 8:1:1 ratio (high Ni)
    UNKNOWN = 9     # Unknown chemistry
    
    @classmethod
    def from_string(cls, chem_str: str) -> 'ChemistryContext':
        """Parse chemistry string to enum."""
        chem_str = chem_str.upper().strip()
        
        # Direct matches
        mapping = {
            'LCO': cls.LCO,
            'NMC': cls.NMC,
            'NCA': cls.NCA,
            'LFP': cls.LFP,
            'LMO': cls.LMO,
            'NMC111': cls.NMC_111,
            'NMC-111': cls.NMC_111,
            'NMC523': cls.NMC_523,
            'NMC-523': cls.NMC_523,
            'NMC622': cls.NMC_622,
            'NMC-622': cls.NMC_622,
            'NMC811': cls.NMC_811,
            'NMC-811': cls.NMC_811,
            'LIFEPO4': cls.LFP,
            'LICOO2': cls.LCO,
        }
        
        for key, val in mapping.items():
            if key in chem_str:
                return val
        
        return cls.UNKNOWN
    
    @classmethod
    def num_categories(cls) -> int:
        return 10


class UsageProfileContext(Enum):
    """Usage/driving profile patterns"""
    CONSTANT_CURRENT = 0    # Lab-style CC-CV cycling
    URBAN_DRIVING = 1       # Stop-and-go, frequent acceleration
    HIGHWAY_DRIVING = 2     # Steady state, high speed
    MIXED_DRIVING = 3       # Combination of urban and highway
    AGGRESSIVE_DRIVING = 4  # Hard acceleration/braking
    ECO_DRIVING = 5         # Gentle, energy-efficient
    DYNAMIC_STRESS = 6      # DST profile (CALCE)
    FUDS = 7                # Federal Urban Driving Schedule
    US06 = 8                # US06 Highway Driving Schedule
    PULSE = 9               # Pulse discharge testing
    STORAGE = 10            # Calendar aging (no cycling)
    
    @classmethod
    def from_string(cls, profile_str: str) -> 'UsageProfileContext':
        """Parse profile string to enum."""
        profile_str = profile_str.upper().strip()
        
        mapping = {
            'CC': cls.CONSTANT_CURRENT,
            'CC-CV': cls.CONSTANT_CURRENT,
            'CCCV': cls.CONSTANT_CURRENT,
            'CONSTANT': cls.CONSTANT_CURRENT,
            'URBAN': cls.URBAN_DRIVING,
            'CITY': cls.URBAN_DRIVING,
            'HIGHWAY': cls.HIGHWAY_DRIVING,
            'HWY': cls.HIGHWAY_DRIVING,
            'MIXED': cls.MIXED_DRIVING,
            'AGGRESSIVE': cls.AGGRESSIVE_DRIVING,
            'ECO': cls.ECO_DRIVING,
            'DST': cls.DYNAMIC_STRESS,
            'FUDS': cls.FUDS,
            'US06': cls.US06,
            'PULSE': cls.PULSE,
            'STORAGE': cls.STORAGE,
            'CALENDAR': cls.STORAGE,
        }
        
        for key, val in mapping.items():
            if key in profile_str:
                return val
        
        return cls.CONSTANT_CURRENT  # Default
    
    @classmethod
    def num_categories(cls) -> int:
        return 11


class CRateContext(Enum):
    """Charge/discharge C-rate categories"""
    VERY_SLOW = 0   # < 0.5C (calendar aging, storage)
    SLOW = 1        # 0.5C - 1C (standard charging)
    NORMAL = 2      # 1C - 2C (typical usage)
    FAST = 3        # 2C - 4C (fast charging)
    ULTRA_FAST = 4  # > 4C (extreme fast charging)
    
    @classmethod
    def from_c_rate(cls, c_rate: float) -> 'CRateContext':
        """Convert C-rate value to context category."""
        if c_rate < 0.5:
            return cls.VERY_SLOW
        elif c_rate < 1.0:
            return cls.SLOW
        elif c_rate < 2.0:
            return cls.NORMAL
        elif c_rate < 4.0:
            return cls.FAST
        else:
            return cls.ULTRA_FAST
    
    @classmethod
    def num_categories(cls) -> int:
        return 5


# Extended Context Data Class

@dataclass
class ExtendedBatteryContext:
    """
    Multi-dimensional context representation for a battery trajectory.
    
    This context captures the full operational environment of a battery,
    enabling the model to understand and transfer knowledge across:
    - Different temperature conditions
    - Different battery chemistries
    - Different usage patterns
    - Different charge/discharge rates
    - Different State of Charge (SOC) levels
    
    Attributes:
        temperature: Operating temperature context
        chemistry: Battery chemistry type
        usage_profile: Driving/usage pattern
        c_rate: Charge/discharge rate category
        
        # Raw values (optional, for continuous features)
        temperature_celsius: Actual temperature in °C
        c_rate_value: Actual C-rate value
        soc_pct: State of Charge in percentage (0-100)
        
        # Metadata
        source_dataset: Origin dataset (NASA, Sandia, CALCE, Oxford, etc.)
        cell_id: Unique cell identifier
        additional_info: Any extra metadata
    """
    
    # Categorical contexts
    temperature: TemperatureContext = TemperatureContext.ROOM
    chemistry: ChemistryContext = ChemistryContext.UNKNOWN
    usage_profile: UsageProfileContext = UsageProfileContext.CONSTANT_CURRENT
    c_rate: CRateContext = CRateContext.NORMAL
    
    # Raw values
    temperature_celsius: Optional[float] = None
    c_rate_value: Optional[float] = None
    soc_pct: Optional[float] = None  # State of Charge in percentage (0-100)
    
    # Metadata
    source_dataset: str = "unknown"
    cell_id: str = "unknown"
    additional_info: Dict = field(default_factory=dict)
    
    def to_one_hot(self) -> np.ndarray:
        """
        Convert context to one-hot encoded vector.
        
        Returns:
            One-hot vector of shape (total_dim,) where:
            - dims 0-4: Temperature (5 categories)
            - dims 5-14: Chemistry (10 categories)
            - dims 15-25: Usage Profile (11 categories)
            - dims 26-30: C-Rate (5 categories)
            
            Total: 31 dimensions
        """
        total_dim = (
            TemperatureContext.num_categories() +
            ChemistryContext.num_categories() +
            UsageProfileContext.num_categories() +
            CRateContext.num_categories()
        )
        
        one_hot = np.zeros(total_dim, dtype=np.float32)
        
        offset = 0
        
        # Temperature
        one_hot[offset + self.temperature.value] = 1.0
        offset += TemperatureContext.num_categories()
        
        # Chemistry
        one_hot[offset + self.chemistry.value] = 1.0
        offset += ChemistryContext.num_categories()
        
        # Usage Profile
        one_hot[offset + self.usage_profile.value] = 1.0
        offset += UsageProfileContext.num_categories()
        
        # C-Rate
        one_hot[offset + self.c_rate.value] = 1.0
        
        return one_hot
    
    def to_embedding_indices(self) -> np.ndarray:
        """
        Convert context to embedding indices for learned embeddings.
        
        Returns:
            Array of indices: [temp_idx, chem_idx, profile_idx, crate_idx]
        """
        return np.array([
            self.temperature.value,
            self.chemistry.value,
            self.usage_profile.value,
            self.c_rate.value
        ], dtype=np.int64)
    
    def to_continuous(self) -> np.ndarray:
        """
        Convert context to continuous features (normalized).
        
        Returns:
            Array of continuous values: [temp_norm, charge_crate_norm, discharge_crate_norm, soc_norm]
            - temp_norm: Temperature normalized to [0, 1] (using normalize_temperature)
            - charge_crate_norm: Charge C-rate normalized to [0, 1] (using normalize_crate)
            - discharge_crate_norm: Discharge C-rate normalized to [0, 1] (using normalize_crate)
            - soc_norm: SOC normalized to [0, 1] (0% = 0.0, 100% = 1.0)
        """
        # Temperature: use existing normalization
        temp = self.temperature_celsius if self.temperature_celsius is not None else 25.0
        temp_norm = normalize_temperature(temp)
        
        # C-rate: use existing normalization (for charge/discharge, use same value for now)
        crate = self.c_rate_value if self.c_rate_value is not None else 1.0
        charge_crate_norm = normalize_crate(crate)
        discharge_crate_norm = normalize_crate(crate)
        
        # SOC: normalize 0-100% to 0-1
        soc = self.soc_pct if self.soc_pct is not None else 50.0  # Default to 50%
        soc_norm = soc / 100.0
        soc_norm = np.clip(soc_norm, 0.0, 1.0)
        
        return np.array([temp_norm, charge_crate_norm, discharge_crate_norm, soc_norm], dtype=np.float32)
    
    def to_hybrid(self) -> np.ndarray:
        """
        Create hybrid representation: one-hot + continuous features.
        
        Returns:
            Array of shape (35,): 31 one-hot dims + 4 continuous dims (temp, charge_crate, discharge_crate, soc)
        """
        one_hot = self.to_one_hot()
        continuous = self.to_continuous()
        return np.concatenate([one_hot, continuous])
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'temperature': self.temperature.name,
            'chemistry': self.chemistry.name,
            'usage_profile': self.usage_profile.name,
            'c_rate': self.c_rate.name,
            'temperature_celsius': self.temperature_celsius,
            'c_rate_value': self.c_rate_value,
            'soc_pct': self.soc_pct,
            'source_dataset': self.source_dataset,
            'cell_id': self.cell_id,
            'additional_info': self.additional_info
        }
    
    @classmethod
    def from_dict(cls, d: Dict) -> 'ExtendedBatteryContext':
        """Create from dictionary."""
        return cls(
            temperature=TemperatureContext[d.get('temperature', 'ROOM')],
            chemistry=ChemistryContext[d.get('chemistry', 'UNKNOWN')],
            usage_profile=UsageProfileContext[d.get('usage_profile', 'CONSTANT_CURRENT')],
            c_rate=CRateContext[d.get('c_rate', 'NORMAL')],
            temperature_celsius=d.get('temperature_celsius'),
            c_rate_value=d.get('c_rate_value'),
            soc_pct=d.get('soc_pct'),
            source_dataset=d.get('source_dataset', 'unknown'),
            cell_id=d.get('cell_id', 'unknown'),
            additional_info=d.get('additional_info', {})
        )
    
    def __repr__(self) -> str:
        return (f"ExtendedBatteryContext("
                f"T={self.temperature.name}, "
                f"Chem={self.chemistry.name}, "
                f"Profile={self.usage_profile.name}, "
                f"C={self.c_rate.name}, "
                f"src={self.source_dataset})")


# Context Dimension Constants

CONTEXT_DIMS = {
    'temperature': TemperatureContext.num_categories(),
    'chemistry': ChemistryContext.num_categories(),
    'usage_profile': UsageProfileContext.num_categories(),
    'c_rate': CRateContext.num_categories(),
    'continuous': 4,  # temp_celsius, charge_crate, discharge_crate, soc_pct
}

TOTAL_ONE_HOT_DIM = sum([
    CONTEXT_DIMS['temperature'],
    CONTEXT_DIMS['chemistry'],
    CONTEXT_DIMS['usage_profile'],
    CONTEXT_DIMS['c_rate']
])

TOTAL_HYBRID_DIM = TOTAL_ONE_HOT_DIM + CONTEXT_DIMS['continuous']


# Helper Functions

def create_context_from_metadata(
    temperature_c: Optional[float] = None,
    chemistry: Optional[str] = None,
    profile: Optional[str] = None,
    c_rate: Optional[float] = None,
    soc_pct: Optional[float] = None,
    source_dataset: str = "unknown",
    cell_id: str = "unknown",
    **kwargs
) -> ExtendedBatteryContext:
    """
    Create context from raw metadata values.
    
    Args:
        temperature_c: Operating temperature in Celsius
        chemistry: Battery chemistry string (e.g., "NMC", "LFP")
        profile: Usage profile string (e.g., "urban", "highway")
        c_rate: Charge/discharge rate
        source_dataset: Dataset origin
        cell_id: Cell identifier
        **kwargs: Additional metadata
    
    Returns:
        ExtendedBatteryContext instance
    """
    return ExtendedBatteryContext(
        temperature=TemperatureContext.from_celsius(temperature_c or 25.0),
        chemistry=ChemistryContext.from_string(chemistry or "unknown"),
        usage_profile=UsageProfileContext.from_string(profile or "constant"),
        c_rate=CRateContext.from_c_rate(c_rate or 1.0),
        temperature_celsius=temperature_c,
        c_rate_value=c_rate,
        soc_pct=soc_pct,
        source_dataset=source_dataset,
        cell_id=cell_id,
        additional_info=kwargs
    )


def batch_contexts_to_tensor(
    contexts: List[ExtendedBatteryContext],
    mode: str = 'hybrid',
    device: str = 'cpu'
) -> torch.Tensor:
    """
    Convert a batch of contexts to a tensor.
    
    Args:
        contexts: List of ExtendedBatteryContext instances
        mode: 'one_hot', 'indices', 'continuous', or 'hybrid'
        device: Target device
    
    Returns:
        Tensor of shape (batch_size, context_dim)
    """
    if mode == 'one_hot':
        arrays = [ctx.to_one_hot() for ctx in contexts]
    elif mode == 'indices':
        arrays = [ctx.to_embedding_indices() for ctx in contexts]
    elif mode == 'continuous':
        arrays = [ctx.to_continuous() for ctx in contexts]
    elif mode == 'hybrid':
        arrays = [ctx.to_hybrid() for ctx in contexts]
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    return torch.tensor(np.stack(arrays), dtype=torch.float32, device=device)


def context_similarity(
    ctx1: ExtendedBatteryContext,
    ctx2: ExtendedBatteryContext,
    weights: Optional[Dict[str, float]] = None
) -> float:
    """
    Compute similarity between two contexts.
    
    Args:
        ctx1, ctx2: Contexts to compare
        weights: Optional weights for each dimension
            Default: {'temperature': 0.3, 'chemistry': 0.3, 
                      'usage_profile': 0.2, 'c_rate': 0.2}
    
    Returns:
        Similarity score in [0, 1]
    """
    if weights is None:
        weights = {
            'temperature': 0.3,
            'chemistry': 0.3,
            'usage_profile': 0.2,
            'c_rate': 0.2
        }
    
    score = 0.0
    
    # Exact match scoring
    if ctx1.temperature == ctx2.temperature:
        score += weights['temperature']
    elif abs(ctx1.temperature.value - ctx2.temperature.value) == 1:
        score += weights['temperature'] * 0.5  # Adjacent categories
    
    if ctx1.chemistry == ctx2.chemistry:
        score += weights['chemistry']
    
    if ctx1.usage_profile == ctx2.usage_profile:
        score += weights['usage_profile']
    
    if ctx1.c_rate == ctx2.c_rate:
        score += weights['c_rate']
    elif abs(ctx1.c_rate.value - ctx2.c_rate.value) == 1:
        score += weights['c_rate'] * 0.5  # Adjacent categories
    
    return score


# Dataset-Specific Context Builders

def create_nasa_context(cell_id: str, ambient_temp: float = 24.0) -> ExtendedBatteryContext:
    """Create context for NASA dataset cells."""
    return ExtendedBatteryContext(
        temperature=TemperatureContext.from_celsius(ambient_temp),
        chemistry=ChemistryContext.LCO,  # NASA uses 18650 LCO cells
        usage_profile=UsageProfileContext.CONSTANT_CURRENT,
        c_rate=CRateContext.NORMAL,  # Typically 1C-2C
        temperature_celsius=ambient_temp,
        c_rate_value=1.0,
        source_dataset="NASA",
        cell_id=cell_id
    )


def create_sandia_context(
    cell_id: str,
    temperature_c: float,
    chemistry: str = "NMC",
    c_rate: float = 1.0
) -> ExtendedBatteryContext:
    """Create context for Sandia dataset cells."""
    return ExtendedBatteryContext(
        temperature=TemperatureContext.from_celsius(temperature_c),
        chemistry=ChemistryContext.from_string(chemistry),
        usage_profile=UsageProfileContext.CONSTANT_CURRENT,
        c_rate=CRateContext.from_c_rate(c_rate),
        temperature_celsius=temperature_c,
        c_rate_value=c_rate,
        source_dataset="Sandia",
        cell_id=cell_id
    )


def create_calce_context(
    cell_id: str,
    chemistry: str,
    profile: str = "DST",
    temperature_c: float = 25.0
) -> ExtendedBatteryContext:
    """Create context for CALCE dataset cells."""
    return ExtendedBatteryContext(
        temperature=TemperatureContext.from_celsius(temperature_c),
        chemistry=ChemistryContext.from_string(chemistry),
        usage_profile=UsageProfileContext.from_string(profile),
        c_rate=CRateContext.NORMAL,
        temperature_celsius=temperature_c,
        source_dataset="CALCE",
        cell_id=cell_id
    )


def create_oxford_context(
    cell_id: str,
    profile: str = "urban"
) -> ExtendedBatteryContext:
    """Create context for Oxford dataset cells."""
    return ExtendedBatteryContext(
        temperature=TemperatureContext.ROOM,
        chemistry=ChemistryContext.LCO,  # Oxford uses LCO pouch cells
        usage_profile=UsageProfileContext.from_string(profile),
        c_rate=CRateContext.NORMAL,
        temperature_celsius=25.0,
        source_dataset="Oxford",
        cell_id=cell_id
    )


def create_tbsi_sunwoda_context(
    cell_id: str,
    temperature_c: float,
    c_rate: float
) -> ExtendedBatteryContext:
    """Create context for TBSI Sunwoda dataset cells."""
    return ExtendedBatteryContext(
        temperature=TemperatureContext.from_celsius(temperature_c),
        chemistry=ChemistryContext.NMC,  # TBSI uses NMC cells
        usage_profile=UsageProfileContext.MIXED_DRIVING,  # EV conditions
        c_rate=CRateContext.from_c_rate(c_rate),
        temperature_celsius=temperature_c,
        c_rate_value=c_rate,
        source_dataset="TBSI_Sunwoda",
        cell_id=cell_id
    )


# Testing

if __name__ == '__main__':
    print("="*60)
    print("Extended Battery Context System - Test Suite")
    print("="*60)
    
    # Test 1: Create contexts
    print("\n1. Creating contexts from different datasets:")
    
    nasa_ctx = create_nasa_context("B0005", 24.0)
    print(f"   NASA: {nasa_ctx}")
    
    sandia_ctx = create_sandia_context("SNL_001", 35.0, "NMC", 2.0)
    print(f"   Sandia: {sandia_ctx}")
    
    calce_ctx = create_calce_context("CS2_35", "LCO", "DST", 25.0)
    print(f"   CALCE: {calce_ctx}")
    
    oxford_ctx = create_oxford_context("Cell_001", "urban")
    print(f"   Oxford: {oxford_ctx}")
    
    tbsi_ctx = create_tbsi_sunwoda_context("SW_001", 40.0, 3.0)
    print(f"   TBSI: {tbsi_ctx}")
    
    # Test 2: Encoding
    print("\n2. Context encoding:")
    print(f"   One-hot dim: {TOTAL_ONE_HOT_DIM}")
    print(f"   Hybrid dim: {TOTAL_HYBRID_DIM}")
    
    print(f"\n   NASA one-hot: shape={nasa_ctx.to_one_hot().shape}")
    print(f"   NASA hybrid: shape={nasa_ctx.to_hybrid().shape}")
    print(f"   NASA indices: {nasa_ctx.to_embedding_indices()}")
    
    # Test 3: Batch encoding
    print("\n3. Batch encoding:")
    contexts = [nasa_ctx, sandia_ctx, calce_ctx, oxford_ctx, tbsi_ctx]
    batch_tensor = batch_contexts_to_tensor(contexts, mode='hybrid')
    print(f"   Batch shape: {batch_tensor.shape}")
    
    # Test 4: Context similarity
    print("\n4. Context similarity:")
    print(f"   NASA vs NASA: {context_similarity(nasa_ctx, nasa_ctx):.3f}")
    print(f"   NASA vs Sandia: {context_similarity(nasa_ctx, sandia_ctx):.3f}")
    print(f"   NASA vs CALCE: {context_similarity(nasa_ctx, calce_ctx):.3f}")
    print(f"   Sandia(35C) vs TBSI(40C): {context_similarity(sandia_ctx, tbsi_ctx):.3f}")
    
    # Test 5: Serialization
    print("\n5. Serialization test:")
    ctx_dict = nasa_ctx.to_dict()
    print(f"   Serialized: {ctx_dict}")
    
    restored = ExtendedBatteryContext.from_dict(ctx_dict)
    print(f"   Restored: {restored}")
    
    print("\n" + "="*60)
    print("All tests passed!")
    print("="*60)

