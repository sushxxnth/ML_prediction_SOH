"""
Lithium Inventory Feature Extraction for Zero-Shot RUL Generalization

This module extracts physics-based differentiating variables that capture
fundamental differences between battery chemistries, enabling better zero-shot
generalization for RUL prediction.

Key Concept: Different battery chemistries have different:
1. Lithium-ion concentration (active Li+ per unit volume)
2. Voltage profile characteristics (reflecting Li+ intercalation behavior)
3. Capacity fade mechanisms (SEI growth, lithium plating, particle cracking)
4. Internal resistance evolution (ionic conductivity differences)

Author: Battery ML Research
Version: 1.0
"""

import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class LithiumInventoryFeatures:
    """
    Physics-based features that differentiate battery chemistries.
    
    These features capture fundamental electrochemical properties that
    vary between LCO, NMC, LFP, etc., enabling zero-shot generalization.
    """
    # Voltage profile characteristics (reflect Li+ intercalation behavior)
    voltage_plateau_center: float  # Center of voltage plateau (V)
    voltage_plateau_width: float   # Width of voltage plateau (V)
    voltage_slope: float           # dV/dQ slope (V/Ah)
    
    # Capacity characteristics (reflect active material content)
    theoretical_capacity_ratio: float  # Measured / Theoretical capacity
    capacity_fade_rate: float          # dQ/dN (Ah/cycle)
    
    # Resistance characteristics (reflect ionic conductivity)
    resistance_growth_rate: float      # dR/dN (Ohm/cycle)
    resistance_initial: float          # Initial resistance (Ohm)
    
    # Lithium inventory indicators
    lithium_utilization: float         # Effective Li+ utilization (0-1)
    lithium_plating_risk: float        # Risk of Li plating (0-1)
    
    # Chemistry-specific signatures
    voltage_profile_entropy: float    # Entropy of voltage distribution
    discharge_curve_curvature: float  # Curvature of discharge curve


def extract_voltage_profile_features(
    voltage: np.ndarray,
    capacity: np.ndarray,
    current: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Extract voltage profile characteristics that differentiate chemistries.
    
    Different chemistries have distinct voltage profiles:
    - LCO: Steep voltage drop, narrow plateau
    - NMC: Moderate slope, wider plateau
    - LFP: Very flat plateau, low voltage
    - NCA: Similar to NMC but higher voltage
    
    Args:
        voltage: Voltage measurements (V)
        capacity: Capacity measurements (Ah) - cumulative or incremental
        current: Current measurements (A) - optional, for dV/dQ calculation
    
    Returns:
        Dictionary of voltage profile features
    """
    if len(voltage) < 2 or len(capacity) < 2:
        return {
            'voltage_plateau_center': 3.7,
            'voltage_plateau_width': 0.5,
            'voltage_slope': -0.1,
            'voltage_profile_entropy': 0.0,
            'discharge_curve_curvature': 0.0
        }
    
    # Ensure capacity is incremental (dQ) if cumulative
    if capacity[-1] > capacity[0] * 10:  # Likely cumulative
        dQ = np.diff(capacity)
        dQ = np.concatenate([[dQ[0]], dQ])  # Keep same length
    else:
        dQ = capacity
    
    # Voltage plateau center: median voltage during discharge
    voltage_plateau_center = np.median(voltage)
    
    # Voltage plateau width: range of voltage during middle 80% of discharge
    mid_start = int(0.1 * len(voltage))
    mid_end = int(0.9 * len(voltage))
    if mid_end > mid_start:
        voltage_plateau_width = np.max(voltage[mid_start:mid_end]) - np.min(voltage[mid_start:mid_end])
    else:
        voltage_plateau_width = np.max(voltage) - np.min(voltage)
    
    # Voltage slope: dV/dQ (change in voltage per unit capacity)
    if len(voltage) > 1 and np.sum(np.abs(dQ)) > 0:
        # Use linear regression for dV/dQ
        valid_mask = np.abs(dQ) > 1e-6
        if valid_mask.sum() > 1:
            dV_dQ = np.polyfit(dQ[valid_mask], voltage[valid_mask], 1)[0]
        else:
            dV_dQ = (voltage[-1] - voltage[0]) / (np.sum(dQ) + 1e-6)
    else:
        dV_dQ = (voltage[-1] - voltage[0]) / (np.sum(dQ) + 1e-6) if np.sum(dQ) > 0 else -0.1
    
    # Voltage profile entropy: measure of voltage distribution complexity
    # Higher entropy = more complex voltage profile (typical of NMC)
    # Lower entropy = simpler profile (typical of LFP)
    voltage_hist, _ = np.histogram(voltage, bins=20)
    voltage_hist = voltage_hist + 1e-10  # Avoid log(0)
    voltage_hist = voltage_hist / np.sum(voltage_hist)
    voltage_profile_entropy = -np.sum(voltage_hist * np.log(voltage_hist))
    
    # Discharge curve curvature: second derivative of voltage w.r.t. capacity
    if len(voltage) > 2:
        # Approximate second derivative
        dV = np.diff(voltage)
        dQ_valid = dQ[1:] if len(dQ) > 1 else dQ
        if len(dQ_valid) > 0 and np.sum(np.abs(dQ_valid)) > 0:
            d2V_dQ2 = np.mean(np.abs(np.diff(dV / (dQ_valid + 1e-6))))
        else:
            d2V_dQ2 = 0.0
    else:
        d2V_dQ2 = 0.0
    
    return {
        'voltage_plateau_center': float(voltage_plateau_center),
        'voltage_plateau_width': float(voltage_plateau_width),
        'voltage_slope': float(dV_dQ),
        'voltage_profile_entropy': float(voltage_profile_entropy),
        'discharge_curve_curvature': float(d2V_dQ2)
    }


def extract_capacity_features(
    capacity_history: np.ndarray,
    cycle_indices: np.ndarray,
    nominal_capacity: float,
    theoretical_capacity: Optional[float] = None
) -> Dict[str, float]:
    """
    Extract capacity-related features that reflect active material content.
    
    Args:
        capacity_history: Capacity measurements across cycles (Ah)
        cycle_indices: Cycle numbers
        nominal_capacity: Nominal capacity (Ah)
        theoretical_capacity: Theoretical capacity if known (Ah)
    
    Returns:
        Dictionary of capacity features
    """
    if len(capacity_history) < 2:
        return {
            'theoretical_capacity_ratio': 1.0,
            'capacity_fade_rate': 0.0,
            'lithium_utilization': 0.9
        }
    
    # Theoretical capacity ratio
    if theoretical_capacity is not None and theoretical_capacity > 0:
        theoretical_capacity_ratio = np.mean(capacity_history) / theoretical_capacity
    else:
        # Estimate from initial capacity (assuming fresh cell is close to theoretical)
        initial_capacity = capacity_history[0] if len(capacity_history) > 0 else nominal_capacity
        theoretical_capacity_ratio = initial_capacity / nominal_capacity
    
    # Capacity fade rate: linear fit of capacity vs cycle
    if len(cycle_indices) > 1:
        # Fit linear model: Q = Q0 - fade_rate * N
        valid_mask = ~np.isnan(capacity_history) & (capacity_history > 0)
        if valid_mask.sum() > 1:
            coeffs = np.polyfit(cycle_indices[valid_mask], capacity_history[valid_mask], 1)
            capacity_fade_rate = -coeffs[0]  # Negative slope = fade
        else:
            capacity_fade_rate = 0.0
    else:
        capacity_fade_rate = 0.0
    
    # Lithium utilization: how effectively Li+ is being used
    # Higher utilization = better performance
    current_capacity = capacity_history[-1] if len(capacity_history) > 0 else nominal_capacity
    initial_capacity = capacity_history[0] if len(capacity_history) > 0 else nominal_capacity
    if initial_capacity > 0:
        lithium_utilization = current_capacity / initial_capacity
    else:
        lithium_utilization = 0.9
    
    return {
        'theoretical_capacity_ratio': float(np.clip(theoretical_capacity_ratio, 0.5, 1.5)),
        'capacity_fade_rate': float(np.clip(capacity_fade_rate, -0.01, 0.01)),
        'lithium_utilization': float(np.clip(lithium_utilization, 0.0, 1.2))
    }


def extract_resistance_features(
    resistance_history: np.ndarray,
    cycle_indices: np.ndarray,
    temperature: Optional[float] = None
) -> Dict[str, float]:
    """
    Extract resistance-related features that reflect ionic conductivity.
    
    Different chemistries show different resistance growth patterns:
    - LCO: Moderate resistance growth
    - NMC: Lower initial resistance, gradual growth
    - LFP: Very low resistance, minimal growth
    
    Args:
        resistance_history: Internal resistance measurements (Ohm)
        cycle_indices: Cycle numbers
        temperature: Operating temperature (°C) - optional
    
    Returns:
        Dictionary of resistance features
    """
    if len(resistance_history) < 1:
        return {
            'resistance_initial': 0.05,
            'resistance_growth_rate': 0.0,
            'lithium_plating_risk': 0.0
        }
    
    # Initial resistance
    resistance_initial = resistance_history[0] if len(resistance_history) > 0 else 0.05
    
    # Resistance growth rate
    if len(resistance_history) > 1 and len(cycle_indices) > 1:
        valid_mask = ~np.isnan(resistance_history) & (resistance_history > 0)
        if valid_mask.sum() > 1:
            coeffs = np.polyfit(cycle_indices[valid_mask], resistance_history[valid_mask], 1)
            resistance_growth_rate = coeffs[0]  # Positive slope = growth
        else:
            resistance_growth_rate = 0.0
    else:
        resistance_growth_rate = 0.0
    
    # Lithium plating risk: higher resistance + low temperature = higher risk
    lithium_plating_risk = 0.0
    if temperature is not None:
        # Cold temperature increases plating risk
        if temperature < 0:
            temp_factor = 0.8
        elif temperature < 10:
            temp_factor = 0.5
        else:
            temp_factor = 0.0
        
        # High resistance also increases risk
        if resistance_initial > 0.1:
            resistance_factor = 0.3
        elif resistance_initial > 0.05:
            resistance_factor = 0.1
        else:
            resistance_factor = 0.0
        
        lithium_plating_risk = min(1.0, temp_factor + resistance_factor)
    
    return {
        'resistance_initial': float(np.clip(resistance_initial, 0.01, 1.0)),
        'resistance_growth_rate': float(np.clip(resistance_growth_rate, -0.001, 0.01)),
        'lithium_plating_risk': float(np.clip(lithium_plating_risk, 0.0, 1.0))
    }


def extract_lithium_inventory_features(
    voltage: np.ndarray,
    capacity: np.ndarray,
    current: Optional[np.ndarray] = None,
    resistance: Optional[float] = None,
    capacity_history: Optional[np.ndarray] = None,
    cycle_indices: Optional[np.ndarray] = None,
    temperature: Optional[float] = None,
    nominal_capacity: float = 2.0,
    theoretical_capacity: Optional[float] = None
) -> LithiumInventoryFeatures:
    """
    Extract comprehensive lithium inventory features for a battery cycle.
    
    This function combines all physics-based features that differentiate
    battery chemistries and enable zero-shot RUL generalization.
    
    Args:
        voltage: Voltage measurements (V)
        capacity: Capacity measurements (Ah)
        current: Current measurements (A) - optional
        resistance: Internal resistance (Ohm) - optional
        capacity_history: Historical capacity values across cycles - optional
        cycle_indices: Cycle numbers - optional
        temperature: Operating temperature (°C) - optional
        nominal_capacity: Nominal capacity (Ah)
        theoretical_capacity: Theoretical capacity if known (Ah)
    
    Returns:
        LithiumInventoryFeatures object with all extracted features
    """
    # Extract voltage profile features
    voltage_features = extract_voltage_profile_features(voltage, capacity, current)
    
    # Extract capacity features
    if capacity_history is not None and cycle_indices is not None:
        capacity_features = extract_capacity_features(
            capacity_history, cycle_indices, nominal_capacity, theoretical_capacity
        )
    else:
        capacity_features = {
            'theoretical_capacity_ratio': 1.0,
            'capacity_fade_rate': 0.0,
            'lithium_utilization': 0.9
        }
    
    # Extract resistance features
    if resistance is not None:
        resistance_array = np.array([resistance])
        cycle_array = np.array([0]) if cycle_indices is None else cycle_indices
        resistance_features = extract_resistance_features(resistance_array, cycle_array, temperature)
    else:
        resistance_features = {
            'resistance_initial': 0.05,
            'resistance_growth_rate': 0.0,
            'lithium_plating_risk': 0.0
        }
    
    # Combine all features
    return LithiumInventoryFeatures(
        voltage_plateau_center=voltage_features['voltage_plateau_center'],
        voltage_plateau_width=voltage_features['voltage_plateau_width'],
        voltage_slope=voltage_features['voltage_slope'],
        theoretical_capacity_ratio=capacity_features['theoretical_capacity_ratio'],
        capacity_fade_rate=capacity_features['capacity_fade_rate'],
        resistance_growth_rate=resistance_features['resistance_growth_rate'],
        resistance_initial=resistance_features['resistance_initial'],
        lithium_utilization=capacity_features['lithium_utilization'],
        lithium_plating_risk=resistance_features['lithium_plating_risk'],
        voltage_profile_entropy=voltage_features['voltage_profile_entropy'],
        discharge_curve_curvature=voltage_features['discharge_curve_curvature']
    )


def lithium_inventory_to_vector(features: LithiumInventoryFeatures) -> np.ndarray:
    """
    Convert LithiumInventoryFeatures to a normalized feature vector.
    
    Returns:
        Normalized feature vector of shape (11,)
    """
    # Normalize each feature to [0, 1] or [-1, 1] range
    return np.array([
        features.voltage_plateau_center / 5.0,  # Normalize to [0, 1] (assuming 0-5V)
        features.voltage_plateau_width / 2.0,   # Normalize to [0, 1] (assuming 0-2V)
        (features.voltage_slope + 1.0) / 2.0,   # Normalize to [0, 1] (assuming -1 to 1 V/Ah)
        features.theoretical_capacity_ratio,    # Already in [0.5, 1.5], clip to [0, 1]
        (features.capacity_fade_rate + 0.01) / 0.02,  # Normalize to [0, 1]
        features.resistance_growth_rate * 100,  # Scale up, then normalize
        features.resistance_initial / 1.0,     # Normalize to [0, 1] (assuming 0-1 Ohm)
        features.lithium_utilization,          # Already in [0, 1]
        features.lithium_plating_risk,         # Already in [0, 1]
        features.voltage_profile_entropy / 5.0, # Normalize to [0, 1]
        np.tanh(features.discharge_curve_curvature)  # Normalize to [-1, 1] then to [0, 1]
    ], dtype=np.float32)


# Chemistry-specific theoretical capacities (Ah per gram of active material)
THEORETICAL_CAPACITIES = {
    'LCO': 0.274,   # LiCoO2: 274 mAh/g
    'NMC': 0.280,   # LiNiMnCoO2: ~280 mAh/g (varies by ratio)
    'NCA': 0.279,   # LiNiCoAlO2: ~279 mAh/g
    'LFP': 0.170,   # LiFePO4: 170 mAh/g
    'LMO': 0.148,   # LiMn2O4: 148 mAh/g
}


def estimate_theoretical_capacity(
    chemistry: str,
    cell_mass: Optional[float] = None,
    nominal_capacity: Optional[float] = None
) -> float:
    """
    Estimate theoretical capacity based on chemistry.
    
    Args:
        chemistry: Battery chemistry (LCO, NMC, LFP, etc.)
        cell_mass: Cell mass in grams (optional)
        nominal_capacity: Nominal capacity in Ah (optional, used as fallback)
    
    Returns:
        Estimated theoretical capacity in Ah
    """
    chemistry = chemistry.upper()
    
    if chemistry in THEORETICAL_CAPACITIES:
        if cell_mass is not None:
            # Estimate based on active material content (typically 70-80% of cell mass)
            active_material_mass = cell_mass * 0.75
            theoretical = active_material_mass * THEORETICAL_CAPACITIES[chemistry]
            return theoretical
        elif nominal_capacity is not None:
            # Use nominal capacity as approximation (typically 80-90% of theoretical)
            return nominal_capacity / 0.85
        else:
            # Default estimate
            return 2.0  # Typical for 18650 cells
    
    # Unknown chemistry - use default
    return nominal_capacity if nominal_capacity is not None else 2.0

