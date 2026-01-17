"""
Integration of Lithium Inventory Features into Data Pipeline

This module provides utilities to extract and integrate lithium inventory
features into the battery data pipeline for zero-shot RUL generalization.
"""

import numpy as np
from typing import List, Optional, Dict
from src.data.base_loader import CellData, CycleData
from src.features.lithium_inventory import (
    extract_lithium_inventory_features,
    lithium_inventory_to_vector,
    estimate_theoretical_capacity
)


def extract_lithium_features_for_cell(cell: CellData) -> Dict[int, np.ndarray]:
    """
    Extract lithium inventory features for all cycles in a cell.
    
    Args:
        cell: CellData object with cycles
    
    Returns:
        Dictionary mapping cycle_index to lithium inventory feature vector
    """
    lithium_features = {}
    
    # Get capacity history for all cycles
    capacity_history = np.array([c.capacity for c in cell.cycles])
    cycle_indices = np.array([c.cycle_index for c in cell.cycles])
    
    # Estimate theoretical capacity based on chemistry
    theoretical_capacity = estimate_theoretical_capacity(
        chemistry=cell.chemistry,
        nominal_capacity=cell.nominal_capacity
    )
    
    for cycle in cell.cycles:
        # For lithium inventory extraction, we need voltage/capacity time series
        # Since we only have aggregated values, we create a synthetic profile
        # based on typical battery discharge characteristics
        
        # Estimate voltage profile from min/max
        # Typical discharge: starts at max, ends at min
        # Create a realistic discharge curve (exponential decay)
        n_points = 100
        voltage_start = cycle.voltage_max if not np.isnan(cycle.voltage_max) else 4.2
        voltage_end = cycle.voltage_min if not np.isnan(cycle.voltage_min) else 2.5
        
        # Create exponential discharge curve (more realistic than linear)
        t = np.linspace(0, 1, n_points)
        # Exponential decay: V = V_start * exp(-alpha * t) + V_end * (1 - exp(-alpha * t))
        alpha = 2.0  # Decay rate
        voltage_profile = voltage_start * np.exp(-alpha * t) + voltage_end * (1 - np.exp(-alpha * t))
        
        # Capacity profile: linear discharge (capacity increases linearly)
        capacity_profile = np.linspace(0, cycle.capacity, n_points)
        
        # Extract lithium inventory features
        try:
            li_features = extract_lithium_inventory_features(
                voltage=voltage_profile,
                capacity=capacity_profile,
                current=None,  # Not available in aggregated data
                resistance=cycle.internal_resistance if not np.isnan(cycle.internal_resistance) else None,
                capacity_history=capacity_history,
                cycle_indices=cycle_indices,
                temperature=cycle.temperature_mean if not np.isnan(cycle.temperature_mean) else cell.test_temperature,
                nominal_capacity=cell.nominal_capacity,
                theoretical_capacity=theoretical_capacity
            )
            
            # Convert to feature vector
            li_vector = lithium_inventory_to_vector(li_features)
            lithium_features[cycle.cycle_index] = li_vector
            
        except Exception as e:
            # Fallback: create default feature vector
            print(f"Warning: Could not extract lithium features for {cell.cell_id} cycle {cycle.cycle_index}: {e}")
            lithium_features[cycle.cycle_index] = np.zeros(11, dtype=np.float32)
    
    return lithium_features


def augment_cycle_with_lithium_features(
    cycle: CycleData,
    cell: CellData,
    lithium_features: Optional[Dict[int, np.ndarray]] = None
) -> np.ndarray:
    """
    Augment cycle feature vector with lithium inventory features.
    
    Args:
        cycle: CycleData object
        cell: Parent CellData object (for context)
        lithium_features: Pre-computed lithium features (optional)
    
    Returns:
        Augmented feature vector (original 9 features + 11 lithium features = 20 total)
    """
    # Get original features
    original_features = cycle.to_feature_vector()  # Shape: (9,)
    
    # Get lithium inventory features
    if lithium_features is None:
        lithium_features = extract_lithium_features_for_cell(cell)
    
    if cycle.cycle_index in lithium_features:
        li_features = lithium_features[cycle.cycle_index]
    else:
        # Fallback: default lithium features
        li_features = np.zeros(11, dtype=np.float32)
    
    # Concatenate
    augmented_features = np.concatenate([original_features, li_features])
    
    return augmented_features


def get_lithium_feature_names() -> List[str]:
    """
    Get names of lithium inventory features for interpretability.
    
    Returns:
        List of feature names
    """
    return [
        'voltage_plateau_center',
        'voltage_plateau_width',
        'voltage_slope',
        'theoretical_capacity_ratio',
        'capacity_fade_rate',
        'resistance_growth_rate',
        'resistance_initial',
        'lithium_utilization',
        'lithium_plating_risk',
        'voltage_profile_entropy',
        'discharge_curve_curvature'
    ]


def get_augmented_feature_dim() -> int:
    """
    Get the dimension of augmented feature vector.
    
    Returns:
        Total feature dimension (9 original + 11 lithium = 20)
    """
    return 9 + 11  # Original features + lithium inventory features

