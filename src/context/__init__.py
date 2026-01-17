"""
Extended Context Module for Multi-Dimensional Battery Analysis

This module provides rich context representations for battery trajectories,
enabling cross-domain transfer learning across:
- Temperature conditions
- Battery chemistries
- Usage/driving profiles
- Charge/discharge rates

Author: Battery ML Research
"""

from .extended_context import (
    # Enums
    TemperatureContext,
    ChemistryContext,
    UsageProfileContext,
    CRateContext,
    
    # Data class
    ExtendedBatteryContext,
    
    # Constants
    CONTEXT_DIMS,
    TOTAL_ONE_HOT_DIM,
    TOTAL_HYBRID_DIM,
    
    # Helper functions
    create_context_from_metadata,
    batch_contexts_to_tensor,
    context_similarity,
    
    # Dataset-specific builders
    create_nasa_context,
    create_sandia_context,
    create_calce_context,
    create_oxford_context,
    create_tbsi_sunwoda_context,
)

__all__ = [
    'TemperatureContext',
    'ChemistryContext',
    'UsageProfileContext',
    'CRateContext',
    'ExtendedBatteryContext',
    'CONTEXT_DIMS',
    'TOTAL_ONE_HOT_DIM',
    'TOTAL_HYBRID_DIM',
    'create_context_from_metadata',
    'batch_contexts_to_tensor',
    'context_similarity',
    'create_nasa_context',
    'create_sandia_context',
    'create_calce_context',
    'create_oxford_context',
    'create_tbsi_sunwoda_context',
]

