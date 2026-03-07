# Models package
# Use try-except to avoid import errors for missing dependencies

try:
    from .retrieval_augmented_dynamics import RADModel, MemoryBank, TrajectoryEncoder, RetrievalAttentionFusion
except ImportError:
    pass

try:
    from .rad_model import FleetRADModel, create_context_tensor
except ImportError:
    pass

try:
    from .causal_attribution import CausalAttributionModel, CausalExplainer, DegradationMechanism
except ImportError:
    pass

try:
    from .multimodal_fusion import ImprovedMultiModalModel
except ImportError:
    pass

__all__ = [
    'RADModel', 
    'MemoryBank', 
    'TrajectoryEncoder', 
    'RetrievalAttentionFusion',
    'FleetRADModel',
    'create_context_tensor',
    'CausalAttributionModel',
    'CausalExplainer',
    'DegradationMechanism',
    'ImprovedMultiModalModel',
]
