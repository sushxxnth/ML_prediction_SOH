"""
Enhanced Battery Advisor with Multi-Modal Early Warning

Integrates:
- Phase 2 Unified Model (SOH/RUL for cycling/storage)
- Multi-Modal v2 Model (early warning with EIS data)
- Warning Engine (4-level alerts)
- Suggestion Generator (mode-based advice)
"""

import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.advisory.warning_engine import (
    WarningEngine, WarningLevel, WarningResult,
    get_warning_color, get_warning_emoji
)
from src.advisory.suggestion_generator import (
    SuggestionGenerator, Suggestion, UsageContext, DegradationMode,
    DriverProfile, format_suggestions_for_display
)


@dataclass
class EnhancedHealthReport:
    """Enhanced battery health report with multi-modal insights."""
    # Status
    soh: float
    soh_pct: str
    rul_cycles: int
    rul_km: float           # NEW: Remaining useful life in kilometers
    rul_miles: float        # NEW: Remaining useful life in miles
    mode: DegradationMode
    
    # Warning
    warning_level: WarningLevel
    warning_emoji: str
    warning_color: str
    warning_message: str
    
    # Multi-modal early warning
    early_warning_triggered: bool
    early_warning_confidence: float
    early_warning_source: str  # "model", "threshold", "both"
    
    # Recommendations
    suggestions: List[Suggestion]
    top_recommendation: str
    
    # Trend
    degradation_rate: float
    rate_status: str
    cycles_to_warning: Optional[int]
    
    # Confidence
    confidence: float
    data_sources: List[str]


class EnhancedBatteryAdvisor:
    """
    Enhanced advisory system with multi-modal early warning.
    
    Combines:
    - SOH-based warnings (from unified model)
    - EIS-based early warning (from multi-modal model)
    - Mode-specific suggestions
    """
    
    # Optimal early warning threshold (from analysis)
    EARLY_WARNING_THRESHOLD = 0.30
    
    def __init__(
        self,
        unified_model_path: Optional[str] = None,
        multimodal_model_path: Optional[str] = None,
        device: str = 'cpu'
    ):
        self.device = device
        self.warning_engine = WarningEngine()
        self.suggestion_generator = SuggestionGenerator()
        
        self.unified_model = None
        self.multimodal_model = None
        
        # Load models
        if unified_model_path and Path(unified_model_path).exists():
            self._load_unified_model(unified_model_path)
        
        if multimodal_model_path and Path(multimodal_model_path).exists():
            self._load_multimodal_model(multimodal_model_path)
    
    def _load_unified_model(self, path: str):
        """Load the improved unified model v3."""
        from src.train.train_improved_v3 import ImprovedUnifiedModel
        
        self.unified_model = ImprovedUnifiedModel().to(self.device)
        state_dict = torch.load(path, map_location=self.device, weights_only=False)
        self.unified_model.load_state_dict(state_dict)
        self.unified_model.eval()
        print(f"  ✓ Improved unified model v3 loaded")
    
    def _load_multimodal_model(self, path: str):
        """Load multi-modal v2 model."""
        from src.train.train_multimodal_v2 import ImprovedMultiModalModel
        
        self.multimodal_model = ImprovedMultiModalModel().to(self.device)
        state_dict = torch.load(path, map_location=self.device, weights_only=False)
        self.multimodal_model.load_state_dict(state_dict)
        self.multimodal_model.eval()
        print(f"  ✓ Multi-modal model loaded")
    
    def analyze(
        self,
        capacity_features: np.ndarray,
        eis_features: Optional[np.ndarray] = None,
        context: np.ndarray = None,
        soh_history: Optional[List[float]] = None,
        chem_id: int = 0,
        driver_profile: str = "normal"  # NEW: "aggressive", "normal", or "eco"
    ) -> EnhancedHealthReport:
        """
        Analyze battery with multi-modal data.
        
        Args:
            capacity_features: 9D capacity/cycling features
            eis_features: 8D EIS impedance features (optional)
            context: 6D context vector
            soh_history: Recent SOH measurements
            chem_id: Chemistry ID
            driver_profile: Driving style - "aggressive", "normal", or "eco"
        """
        data_sources = ["capacity"]
        has_eis = eis_features is not None and np.sum(np.abs(eis_features)) > 0
        
        if has_eis:
            data_sources.append("eis")
        
        # Get predictions
        if self.unified_model is not None:
            soh, rul_norm, mode_pred = self._predict_unified(
                capacity_features, context, chem_id
            )
        else:
            soh = 0.85
            rul_norm = 0.5
            mode_pred = int(context[-1]) if context is not None and len(context) > 0 else 0
        
        # Get multi-modal early warning
        early_warning_score = 0.0
        early_warning_source = "threshold"
        
        if self.multimodal_model is not None and has_eis:
            early_warning_score = self._predict_early_warning(
                capacity_features, eis_features, context
            )
            early_warning_source = "model"
            data_sources.append("multimodal")
        
        # Combine warning sources
        ew_from_model = early_warning_score > self.EARLY_WARNING_THRESHOLD
        ew_from_soh = soh < 0.80
        early_warning_triggered = ew_from_model or ew_from_soh
        
        if ew_from_model and ew_from_soh:
            early_warning_source = "both"
        elif ew_from_soh:
            early_warning_source = "soh_threshold"
        
        # Convert predictions
        rul_cycles = int(rul_norm * 500)
        
        # Use context mode (context[5]) if provided, otherwise use model prediction
        # This ensures storage mode suggestions work correctly
        if context is not None and len(context) > 5 and context[5] > 0.5:
            mode = DegradationMode.STORAGE
        else:
            mode = DegradationMode.STORAGE if mode_pred == 1 else DegradationMode.CYCLING
        
        # Get warning from engine
        warning_result = self.warning_engine.evaluate(soh, rul_cycles, soh_history)
        
        # Upgrade warning if early warning triggered
        if early_warning_triggered and warning_result.level == WarningLevel.GREEN:
            warning_result.level = WarningLevel.YELLOW
            warning_result.message = "Early warning: approaching degradation threshold"
        
        # Map driver profile string to enum
        profile_map = {
            "aggressive": DriverProfile.AGGRESSIVE,
            "normal": DriverProfile.NORMAL,
            "eco": DriverProfile.ECO
        }
        driver_enum = profile_map.get(driver_profile.lower(), DriverProfile.NORMAL)
        
        # Generate suggestions
        usage_context = UsageContext(
            mode=mode,
            soh=soh,
            temperature=context[0] * 100 if context is not None and len(context) > 0 else 25,
            avg_soc=context[3] if context is not None and len(context) > 3 else 0.5,
            charge_rate=context[1] if context is not None and len(context) > 1 else 0.5,
            discharge_rate=context[2] if context is not None and len(context) > 2 else 0.5,
            deep_discharge_freq=0.1,
            driver_profile=driver_enum
        )
        
        suggestions = self.suggestion_generator.get_top_suggestions(usage_context, n=3)
        
        # Add early warning specific suggestion
        if early_warning_triggered:
            suggestions.insert(0, Suggestion(
                title="⚠️ Early warning: Plan for battery service",
                description="Multi-modal analysis indicates increased degradation risk. "
                           "Schedule a battery health check.",
                impact="Prevent unexpected failure",
                priority=suggestions[0].priority if suggestions else None
            ))
        
        top_rec = suggestions[0].title if suggestions else "Continue normal operation"
        
        # Rate status
        if warning_result.degradation_rate > 0.005:
            rate_status = "critical"
        elif warning_result.degradation_rate > 0.001:
            rate_status = "elevated"
        else:
            rate_status = "normal"
        
        # Calculate range-based RUL (km and miles)
        # Assumptions:
        # - Average EV range per full cycle: ~300 km (based on 60kWh battery, ~5km/kWh)
        # - This can be customized per vehicle in future
        KM_PER_CYCLE = 300.0  # Average km driven per charge cycle
        rul_km = rul_cycles * KM_PER_CYCLE
        rul_miles = rul_km * 0.621371
        
        return EnhancedHealthReport(
            soh=soh,
            soh_pct=f"{soh:.0%}",
            rul_cycles=rul_cycles,
            rul_km=rul_km,
            rul_miles=rul_miles,
            mode=mode,
            warning_level=warning_result.level,
            warning_emoji=get_warning_emoji(warning_result.level),
            warning_color=get_warning_color(warning_result.level),
            warning_message=warning_result.message,
            early_warning_triggered=early_warning_triggered,
            early_warning_confidence=float(early_warning_score),
            early_warning_source=early_warning_source,
            suggestions=suggestions[:3],
            top_recommendation=top_rec,
            degradation_rate=warning_result.degradation_rate,
            rate_status=rate_status,
            cycles_to_warning=warning_result.cycles_to_warning_zone,
            confidence=0.91 if has_eis else 0.85,
            data_sources=data_sources
        )
    
    def _predict_unified(self, features, context, chem_id):
        """Get predictions from unified model."""
        with torch.no_grad():
            feat_t = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)
            ctx_t = torch.tensor(context, dtype=torch.float32).unsqueeze(0).to(self.device)
            chem_t = torch.tensor([chem_id], dtype=torch.long).to(self.device)
            
            soh_pred, rul_pred, domain_logits, _ = self.unified_model(feat_t, ctx_t, chem_t)
            
            return (
                float(soh_pred.squeeze().cpu()),
                float(rul_pred.squeeze().cpu()),
                int(torch.argmax(domain_logits, dim=1).cpu())
            )
    
    def _predict_early_warning(self, capacity_features, eis_features, context):
        """Get early warning score from multi-modal model."""
        with torch.no_grad():
            cap_t = torch.tensor(capacity_features, dtype=torch.float32).unsqueeze(0).to(self.device)
            eis_t = torch.tensor(eis_features, dtype=torch.float32).unsqueeze(0).to(self.device)
            ctx_t = torch.tensor(context, dtype=torch.float32).unsqueeze(0).to(self.device)
            has_eis = torch.tensor([1.0], dtype=torch.float32).to(self.device)
            
            outputs = self.multimodal_model(cap_t, eis_t, ctx_t, has_eis)
            
            return float(outputs['early_warning'].squeeze().cpu())
    
    def format_report(self, report: EnhancedHealthReport) -> str:
        """Format enhanced report for display."""
        lines = [
            "┌──────────────────────────────────────────────────────────────────┐",
            "│              ENHANCED BATTERY HEALTH REPORT                      │",
            "│                    (Multi-Modal Analysis)                        │",
            "├──────────────────────────────────────────────────────────────────┤",
        ]
        
        # Status
        lines.append(f"│  Status: {report.warning_emoji} {report.warning_level.value.upper():8s} - {report.warning_message:30s}│")
        lines.append("│                                                                  │")
        
        # Metrics
        bar = '█' * int(report.soh * 10) + '░' * (10 - int(report.soh * 10))
        lines.append(f"│  State of Health:     {bar}  {report.soh_pct:6s}                    │")
        lines.append(f"│  Remaining Life:      ~{report.rul_cycles:4d} cycles                              │")
        lines.append(f"│  Mode:                {report.mode.value.capitalize():10s}                            │")
        lines.append("│                                                                  │")
        
        # Early warning
        ew_status = "⚠️  TRIGGERED" if report.early_warning_triggered else "✅ Normal"
        lines.append(f"│  Early Warning:       {ew_status:15s} (conf: {report.early_warning_confidence:.0%})       │")
        lines.append(f"│  Data Sources:        {', '.join(report.data_sources):35s}│")
        lines.append("│                                                                  │")
        
        lines.append("├──────────────────────────────────────────────────────────────────┤")
        lines.append("│  💡 Recommendations:                                             │")
        
        for i, s in enumerate(report.suggestions[:2], 1):
            title = s.title[:55]
            lines.append(f"│  {i}. {title:58s}│")
        
        lines.append("├──────────────────────────────────────────────────────────────────┤")
        lines.append(f"│  📊 Trend: {report.rate_status.capitalize():10s} degradation rate                      │")
        
        if report.cycles_to_warning:
            lines.append(f"│  ⏰ Cycles to warning zone: {report.cycles_to_warning:5d}                              │")
        
        lines.append(f"│  🎯 Confidence: {report.confidence:.0%}                                            │")
        lines.append("└──────────────────────────────────────────────────────────────────┘")
        
        return "\n".join(lines)


def demo():
    """Demo the enhanced advisor."""
    print("=" * 70)
    print("ENHANCED BATTERY ADVISOR - MULTI-MODAL DEMO")
    print("=" * 70)
    
    print("\n[1/2] Loading models...")
    advisor = EnhancedBatteryAdvisor(
        unified_model_path='reports/phase2_balanced/balanced_model.pt',
        multimodal_model_path='reports/multimodal_v2/improved_multimodal.pt'
    )
    
    print("\n[2/2] Running analysis...")
    
    # Test case: Battery with EIS data
    capacity_features = np.random.randn(9).astype(np.float32)
    eis_features = np.array([0.12, 0.05, 0.18, 45, 2.5, 0.01, -0.03, 0.008], dtype=np.float32)
    context = np.array([0.35, 0.8, 0.5, 0.6, 0, 0], dtype=np.float32)
    soh_history = [0.88, 0.86, 0.84, 0.82]
    
    print("\n--- Test: Battery with EIS Data (Potential Early Warning) ---")
    report = advisor.analyze(
        capacity_features=capacity_features,
        eis_features=eis_features,
        context=context,
        soh_history=soh_history
    )
    print(advisor.format_report(report))


if __name__ == '__main__':
    demo()
