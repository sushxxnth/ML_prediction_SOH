"""
Battery Advisory System - Warning Engine

Provides 4-level warnings based on:
- State of Health (SOH)
- Remaining Useful Life (RUL)
- Degradation rate trends
"""

from enum import Enum
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np


class WarningLevel(Enum):
    """4-level warning system for battery health."""
    GREEN = "green"      # Healthy, normal operation
    YELLOW = "yellow"    # Optimize usage recommended
    ORANGE = "orange"    # Schedule replacement soon
    RED = "red"          # Critical, replace immediately


@dataclass
class WarningResult:
    """Result of warning evaluation."""
    level: WarningLevel
    soh: float
    rul_cycles: int
    degradation_rate: float
    message: str
    recommendation: str
    cycles_to_warning_zone: Optional[int] = None


class WarningEngine:
    """
    Battery warning engine with safe-zone detection.
    
    Warning thresholds:
        GREEN:  SOH > 90%
        YELLOW: SOH 80-90%
        ORANGE: SOH 70-80%
        RED:    SOH < 70%
    """
    
    # Threshold configuration
    THRESHOLDS = {
        WarningLevel.GREEN: 0.90,
        WarningLevel.YELLOW: 0.80,
        WarningLevel.ORANGE: 0.70,
        WarningLevel.RED: 0.0
    }
    
    # Degradation rate thresholds (per cycle)
    FAST_DEGRADATION_RATE = 0.005  # 0.5% per cycle
    NORMAL_DEGRADATION_RATE = 0.001  # 0.1% per cycle
    
    def __init__(self, warning_soh: float = 0.80, critical_soh: float = 0.70):
        """
        Args:
            warning_soh: SOH threshold for entering warning zone
            critical_soh: SOH threshold for critical zone
        """
        self.warning_soh = warning_soh
        self.critical_soh = critical_soh
    
    def evaluate(
        self,
        soh: float,
        rul_cycles: int,
        soh_history: Optional[List[float]] = None
    ) -> WarningResult:
        """
        Evaluate battery status and return warning result.
        
        Args:
            soh: Current state of health (0-1)
            rul_cycles: Estimated remaining useful life in cycles
            soh_history: Recent SOH measurements for trend analysis
        
        Returns:
            WarningResult with warning level and recommendations
        """
        # Calculate degradation rate from history
        degradation_rate = self._calculate_degradation_rate(soh_history)
        
        # Check for accelerated aging first (overrides SOH-based level)
        if degradation_rate > self.FAST_DEGRADATION_RATE:
            level = WarningLevel.ORANGE
            message = "Accelerated aging detected"
        else:
            # SOH-based warning level
            level = self._get_level_from_soh(soh)
            message = self._get_message_for_level(level)
        
        # Calculate cycles until warning zone
        cycles_to_warning = self._calculate_cycles_to_threshold(
            soh, degradation_rate, self.warning_soh
        )
        
        # Get recommendation
        recommendation = self._get_recommendation(level, degradation_rate)
        
        return WarningResult(
            level=level,
            soh=soh,
            rul_cycles=rul_cycles,
            degradation_rate=degradation_rate,
            message=message,
            recommendation=recommendation,
            cycles_to_warning_zone=cycles_to_warning
        )
    
    def _get_level_from_soh(self, soh: float) -> WarningLevel:
        """Determine warning level based on SOH."""
        if soh > self.THRESHOLDS[WarningLevel.GREEN]:
            return WarningLevel.GREEN
        elif soh > self.THRESHOLDS[WarningLevel.YELLOW]:
            return WarningLevel.YELLOW
        elif soh > self.THRESHOLDS[WarningLevel.ORANGE]:
            return WarningLevel.ORANGE
        else:
            return WarningLevel.RED
    
    def _calculate_degradation_rate(
        self, 
        soh_history: Optional[List[float]]
    ) -> float:
        """Calculate degradation rate from SOH history."""
        if soh_history is None or len(soh_history) < 2:
            return 0.0
        
        # Use last 10 measurements for trend
        recent = soh_history[-10:] if len(soh_history) >= 10 else soh_history
        
        # Linear regression for trend
        n = len(recent)
        x = np.arange(n)
        y = np.array(recent)
        
        # Slope = (n*sum(xy) - sum(x)*sum(y)) / (n*sum(x²) - sum(x)²)
        slope = (n * np.sum(x * y) - np.sum(x) * np.sum(y)) / \
                (n * np.sum(x**2) - np.sum(x)**2 + 1e-8)
        
        return abs(slope)  # Return absolute value (rate is always positive for degradation)
    
    def _calculate_cycles_to_threshold(
        self,
        current_soh: float,
        degradation_rate: float,
        threshold: float
    ) -> Optional[int]:
        """Estimate cycles until reaching a threshold."""
        if degradation_rate <= 0 or current_soh <= threshold:
            return None
        
        cycles = int((current_soh - threshold) / degradation_rate)
        return max(1, cycles)
    
    def _get_message_for_level(self, level: WarningLevel) -> str:
        """Get user-friendly message for warning level."""
        messages = {
            WarningLevel.GREEN: "Battery healthy",
            WarningLevel.YELLOW: "Consider optimizing usage",
            WarningLevel.ORANGE: "Plan battery replacement",
            WarningLevel.RED: "Battery replacement needed"
        }
        return messages[level]
    
    def _get_recommendation(
        self, 
        level: WarningLevel, 
        degradation_rate: float
    ) -> str:
        """Get actionable recommendation based on level and rate."""
        if level == WarningLevel.GREEN:
            if degradation_rate > self.NORMAL_DEGRADATION_RATE:
                return "Battery healthy but degrading faster than normal. Consider reducing fast charging."
            return "Continue normal operation."
        
        elif level == WarningLevel.YELLOW:
            return "Reduce fast charging and avoid deep discharges to extend battery life."
        
        elif level == WarningLevel.ORANGE:
            return "Schedule battery replacement within the next 3-6 months."
        
        else:  # RED
            return "Replace battery as soon as possible to avoid unexpected failure."


def get_warning_color(level: WarningLevel) -> str:
    """Get hex color for warning level visualization."""
    colors = {
        WarningLevel.GREEN: "#27AE60",
        WarningLevel.YELLOW: "#F1C40F",
        WarningLevel.ORANGE: "#E67E22",
        WarningLevel.RED: "#E74C3C"
    }
    return colors[level]


def get_warning_emoji(level: WarningLevel) -> str:
    """Get emoji for warning level."""
    emojis = {
        WarningLevel.GREEN: "🟢",
        WarningLevel.YELLOW: "🟡",
        WarningLevel.ORANGE: "🟠",
        WarningLevel.RED: "🔴"
    }
    return emojis[level]


if __name__ == '__main__':
    # Test the warning engine
    engine = WarningEngine()
    
    # Test cases
    test_cases = [
        (0.95, 500, None),
        (0.85, 200, [0.88, 0.87, 0.86, 0.85]),
        (0.75, 50, [0.82, 0.80, 0.78, 0.76, 0.75]),
        (0.65, 10, [0.70, 0.68, 0.66, 0.65]),
    ]
    
    print("Warning Engine Test Results")
    print("=" * 60)
    
    for soh, rul, history in test_cases:
        result = engine.evaluate(soh, rul, history)
        emoji = get_warning_emoji(result.level)
        print(f"\n{emoji} SOH: {soh:.0%}, RUL: {rul} cycles")
        print(f"   Level: {result.level.value.upper()}")
        print(f"   Message: {result.message}")
        print(f"   Recommendation: {result.recommendation}")
        if result.cycles_to_warning_zone:
            print(f"   Cycles to warning zone: {result.cycles_to_warning_zone}")
