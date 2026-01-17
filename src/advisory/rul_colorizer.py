"""
RUL Color Coding - Visual Health Indicator
Provides color-coded health status based on remaining range
"""

from enum import Enum
from dataclasses import dataclass
from typing import Tuple


class HealthStatus(Enum):
    """Battery health status categories."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CRITICAL = "critical"


@dataclass
class RULColorConfig:
    """Configuration for RUL color coding thresholds."""
    healthy_threshold_km: float = 90000  # >= 90k km = Blue (Healthy)
    critical_threshold_km: float = 50000  # < 50k km = Red (Critical, immediate replacement)
    
    # Color codes
    color_healthy: str = "blue"
    color_degraded: str = "purple"
    color_critical: str = "red"
    
    # Emoji indicators
    emoji_healthy: str = "🔵"
    emoji_degraded: str = "🟣"
    emoji_critical: str = "🔴"


class RULColorizer:
    """
    Assigns visual health indicators (colors/emojis) based on remaining range.
    
    This provides instant visual feedback to users about battery health
    without needing to interpret technical metrics.
    """
    
    def __init__(self, config: RULColorConfig = None):
        """
        Initialize colorizer with configuration.
        
        Args:
            config: Color coding configuration (uses defaults if None)
        """
        self.config = config or RULColorConfig()
    
    def get_health_status(self, rul_km: float) -> HealthStatus:
        """
        Determine health status from RUL in kilometers.
        
        Args:
            rul_km: Remaining useful life in kilometers
            
        Returns:
            HealthStatus enum value
        """
        if rul_km >= self.config.healthy_threshold_km:
            return HealthStatus.HEALTHY
        elif rul_km >= self.config.critical_threshold_km:
            return HealthStatus.DEGRADED
        else:
            return HealthStatus.CRITICAL
    
    def get_color(self, rul_km: float) -> str:
        """
        Get color code for given RUL.
        
        Args:
            rul_km: Remaining useful life in kilometers
            
        Returns:
            Color string ('blue', 'purple', or 'red')
        """
        status = self.get_health_status(rul_km)
        
        if status == HealthStatus.HEALTHY:
            return self.config.color_healthy
        elif status == HealthStatus.DEGRADED:
            return self.config.color_degraded
        else:
            return self.config.color_critical
    
    def get_emoji(self, rul_km: float) -> str:
        """
        Get emoji indicator for given RUL.
        
        Args:
            rul_km: Remaining useful life in kilometers
            
        Returns:
            Emoji string ('🔵', '🟣', or '🔴')
        """
        status = self.get_health_status(rul_km)
        
        if status == HealthStatus.HEALTHY:
            return self.config.emoji_healthy
        elif status == HealthStatus.DEGRADED:
            return self.config.emoji_degraded
        else:
            return self.config.emoji_critical
    
    def get_status_message(self, rul_km: float) -> str:
        """
        Get user-friendly status message.
        
        Args:
            rul_km: Remaining useful life in kilometers
            
        Returns:
            Human-readable status message
        """
        status = self.get_health_status(rul_km)
        
        if status == HealthStatus.HEALTHY:
            years = rul_km / 15000  # Assume 15k km/year average
            return f"Battery healthy - approximately {years:.1f} years of life remaining"
        elif status == HealthStatus.DEGRADED:
            return "Battery aging - consider replacement planning within 2-3 years"
        else:
            return "Battery critically degraded - replacement recommended soon"
    
    def get_full_status(self, rul_km: float) -> Tuple[str, str, str]:
        """
        Get complete status information (color, emoji, message).
        
        Args:
            rul_km: Remaining useful life in kilometers
            
        Returns:
            Tuple of (color, emoji, message)
        """
        return (
            self.get_color(rul_km),
            self.get_emoji(rul_km),
            self.get_status_message(rul_km)
        )


def calculate_rul_km(soh: float, max_range_km: float, rul_cycles: int) -> float:
    """
    Convert RUL from cycles to kilometers.
    
    Args:
        soh: State of Health (0.0 - 1.0)
        max_range_km: Maximum range per charge at 100% SOH (km)
        rul_cycles: Remaining useful life in charge cycles
        
    Returns:
        RUL in kilometers
    """
    current_range_per_charge = soh * max_range_km
    total_rul_km = current_range_per_charge * rul_cycles
    return total_rul_km


if __name__ == '__main__':
    # Example usage
    colorizer = RULColorizer()
    
    test_cases = [
        ("Brand New", 126000),
        ("Well Maintained", 94000),
        ("Aging", 88000),
        ("Degraded", 66000),
        ("Critical", 45000),
    ]
    
    print("="*60)
    print("RUL Color Coding Examples")
    print("="*60)
    
    for name, rul in test_cases:
        color, emoji, message = colorizer.get_full_status(rul)
        print(f"\n{name}: {rul/1000:.0f}k km")
        print(f"  Color: {color} {emoji}")
        print(f"  Status: {message}")
