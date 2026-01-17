"""
Battery Advisory System - Suggestion Generator

Provides mode-specific suggestions based on:
- Degradation mode (cycling vs storage)
- Current usage patterns
- Battery physics principles
"""

from enum import Enum
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple

#RUL
class HealthStatus(Enum):
    """Battery health status based on remaining range."""
    HEALTHY = "healthy"      # >= 90k km (Blue)
    DEGRADED = "degraded"    # 50-90k km (Purple)
    CRITICAL = "critical"    # < 50k km (Red)


@dataclass
class RULColorConfig:
    """Configuration for RUL visual indicators."""
    healthy_threshold_km: float = 90000   # >= 90k = Blue
    critical_threshold_km: float = 50000  # < 50k = Red
    
    # Colors
    color_healthy: str = "blue"
    color_degraded: str = "purple"
    color_critical: str = "red"
    

    emoji_healthy: str = "🔵"
    emoji_degraded: str = "🟣"
    emoji_critical: str = "🔴"


def calculate_rul_km(soh: float, max_range_km: float, rul_cycles: int) -> float:
    """
    Convert RUL from cycles to kilometers.
    
    Args:
        soh: State of Health (0.0 - 1.0)
        max_range_km: Maximum range per charge at 100% SOH
        rul_cycles: Remaining cycles until 80% SOH
        
    Returns:
        Total remaining range in kilometers
        
    Example:
        >>> calculate_rul_km(0.93, 380, 265)
        93545.0  # ~94k km
    """
    current_range_per_charge = soh * max_range_km
    return current_range_per_charge * rul_cycles


def get_rul_health_status(rul_km: float, config: RULColorConfig = None) -> HealthStatus:
    """
    Determine health status from RUL.
    
    Args:
        rul_km: Remaining useful life in kilometers
        config: Color configuration (uses defaults if None)
        
    Returns:
        HealthStatus enum
    """
    if config is None:
        config = RULColorConfig()
    
    if rul_km >= config.healthy_threshold_km:
        return HealthStatus.HEALTHY
    elif rul_km >= config.critical_threshold_km:
        return HealthStatus.DEGRADED
    else:
        return HealthStatus.CRITICAL


def get_rul_color(rul_km: float, config: RULColorConfig = None) -> str:
    """
    Get color code for RUL visualization.
    
    Args:
        rul_km: Remaining useful life in kilometers
        config: Color configuration
        
    Returns:
        Color string ('blue', 'purple', or 'red')
    """
    if config is None:
        config = RULColorConfig()
    
    status = get_rul_health_status(rul_km, config)
    
    if status == HealthStatus.HEALTHY:
        return config.color_healthy
    elif status == HealthStatus.DEGRADED:
        return config.color_degraded
    else:
        return config.color_critical


def get_rul_emoji(rul_km: float, config: RULColorConfig = None) -> str:
    """
    Get emoji indicator for RUL.
    
    Args:
        rul_km: Remaining useful life in kilometers
        config: Color configuration
        
    Returns:
        Emoji string ('🔵', '🟣', or '🔴')
    """
    if config is None:
        config = RULColorConfig()
    
    status = get_rul_health_status(rul_km, config)
    
    if status == HealthStatus.HEALTHY:
        return config.emoji_healthy
    elif status == HealthStatus.DEGRADED:
        return config.emoji_degraded
    else:
        return config.emoji_critical


def get_rul_status_message(rul_km: float) -> str:
    """
    Get user-friendly status message.
    
    Args:
        rul_km: Remaining useful life in kilometers
        
    Returns:
        Human-readable message
    """
    status = get_rul_health_status(rul_km)
    
    if status == HealthStatus.HEALTHY:
        years = rul_km / 15000  # Assume 15k km/year
        return f"Battery healthy - approximately {years:.1f} years remaining"
    elif status == HealthStatus.DEGRADED:
        return "Battery aging - plan replacement within 2-3 years"
    else:
        return "Battery critical - replacement recommended soon"


def get_rul_full_status(rul_km: float) -> Tuple[str, str, str]:
    """
    Get complete RUL status (color, emoji, message).
    
    Args:
        rul_km: Remaining useful life in kilometers
        
    Returns:
        Tuple of (color, emoji, message)
    """
    return (
        get_rul_color(rul_km),
        get_rul_emoji(rul_km),
        get_rul_status_message(rul_km)
    )



# SUGGESTION SYSTEM



class DegradationMode(Enum):
    """Battery degradation mode."""
    CYCLING = "cycling"
    STORAGE = "storage"


class DriverProfile(Enum):
    """Driving style profile - affects degradation rate."""
    AGGRESSIVE = "aggressive"  # Hard acceleration, high discharge
    NORMAL = "normal"          # Standard driving
    ECO = "eco"                # Gentle, energy-efficient


class SuggestionPriority(Enum):
    """Priority level for suggestions."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class Suggestion:
    """A single actionable suggestion."""
    title: str
    description: str
    impact: str  # Expected impact on battery life
    priority: SuggestionPriority
    mode: Optional[DegradationMode] = None  # None = applies to both


@dataclass
class UsageContext:
    """Current usage context for generating suggestions."""
    mode: DegradationMode
    soh: float
    temperature: float  # Celsius
    avg_soc: float  # Average state of charge (0-1)
    charge_rate: float  # C-rate
    discharge_rate: float  # C-rate
    deep_discharge_freq: float  # Frequency of discharges below 20%
    driver_profile: DriverProfile = DriverProfile.NORMAL  # NEW: Driver profile
    # Trip planning (optional)
    planned_trip_km: Optional[float] = None  # Planned trip distance in km
    current_range_km: Optional[float] = None  # Current available range in km
    max_range_km: Optional[float] = None  # Maximum range at 100% SOC


class SuggestionGenerator:
    """
    Generates personalized suggestions to maximize battery life.
    
    Suggestions are based on:
    - Current degradation mode (cycling vs storage)
    - Usage patterns
    - Battery physics principles
    """
    
    # Physics-based thresholds
    HIGH_TEMP_THRESHOLD = 35  # Celsius
    LOW_TEMP_THRESHOLD = 10
    HIGH_SOC_STORAGE = 0.80  # SOC above which storage is harmful
    LOW_SOC_STORAGE = 0.20
    OPTIMAL_SOC_STORAGE = 0.50
    HIGH_CRATE = 1.0
    DEEP_DISCHARGE_SOC = 0.20
    
    def generate(self, context: UsageContext) -> List[Suggestion]:
        """
        Generate suggestions based on usage context.
        
        Args:
            context: Current usage context
            
        Returns:
            List of suggestions sorted by priority
        """
        suggestions = []
        
        # Mode-specific suggestions
        if context.mode == DegradationMode.CYCLING:
            suggestions.extend(self._cycling_suggestions(context))
        else:
            suggestions.extend(self._storage_suggestions(context))
        
        # Driver profile suggestions (NEW)
        suggestions.extend(self._driver_profile_suggestions(context))
        
        # Trip planning suggestions (NEW)
        suggestions.extend(self._trip_planning_suggestions(context))
        
        # General suggestions (apply to both modes)
        suggestions.extend(self._general_suggestions(context))
        
        # Sort by priority
        priority_order = {
            SuggestionPriority.HIGH: 0,
            SuggestionPriority.MEDIUM: 1,
            SuggestionPriority.LOW: 2
        }
        suggestions.sort(key=lambda s: priority_order[s.priority])
        
        return suggestions
    
    def _cycling_suggestions(self, context: UsageContext) -> List[Suggestion]:
        """Generate cycling-mode specific suggestions."""
        suggestions = []
        
        # High C-rate charging
        if context.charge_rate > self.HIGH_CRATE:
            suggestions.append(Suggestion(
                title="Reduce fast charging frequency",
                description=f"Current charge rate ({context.charge_rate:.1f}C) accelerates degradation. "
                           f"Use slower charging when possible.",
                impact="Can extend battery life by 10-15%",
                priority=SuggestionPriority.HIGH,
                mode=DegradationMode.CYCLING
            ))
        
        # Deep discharge
        if context.deep_discharge_freq > 0.2:  # More than 20% of cycles are deep discharge
            suggestions.append(Suggestion(
                title="Avoid deep discharges",
                description="Frequent discharges below 20% SOC cause accelerated aging. "
                           "Consider charging before reaching low battery.",
                impact="Can extend battery life by 15-20%",
                priority=SuggestionPriority.HIGH,
                mode=DegradationMode.CYCLING
            ))
        
        # High temperature cycling
        if context.temperature > self.HIGH_TEMP_THRESHOLD:
            suggestions.append(Suggestion(
                title="Cool battery during active use",
                description=f"Operating at {context.temperature:.0f}°C accelerates degradation. "
                           f"Ensure adequate cooling during charging/discharging.",
                impact="Can reduce degradation rate by 20-30%",
                priority=SuggestionPriority.HIGH,
                mode=DegradationMode.CYCLING
            ))
        
        # High discharge rate
        if context.discharge_rate > self.HIGH_CRATE:
            suggestions.append(Suggestion(
                title="Reduce peak power demands",
                description="High discharge rates stress the battery. Consider smoother acceleration "
                           "or reduced power mode.",
                impact="Can extend battery life by 5-10%",
                priority=SuggestionPriority.MEDIUM,
                mode=DegradationMode.CYCLING
            ))
        
        return suggestions
    
    def _storage_suggestions(self, context: UsageContext) -> List[Suggestion]:
        """Generate storage-mode specific suggestions."""
        suggestions = []
        
        # High SOC storage
        if context.avg_soc > self.HIGH_SOC_STORAGE:
            suggestions.append(Suggestion(
                title=f"Store at 50-60% SOC",
                description=f"Currently stored at {context.avg_soc:.0%} SOC. "
                           f"High SOC accelerates calendar aging. Discharge to ~50% for long storage.",
                impact="Can reduce calendar aging by 30-40%",
                priority=SuggestionPriority.HIGH,
                mode=DegradationMode.STORAGE
            ))
        
        # Low SOC storage
        elif context.avg_soc < self.LOW_SOC_STORAGE:
            suggestions.append(Suggestion(
                title="Increase storage SOC to 40-50%",
                description=f"Currently at {context.avg_soc:.0%} SOC. Very low SOC during storage "
                           f"can cause copper dissolution. Charge to ~50%.",
                impact="Prevents irreversible damage",
                priority=SuggestionPriority.HIGH,
                mode=DegradationMode.STORAGE
            ))
        
        # High temperature storage
        if context.temperature > self.HIGH_TEMP_THRESHOLD:
            suggestions.append(Suggestion(
                title="Store in cool environment",
                description=f"Storage at {context.temperature:.0f}°C accelerates calendar aging. "
                           f"Ideal storage temperature is 15-25°C.",
                impact="Can reduce calendar aging by 50%+",
                priority=SuggestionPriority.HIGH,
                mode=DegradationMode.STORAGE
            ))
        
        # General storage advice
        suggestions.append(Suggestion(
            title="Cycle battery periodically",
            description="For long-term storage, perform a full charge-discharge cycle monthly "
                       "to prevent lithium plating.",
            impact="Maintains electrode health",
            priority=SuggestionPriority.MEDIUM,
            mode=DegradationMode.STORAGE
        ))
        
        return suggestions
    
    def _driver_profile_suggestions(self, context: UsageContext) -> List[Suggestion]:
        """Generate driver profile-specific suggestions."""
        suggestions = []
        
        # Only apply for cycling mode
        if context.mode != DegradationMode.CYCLING:
            return suggestions
        
        if context.driver_profile == DriverProfile.AGGRESSIVE:
            suggestions.append(Suggestion(
                title="Aggressive driving detected - Consider switching to Eco mode",
                description="Aggressive driving (hard acceleration/braking) increases discharge peaks "
                           "and thermal stress. Switching to smoother driving can extend battery life.",
                impact="Can extend battery life by 15-25%",
                priority=SuggestionPriority.HIGH,
                mode=DegradationMode.CYCLING
            ))
            
            # Additional aggressive-specific advice
            if context.temperature > 30:
                suggestions.append(Suggestion(
                    title="Pre-condition battery before aggressive use",
                    description="Battery temperature is elevated. Allow cooling between "
                               "high-performance driving sessions.",
                    impact="Reduces thermal degradation",
                    priority=SuggestionPriority.HIGH,
                    mode=DegradationMode.CYCLING
                ))
        
        elif context.driver_profile == DriverProfile.ECO:
            # Eco drivers get positive reinforcement
            suggestions.append(Suggestion(
                title="Eco driving style - Optimal for battery longevity",
                description="Your gentle driving style minimizes stress on the battery. "
                           "Continue avoiding rapid acceleration and hard braking.",
                impact="Maximizing battery lifespan",
                priority=SuggestionPriority.LOW,
                mode=DegradationMode.CYCLING
            ))
        
        else:  # NORMAL
            # Normal drivers get balanced advice
            if context.discharge_rate > 0.8:
                suggestions.append(Suggestion(
                    title="Consider smoother acceleration patterns",
                    description="Moderate driving detected. Slightly gentler acceleration "
                               "during daily commutes can improve battery health.",
                    impact="Can extend battery life by 5-10%",
                    priority=SuggestionPriority.LOW,
                    mode=DegradationMode.CYCLING
                ))
        
        return suggestions
    
    def _trip_planning_suggestions(self, context: UsageContext) -> List[Suggestion]:
        """Generate trip planning-specific suggestions."""
        suggestions = []
        
        # Only applicable if trip is planned
        if context.planned_trip_km is None or context.current_range_km is None:
            return suggestions
        
        # Only for cycling mode
        if context.mode != DegradationMode.CYCLING:
            return suggestions
        
        trip_km = context.planned_trip_km
        current_range = context.current_range_km
        max_range = context.max_range_km or (current_range / context.avg_soc if context.avg_soc > 0 else 400)
        
        # Calculate required SOC for trip (with 15% safety margin)
        safety_margin = 1.15
        required_range = trip_km * safety_margin
        required_soc = required_range / max_range
        
        # Recommend not overcharging if current charge is sufficient or trip needs <80%
        if current_range >= required_range:
            suggestions.append(Suggestion(
                title=f"Trip feasible with current charge ({context.avg_soc:.0%})",
                description=f"Your {trip_km:.0f}km trip requires {required_range:.0f}km range (with 15% safety margin). "
                           f"Current range ({current_range:.0f}km) is sufficient. No charging needed.",
                impact="Preserves battery health by avoiding unnecessary charging",
                priority=SuggestionPriority.MEDIUM,
                mode=DegradationMode.CYCLING
            ))
        elif required_soc <= 0.80:
            # Trip needs charging but not to 100%
            recommended_soc = min(0.80, required_soc + 0.05)  # Add 5% buffer, cap at 80%
            suggestions.append(Suggestion(
                title=f"Charge to {recommended_soc:.0%} instead of 100%",
                description=f"Trip requires {required_range:.0f}km range ({recommended_soc:.0%} SOC with safety margin). "
                           f"Charging to 100% is unnecessary and increases SEI growth by 3×. "
                           f"Recommended charge: {recommended_soc:.0%}.",
                impact="Reduces degradation while ensuring trip completion",
                priority=SuggestionPriority.HIGH,
                mode=DegradationMode.CYCLING
            ))
        elif required_soc <= 0.95:
            # Trip needs high charge but warn about degradation
            suggestions.append(Suggestion(
                title=f"High charge required for trip ({required_soc:.0%})",
                description=f"Your {trip_km:.0f}km trip requires charging to {required_soc:.0%}. "
                           f"Consider route optimization or charging stops to reduce peak SOC.",
                impact="Balances trip needs with battery health",
                priority=SuggestionPriority.MEDIUM,
                mode=DegradationMode.CYCLING
            ))
        else:
            # Trip requires >95% charge - warn about range anxiety vs health trade-off
            suggestions.append(Suggestion(
                title="Trip at maximum range - plan charging stops",
                description=f"Trip requires {required_range:.0f}km range (near maximum capacity). "
                           f"Consider planning charging stops to avoid deep discharge and reduce stress.",
                impact="Prevents deep discharge and reduces degradation",
                priority=SuggestionPriority.HIGH,
                mode=DegradationMode.CYCLING
            ))
        
        return suggestions
    
    def _general_suggestions(self, context: UsageContext) -> List[Suggestion]:
        """Generate suggestions that apply to both modes."""
        suggestions = []
        
        # Temperature-based suggestions
        if context.temperature < self.LOW_TEMP_THRESHOLD:
            suggestions.append(Suggestion(
                title="Warm battery before charging",
                description=f"Charging at {context.temperature:.0f}°C can cause lithium plating. "
                           f"Warm battery to >10°C before charging.",
                impact="Prevents irreversible capacity loss",
                priority=SuggestionPriority.HIGH,
                mode=None
            ))
        
        # Low SOH warning
        if context.soh < 0.80:
            suggestions.append(Suggestion(
                title="Consider battery replacement planning",
                description=f"Battery at {context.soh:.0%} SOH. Start planning for replacement "
                           f"to avoid unexpected failure.",
                impact="Ensures reliability",
                priority=SuggestionPriority.MEDIUM,
                mode=None
            ))
        
        return suggestions
    
    def get_top_suggestions(
        self, 
        context: UsageContext, 
        n: int = 3
    ) -> List[Suggestion]:
        """Get top N suggestions by priority."""
        all_suggestions = self.generate(context)
        return all_suggestions[:n]


def format_suggestions_for_display(suggestions: List[Suggestion]) -> str:
    """Format suggestions for user display."""
    if not suggestions:
        return "No specific recommendations at this time."
    
    lines = ["💡 Recommendations to Maximize Battery Life:\n"]
    
    priority_emoji = {
        SuggestionPriority.HIGH: "🔴",
        SuggestionPriority.MEDIUM: "🟡",
        SuggestionPriority.LOW: "🟢"
    }
    
    for i, s in enumerate(suggestions, 1):
        emoji = priority_emoji[s.priority]
        lines.append(f"{i}. {emoji} {s.title}")
        lines.append(f"   {s.description}")
        lines.append(f"   Impact: {s.impact}\n")
    
    return "\n".join(lines)


if __name__ == '__main__':
    # Test the suggestion generator
    generator = SuggestionGenerator()
    
    # Test cycling context
    cycling_context = UsageContext(
        mode=DegradationMode.CYCLING,
        soh=0.85,
        temperature=38,
        avg_soc=0.50,
        charge_rate=1.5,
        discharge_rate=0.8,
        deep_discharge_freq=0.3
    )
    
    print("=" * 60)
    print("CYCLING MODE SUGGESTIONS")
    print("=" * 60)
    suggestions = generator.generate(cycling_context)
    print(format_suggestions_for_display(suggestions))
    
    # Test storage context
    storage_context = UsageContext(
        mode=DegradationMode.STORAGE,
        soh=0.90,
        temperature=28,
        avg_soc=0.95,
        charge_rate=0,
        discharge_rate=0,
        deep_discharge_freq=0
    )
    
    print("\n" + "=" * 60)
    print("STORAGE MODE SUGGESTIONS")
    print("=" * 60)
    suggestions = generator.generate(storage_context)
    print(format_suggestions_for_display(suggestions))
