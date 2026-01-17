"""Battery Advisory System - Package Init."""

from .warning_engine import (
    WarningEngine,
    WarningLevel,
    WarningResult,
    get_warning_color,
    get_warning_emoji
)
from .suggestion_generator import (
    SuggestionGenerator,
    Suggestion,
    UsageContext,
    DegradationMode
)
from .battery_advisor import BatteryAdvisor, BatteryHealthReport
from .enhanced_advisor import EnhancedBatteryAdvisor, EnhancedHealthReport

__all__ = [
    # Warning Engine
    'WarningEngine',
    'WarningLevel', 
    'WarningResult',
    'get_warning_color',
    'get_warning_emoji',
    # Suggestions
    'SuggestionGenerator',
    'Suggestion',
    'UsageContext',
    'DegradationMode',
    # Advisors
    'BatteryAdvisor',
    'BatteryHealthReport',
    'EnhancedBatteryAdvisor',
    'EnhancedHealthReport'
]

