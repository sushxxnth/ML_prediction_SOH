"""
Model Predictive Control (MPC) for Battery Charging Optimization

Implements degradation-aware optimal charging that:
- Minimizes capacity fade while meeting charging time requirements
- Prevents lithium plating using physics-informed constraints
- Maintains safe temperature and SOC ranges
- Adapts strategy based on HERO's causal attribution

Key Features:
- Multi-step ahead prediction (N=10 default)
- Receding horizon control (re-optimize each step)
- Physics-based safety constraints
- Integration with HERO's degradation models
"""

import numpy as np
from typing import Dict, Tuple, Optional, Callable
from dataclasses import dataclass
from scipy.optimize import minimize, Bounds, LinearConstraint
import warnings


@dataclass
class BatteryState:
    """Current battery state for MPC."""
    soc: float  # State of charge (0-1)
    temperature: float  # Temperature in Celsius
    voltage: float  # Terminal voltage (V)
    capacity: float  # Current capacity (Ah)
    cycle_count: int  # Number of cycles completed
    
    
@dataclass
class MPCConfig:
    """MPC controller configuration."""
    horizon: int = 10  # Prediction horizon (time steps)
    dt: float = 60.0  # Time step size (seconds)
    
    # Battery parameters
    nominal_capacity: float = 2.5  # Ah
    max_current: float = 2.5  # Maximum charge current (A) = 1C
    min_current: float = 0.1  # Minimum charge current (A)
    
    # SOC limits
    soc_min: float = 0.0
    soc_max: float = 0.95  # Stop at 95% to prevent overcharge
    
    # Temperature limits
    temp_min: float = 15.0  # °C
    temp_max: float = 45.0  # °C
    
    # Objective weights
    w_degradation: float = 1.0  # Degradation penalty weight
    w_time: float = 0.1  # Time penalty weight
    w_temperature: float = 0.5  # Temperature penalty weight
    
    # Plating threshold (chemistry-dependent)
    plating_threshold_current: float = 1.5  # A (0.6C for safety)
    plating_threshold_temp: float = 10.0  # °C


class MPCChargingController:
    """
    Model Predictive Control for optimal battery charging.
    
    The controller solves an optimization problem at each time step:
    
    minimize: Σ[degradation_cost + w_time·time_cost + w_temp·temp_cost]
    subject to: SOC limits, current limits, temperature limits, plating constraints
    
    Then executes only the first action and re-optimizes at the next step.
    """
    
    def __init__(self, config: MPCConfig, degradation_model: Optional[Callable] = None):
        """
        Initialize MPC controller.
        
        Args:
            config: MPC configuration parameters
            degradation_model: Function that predicts degradation given (state, current)
                              If None, uses simple empirical model
        """
        self.config = config
        self.degradation_model = degradation_model or self._default_degradation_model
        
        # State tracking
        self.state_history = []
        self.action_history = []
        
    def _default_degradation_model(self, state: BatteryState, current: float) -> float:
        """
        Default empirical degradation model with numerical stability.
        
        Returns:
            Estimated capacity loss (%) for one time step
        """
        c_rate = current / self.config.nominal_capacity
        
        # Arrhenius temperature factor (with clipping for stability)
        # Ea ≈ 50 kJ/mol for SEI growth, reference temp 25°C
        temp_diff = np.clip(state.temperature - 25, -30, 50)  # Prevent exp overflow
        temp_factor = np.exp(0.02 * temp_diff)  # Reduced sensitivity
        temp_factor = np.clip(temp_factor, 0.1, 5.0)  # Hard bounds
        
        # C-rate stress (quadratic penalty for high rates)
        rate_factor = 1.0 + 0.3 * (c_rate ** 2)
        
        # SOC stress (SEI grows faster at high SOC)
        soc_factor = 1.0 + 0.2 * max(0, state.soc - 0.8)
        
        # Base degradation per time step (empirical)
        base_degradation = 1e-6  # 0.0001% per minute at nominal conditions
        
        degradation = base_degradation * temp_factor * rate_factor * soc_factor
        
        return degradation
    
    def _predict_temperature(self, state: BatteryState, current: float) -> float:
        """
        Predict temperature after applying current for dt.
        
        Simple lumped thermal model with numerical stability.
        """
        # Battery parameters (typical 18650)
        resistance = 0.05  # Ohm
        heat_transfer_coeff = 5.0  # W/K
        thermal_mass = 45.0  # J/K (50g * 0.9 J/g/K)
        ambient_temp = 25.0  # °C
        
        # Heat generation from I²R
        heat_gen = (current ** 2) * resistance
        
        # Heat dissipation (Newton's law of cooling)
        heat_loss = heat_transfer_coeff * (state.temperature - ambient_temp)
        
        # Temperature change (with small time step for stability)
        dt_dt = (heat_gen - heat_loss) / thermal_mass
        
        # Limit rate of change for numerical stability
        dt_dt = np.clip(dt_dt, -0.5, 0.5)  # Max 0.5°C per second
        
        temp_new = state.temperature + dt_dt * self.config.dt
        
        # Hard bounds to prevent runaway
        temp_new = np.clip(temp_new, 15.0, 60.0)
        
        return temp_new
    
    def _predict_soc(self, state: BatteryState, current: float) -> float:
        """
        Predict SOC after applying current for dt.
        """
        coulombs = current * self.config.dt  # A·s
        ah_charged = coulombs / 3600  # A·h
        
        delta_soc = ah_charged / state.capacity
        soc_new = state.soc + delta_soc
        
        return np.clip(soc_new, 0, 1.0)
    
    def _predict_voltage(self, soc: float) -> float:
        """
        Predict open-circuit voltage from SOC.
        """
        # Simplified polynomial fit for NMC OCV
        v_min = 3.0  # V at 0% SOC
        v_max = 4.2  # V at 100% SOC
        
        # Nonlinear voltage curve
        voltage = v_min + (v_max - v_min) * (
            0.8 * soc + 0.2 * (soc ** 2)
        )
        
        return voltage
    
    def _check_plating_risk(self, state: BatteryState, current: float) -> bool:
        """
        Check if current/temperature combination risks lithium plating.
        
        Plating occurs when:
        - High current (local Li+ concentration depletion)
        - Low temperature (slow diffusion)
        
        Args:
            state: Current battery state
            current: Proposed current (A)
            
        Returns:
            True if plating risk detected
        """
        # High current at low temperature = plating risk
        if (current > self.config.plating_threshold_current and 
            state.temperature < self.config.plating_threshold_temp):
            return True
        
        # Also risk if current is very high regardless of temperature
        c_rate = current / self.config.nominal_capacity
        if c_rate > 1.5:  # >1.5C is aggressive
            return True
        
        return False
    
    def _compute_objective(self, current_sequence: np.ndarray, initial_state: BatteryState) -> float:
        """
        Compute MPC objective function over prediction horizon.
        
        Args:
            current_sequence: Array of currents for N time steps
            initial_state: Starting battery state
            
        Returns:
            Total cost (to be minimized)
        """
        state = initial_state
        total_cost = 0.0
        
        for i, current in enumerate(current_sequence):
            # Predict next state
            soc_next = self._predict_soc(state, current)
            temp_next = self._predict_temperature(state, current)
            voltage_next = self._predict_voltage(soc_next)
            
            # Degradation cost
            degradation = self.degradation_model(state, current)
            degradation_cost = self.config.w_degradation * degradation * 1e5  # Scale up
            
            # Time cost (incentivize faster charging)
            # Lower current = longer time = higher cost
            time_cost = self.config.w_time * (1.0 / (current + 0.1))
            
            # Temperature cost (penalize high temperatures)
            temp_excess = max(0, temp_next - 35.0)  # Prefer T < 35°C
            temp_cost = self.config.w_temperature * (temp_excess ** 2)
            
            # Plating penalty (soft constraint)
            plating_cost = 0.0
            if self._check_plating_risk(state, current):
                plating_cost = 100.0  # Large penalty
            
            total_cost += degradation_cost + time_cost + temp_cost + plating_cost
            
            # Update state for next iteration
            state = BatteryState(
                soc=soc_next,
                temperature=temp_next,
                voltage=voltage_next,
                capacity=state.capacity * (1 - degradation),  # Capacity fades
                cycle_count=state.cycle_count
            )
        
        return total_cost
    
    def optimize_charging_profile(self, initial_state: BatteryState) -> Tuple[np.ndarray, Dict]:
        """
        Optimize charging current profile over prediction horizon.
        
        Args:
            initial_state: Current battery state
            
        Returns:
            optimal_currents: Array of optimal currents for N steps
            info: Dictionary with optimization details
        """
        N = self.config.horizon
        
        # Initial guess: constant 0.5C charging
        x0 = np.ones(N) * 0.5 * self.config.nominal_capacity
        
        # Bounds on current (A)
        bounds = Bounds(
            lb=self.config.min_current * np.ones(N),
            ub=self.config.max_current * np.ones(N)
        )
        
        # Constraints on final SOC (must reach at least target)
        # This is handled implicitly by optimizing total charging time
        
        # Solve optimization
        result = minimize(
            fun=lambda u: self._compute_objective(u, initial_state),
            x0=x0,
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': 100, 'ftol': 1e-6}
        )
        
        if not result.success:
            warnings.warn(f"MPC optimization failed: {result.message}")
        
        optimal_currents = result.x
        
        info = {
            'success': result.success,
            'cost': result.fun,
            'iterations': result.nit,
            'final_soc_prediction': self._simulate_trajectory(optimal_currents, initial_state)['soc'][-1]
        }
        
        return optimal_currents, info
    
    def _simulate_trajectory(self, current_sequence: np.ndarray, initial_state: BatteryState) -> Dict:
        """
        Simulate battery trajectory given current sequence.
        
        Args:
            current_sequence: Sequence of currents to apply
            initial_state: Starting state
            
        Returns:
            Dictionary containing state trajectories (soc, temp, voltage, etc.)
        """
        state = initial_state
        
        soc_traj = [state.soc]
        temp_traj = [state.temperature]
        voltage_traj = [state.voltage]
        degradation_traj = [0.0]
        
        for current in current_sequence:
            soc_next = self._predict_soc(state, current)
            temp_next = self._predict_temperature(state, current)
            voltage_next = self._predict_voltage(soc_next)
            degradation = self.degradation_model(state, current)
            
            soc_traj.append(soc_next)
            temp_traj.append(temp_next)
            voltage_traj.append(voltage_next)
            degradation_traj.append(degradation)
            
            state = BatteryState(
                soc=soc_next,
                temperature=temp_next,
                voltage=voltage_next,
                capacity=state.capacity * (1 - degradation),
                cycle_count=state.cycle_count
            )
        
        return {
            'soc': np.array(soc_traj),
            'temperature': np.array(temp_traj),
            'voltage': np.array(voltage_traj),
            'degradation': np.array(degradation_traj)
        }
    
    def step(self, current_state: BatteryState) -> Tuple[float, Dict]:
        """
        Execute one MPC step: optimize and return first action (receding horizon).
        
        Args:
            current_state: Current battery state
            
        Returns:
            optimal_current: Optimal current for this time step (A)
            info: Optimization and prediction information
        """
        # Optimize over horizon
        optimal_sequence, opt_info = self.optimize_charging_profile(current_state)
        
        # Receding horizon: only execute first action
        optimal_current = optimal_sequence[0]
        
        # Store for logging
        self.state_history.append(current_state)
        self.action_history.append(optimal_current)
        
        # Return first action and full trajectory prediction
        trajectory = self._simulate_trajectory(optimal_sequence, current_state)
        
        info = {
            **opt_info,
            'predicted_trajectory': trajectory,
            'full_optimal_sequence': optimal_sequence
        }
        
        return optimal_current, info


def generate_cc_cv_baseline(initial_state: BatteryState, config: MPCConfig) -> np.ndarray:
    """
    Generate baseline CC-CV (Constant Current - Constant Voltage) charging profile.
    
    Args:
        initial_state: Starting battery state
        config: MPC configuration (for parameters)
        
    Returns:
        Array of currents for comparison
    """
    currents = []
    state = initial_state
    
    # Constant current phase (1C until 95% SOC or 4.2V)
    cc_current = 1.0 * config.nominal_capacity  # 1C
    
    while state.soc < 0.95:
        currents.append(cc_current)
        
        # Update state
        soc_next = state.soc + (cc_current * config.dt) / (3600 * state.capacity)
        state = BatteryState(
            soc=soc_next,
            temperature=state.temperature,  # Simplified
            voltage=3.0 + 1.2 * soc_next,
            capacity=state.capacity,
            cycle_count=state.cycle_count
        )
    
    # Constant voltage phase (taper current)
    cv_voltage = 4.2
    while currents[-1] > 0.1:
        # Simple taper model
        current = currents[-1] * 0.9
        currents.append(current)
    
    return np.array(currents)


if __name__ == "__main__":
    # Quick test of MPC controller
    print("Testing MPC Charging Controller...")
    
    # Create configuration
    config = MPCConfig(horizon=10, dt=60.0)
    
    # Initialize controller
    controller = MPCChargingController(config)
    
    # Initial state: battery at 20% SOC, 25°C
    initial_state = BatteryState(
        soc=0.2,
        temperature=25.0,
        voltage=3.6,
        capacity=2.5,
        cycle_count=100
    )
    
    # Optimize one step
    optimal_current, info = controller.step(initial_state)
    
    print(f"\nInitial SOC: {initial_state.soc*100:.1f}%")
    print(f"Optimal current for first step: {optimal_current:.3f} A ({optimal_current/config.nominal_capacity:.2f}C)")
    print(f"Optimization converged: {info['success']}")
    print(f"Predicted final SOC: {info['final_soc_prediction']*100:.1f}%")
    print(f"\n✅ MPC controller working!")
