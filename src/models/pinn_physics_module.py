"""
Physics-Informed Neural Network (PINN) Module for Causal Attribution

This module implements physics-based residual losses that enforce electrochemical
governing equations during training. The key innovation is embedding degradation
PDEs directly into the loss function.

Key Equations:
1. SEI Growth: Q_SEI = k * sqrt(t) * exp(-E_a/2RT) * f(SOC)
2. Lithium Plating: Butler-Volmer kinetics at low temperatures
3. Active Material Loss: Q_AM = k * C_rate^β * N^γ

Author: Battery ML Research
Week 2: PINN Integration for Causal Attribution
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass


# Physical Constants

@dataclass
class PhysicsConstants:
    """Electrochemical constants for battery degradation."""
    R: float = 8.314  # Gas constant (J/mol·K)
    F: float = 96485  # Faraday constant (C/mol)
    
    # Literature values for SEI growth
    E_a_SEI_min: float = 35000  # Minimum activation energy (J/mol) ~35 kJ/mol
    E_a_SEI_max: float = 60000  # Maximum activation energy (J/mol) ~60 kJ/mol
    E_a_SEI_typical: float = 50000  # Typical value ~50 kJ/mol
    
    # Temperature thresholds
    T_plating_threshold: float = 278.15  # 5°C - plating becomes significant below this
    T_reference: float = 298.15  # 25°C reference temperature
    
    # C-rate exponents (from literature)
    beta_AM_typical: float = 1.5  # C-rate exponent for AM loss
    gamma_AM_typical: float = 0.5  # Cycle number exponent

PHYSICS = PhysicsConstants()


# Physics Parameter Network

class PhysicsParameterNetwork(nn.Module):
    """
    Neural network that learns physically meaningful parameters.
    
    Instead of learning arbitrary weights, this network outputs parameters
    that have physical interpretation (activation energies, rate constants).
    Parameters are bounded to physically plausible ranges.
    """
    
    def __init__(
        self,
        input_dim: int = 15,  # Features + context
        hidden_dim: int = 64
    ):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
        )
        
        # Separate heads for each physics parameter
        # Using sigmoid/softplus to enforce physical bounds
        
        # SEI parameters
        self.sei_E_a_head = nn.Linear(hidden_dim // 2, 1)  # Activation energy
        self.sei_k_head = nn.Linear(hidden_dim // 2, 1)    # Rate constant
        
        # Plating parameters
        self.plating_k_head = nn.Linear(hidden_dim // 2, 1)  # Rate constant
        self.plating_alpha_head = nn.Linear(hidden_dim // 2, 1)  # Transfer coeff
        
        # Active material loss parameters
        self.am_k_head = nn.Linear(hidden_dim // 2, 1)     # Rate constant
        self.am_beta_head = nn.Linear(hidden_dim // 2, 1)  # C-rate exponent
        self.am_gamma_head = nn.Linear(hidden_dim // 2, 1) # Cycle exponent
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass returning bounded physics parameters.
        
        Args:
            x: Input features (batch, input_dim)
            
        Returns:
            Dictionary of physics parameters with physical bounds enforced
        """
        h = self.encoder(x)
        
        # SEI activation energy: bounded to [35, 60] kJ/mol
        E_a_raw = self.sei_E_a_head(h)
        E_a_SEI = PHYSICS.E_a_SEI_min + (PHYSICS.E_a_SEI_max - PHYSICS.E_a_SEI_min) * torch.sigmoid(E_a_raw)
        
        # Rate constants: positive (softplus)
        k_SEI = F.softplus(self.sei_k_head(h)) * 0.01  # Scale to reasonable range
        k_plating = F.softplus(self.plating_k_head(h)) * 0.01
        k_AM = F.softplus(self.am_k_head(h)) * 0.01
        
        # Transfer coefficient: bounded to [0.3, 0.7]
        alpha_plating = 0.3 + 0.4 * torch.sigmoid(self.plating_alpha_head(h))
        
        # Exponents: bounded to physically meaningful ranges
        # β (C-rate exponent): [1.0, 2.0]
        beta_AM = 1.0 + 1.0 * torch.sigmoid(self.am_beta_head(h))
        
        # γ (cycle exponent): [0.3, 1.0]
        gamma_AM = 0.3 + 0.7 * torch.sigmoid(self.am_gamma_head(h))
        
        return {
            'E_a_SEI': E_a_SEI,      # J/mol
            'k_SEI': k_SEI,          # dimensionless rate
            'k_plating': k_plating,
            'alpha_plating': alpha_plating,
            'k_AM': k_AM,
            'beta_AM': beta_AM,      # C-rate exponent
            'gamma_AM': gamma_AM,    # Cycle exponent
        }


# SEI Growth Physics

class SEIGrowthEquation(nn.Module):
    """
    Implements SEI layer growth physics following diffusion-limited kinetics.
    
    Governing Equation:
        δ_SEI(t) = sqrt(2 * D_s * t) * exp(-E_a / 2RT)
        
    Capacity Loss:
        Q_SEI = k_SEI * sqrt(t) * exp(-E_a / 2RT) * f(SOC)
        
    Where f(SOC) captures higher reactivity at extreme SOC:
        f(SOC) = 1 + α * (SOC - 0.5)²
    """
    
    def __init__(self, soc_sensitivity: float = 0.5):
        super().__init__()
        self.soc_sensitivity = soc_sensitivity
        
    def forward(
        self,
        time: torch.Tensor,          # Time in months or normalized
        temperature: torch.Tensor,   # Temperature in Kelvin
        soc: torch.Tensor,           # State of charge [0, 1]
        E_a: torch.Tensor,           # Activation energy (J/mol)
        k: torch.Tensor,             # Rate constant
    ) -> torch.Tensor:
        """
        Compute physics-based SEI capacity loss.
        
        Args:
            time: Storage/cycling time 
            temperature: Battery temperature (K)
            soc: State of charge
            E_a: Activation energy from PhysicsParameterNetwork
            k: Rate constant from PhysicsParameterNetwork
            
        Returns:
            Predicted SEI-induced capacity loss
        """
        # Arrhenius factor: exp(-E_a / 2RT)
        # Factor of 2 in denominator from sqrt of exponential
        arrhenius = torch.exp(-E_a / (2 * PHYSICS.R * temperature))
        
        # Sqrt(t) scaling from diffusion-limited growth
        sqrt_t = torch.sqrt(time.clamp(min=1e-6))
        
        # SOC dependence: higher reactivity at extreme SOC
        # f(SOC) = 1 + α * (SOC - 0.5)²
        soc_factor = 1.0 + self.soc_sensitivity * (soc - 0.5) ** 2
        
        # Combined SEI capacity loss
        Q_SEI = k * sqrt_t * arrhenius * soc_factor
        
        return Q_SEI
    
    def compute_residual(
        self,
        predicted_Q_SEI: torch.Tensor,
        time: torch.Tensor,
        temperature: torch.Tensor,
        soc: torch.Tensor,
        E_a: torch.Tensor,
        k: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute PDE residual for SEI growth equation.
        
        The residual measures how well the predicted Q_SEI satisfies
        the governing equation.
        """
        physics_Q_SEI = self.forward(time, temperature, soc, E_a, k)
        residual = predicted_Q_SEI - physics_Q_SEI
        return residual


# Lithium Plating Physics

class LithiumPlatingEquation(nn.Module):
    """
    Implements lithium plating physics following Butler-Volmer kinetics.
    
    Plating occurs when:
        1. Temperature is low (< 5°C / 278 K)
        2. Charging rate is high
        3. Anode potential drops below Li/Li+ reference
        
    Simplified capacity loss model:
        Q_plating = k * I[T < T_crit] * f_T(T) * C_rate^α
        
    Where:
        - I[T < T_crit] is indicator for cold conditions
        - f_T(T) = exp((T_ref - T) / T_scale) for enhanced plating at low T
    """
    
    def __init__(
        self,
        T_critical: float = 278.15,  # 5°C
        T_scale: float = 10.0,       # Temperature scaling
    ):
        super().__init__()
        self.T_critical = T_critical
        self.T_scale = T_scale
        
    def forward(
        self,
        temperature: torch.Tensor,   # Temperature in Kelvin
        c_rate: torch.Tensor,        # Charging C-rate
        k: torch.Tensor,             # Rate constant
        alpha: torch.Tensor,         # Transfer coefficient
    ) -> torch.Tensor:
        """
        Compute physics-based lithium plating capacity loss.
        
        Args:
            temperature: Battery temperature (K)
            c_rate: Charging rate (C)
            k: Rate constant
            alpha: Transfer coefficient
            
        Returns:
            Predicted plating-induced capacity loss
        """
        # Temperature factor: enhanced plating at low T
        # Smooth transition using sigmoid instead of hard threshold
        cold_factor = torch.sigmoid((self.T_critical - temperature) / 5.0)
        
        # Exponential enhancement at colder temperatures
        temp_enhancement = torch.exp((PHYSICS.T_reference - temperature) / self.T_scale)
        temp_enhancement = temp_enhancement.clamp(max=10.0)  # Prevent explosion
        
        # C-rate dependence: higher charging rates increase plating
        # Using learned alpha exponent
        c_rate_factor = c_rate.clamp(min=0.1) ** alpha
        
        # Combined plating capacity loss
        Q_plating = k * cold_factor * temp_enhancement * c_rate_factor
        
        return Q_plating
    
    def compute_residual(
        self,
        predicted_Q_plating: torch.Tensor,
        temperature: torch.Tensor,
        c_rate: torch.Tensor,
        k: torch.Tensor,
        alpha: torch.Tensor,
    ) -> torch.Tensor:
        """Compute PDE residual for plating equation."""
        physics_Q_plating = self.forward(temperature, c_rate, k, alpha)
        residual = predicted_Q_plating - physics_Q_plating
        return residual


# Active Material Loss Physics

class ActiveMaterialLossEquation(nn.Module):
    """
    Implements active material loss from mechanical stress during cycling.
    
    Particle fracture occurs due to:
        1. Volume changes during lithiation/delithiation
        2. Mechanical stress from high C-rates
        3. Fatigue accumulation over cycles
        
    Capacity loss model:
        Q_AM = k * C_rate^β * N^γ
        
    Where:
        - β ≈ 1.5 (C-rate exponent from literature)
        - γ ≈ 0.5-1.0 (cycle exponent, sub-linear fatigue)
    """
    
    def __init__(self):
        super().__init__()
        
    def forward(
        self,
        c_rate: torch.Tensor,        # Discharge C-rate
        n_cycles: torch.Tensor,      # Cycle count
        k: torch.Tensor,             # Rate constant
        beta: torch.Tensor,          # C-rate exponent
        gamma: torch.Tensor,         # Cycle exponent
    ) -> torch.Tensor:
        """
        Compute physics-based active material loss.
        
        Args:
            c_rate: Cycling C-rate
            n_cycles: Number of cycles
            k: Rate constant
            beta: C-rate exponent (learned, typically ~1.5)
            gamma: Cycle exponent (learned, typically ~0.5)
            
        Returns:
            Predicted AM-loss-induced capacity loss
        """
        # C-rate factor: higher rates cause more mechanical stress
        c_rate_factor = c_rate.clamp(min=0.1) ** beta
        
        # Cycle factor: fatigue accumulation (sub-linear often)
        cycle_factor = n_cycles.clamp(min=1.0) ** gamma
        
        # Combined AM loss
        Q_AM = k * c_rate_factor * cycle_factor
        
        return Q_AM
    
    def compute_residual(
        self,
        predicted_Q_AM: torch.Tensor,
        c_rate: torch.Tensor,
        n_cycles: torch.Tensor,
        k: torch.Tensor,
        beta: torch.Tensor,
        gamma: torch.Tensor,
    ) -> torch.Tensor:
        """Compute PDE residual for AM loss equation."""
        physics_Q_AM = self.forward(c_rate, n_cycles, k, beta, gamma)
        residual = predicted_Q_AM - physics_Q_AM
        return residual


# Combined Physics Residual Loss

class PhysicsResidualLoss(nn.Module):
    """
    Combined physics-informed loss for all degradation mechanisms.
    
    Total physics loss:
        L_physics = λ_SEI * ||R_SEI||² + λ_plating * ||R_plating||² + λ_AM * ||R_AM||²
        
    Where R_* are the residuals from each governing equation.
    """
    
    def __init__(
        self,
        lambda_SEI: float = 1.0,
        lambda_plating: float = 1.0,
        lambda_AM: float = 1.0,
        reduction: str = 'mean'
    ):
        super().__init__()
        
        self.lambda_SEI = lambda_SEI
        self.lambda_plating = lambda_plating
        self.lambda_AM = lambda_AM
        self.reduction = reduction
        
        # Physics equation modules
        self.sei_equation = SEIGrowthEquation()
        self.plating_equation = LithiumPlatingEquation()
        self.am_equation = ActiveMaterialLossEquation()
        
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],  # Predicted mechanism contributions
        physics_params: Dict[str, torch.Tensor],  # Learned physics parameters
        context: Dict[str, torch.Tensor],  # Operating context
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute combined physics residual loss.
        
        Args:
            predictions: Dict with 'sei', 'plating', 'am_loss' predictions
            physics_params: Dict from PhysicsParameterNetwork
            context: Dict with 'temperature', 'c_rate', 'soc', 'time', 'n_cycles'
            
        Returns:
            total_loss: Combined physics loss
            losses_dict: Individual losses for logging
        """
        losses = {}
        
        # SEI residual loss
        if 'sei' in predictions and 'temperature' in context:
            R_SEI = self.sei_equation.compute_residual(
                predicted_Q_SEI=predictions['sei'],
                time=context.get('time', torch.ones_like(predictions['sei'])),
                temperature=context['temperature'],
                soc=context.get('soc', torch.ones_like(predictions['sei']) * 0.5),
                E_a=physics_params['E_a_SEI'],
                k=physics_params['k_SEI'],
            )
            losses['L_SEI'] = self._reduce(R_SEI ** 2)
        else:
            losses['L_SEI'] = torch.tensor(0.0)
            
        # Plating residual loss
        if 'plating' in predictions and 'temperature' in context:
            R_plating = self.plating_equation.compute_residual(
                predicted_Q_plating=predictions['plating'],
                temperature=context['temperature'],
                c_rate=context.get('c_rate', torch.ones_like(predictions['plating'])),
                k=physics_params['k_plating'],
                alpha=physics_params['alpha_plating'],
            )
            losses['L_plating'] = self._reduce(R_plating ** 2)
        else:
            losses['L_plating'] = torch.tensor(0.0)
            
        # AM loss residual
        if 'am_loss' in predictions and 'c_rate' in context:
            R_AM = self.am_equation.compute_residual(
                predicted_Q_AM=predictions['am_loss'],
                c_rate=context.get('c_rate', torch.ones_like(predictions['am_loss'])),
                n_cycles=context.get('n_cycles', torch.ones_like(predictions['am_loss']) * 100),
                k=physics_params['k_AM'],
                beta=physics_params['beta_AM'],
                gamma=physics_params['gamma_AM'],
            )
            losses['L_AM'] = self._reduce(R_AM ** 2)
        else:
            losses['L_AM'] = torch.tensor(0.0)
            
        # Combined loss
        total_loss = (
            self.lambda_SEI * losses['L_SEI'] +
            self.lambda_plating * losses['L_plating'] +
            self.lambda_AM * losses['L_AM']
        )
        
        return total_loss, losses
    
    def _reduce(self, x: torch.Tensor) -> torch.Tensor:
        """Apply reduction to tensor."""
        if self.reduction == 'mean':
            return x.mean()
        elif self.reduction == 'sum':
            return x.sum()
        else:
            return x


# Parameter Regularization

class PhysicsParameterRegularization(nn.Module):
    """
    Regularization to keep learned physics parameters within expected ranges.
    
    This provides a soft constraint that pulls parameters toward literature values
    while allowing the network to learn data-driven corrections.
    """
    
    def __init__(
        self,
        E_a_target: float = 50000,  # 50 kJ/mol
        beta_target: float = 1.5,
        gamma_target: float = 0.5,
        lambda_reg: float = 0.1
    ):
        super().__init__()
        self.E_a_target = E_a_target
        self.beta_target = beta_target
        self.gamma_target = gamma_target
        self.lambda_reg = lambda_reg
        
    def forward(
        self,
        physics_params: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute regularization loss pulling params toward literature values.
        """
        loss = 0.0
        
        # E_a regularization (normalized by typical magnitude)
        E_a_diff = (physics_params['E_a_SEI'].mean() - self.E_a_target) / self.E_a_target
        loss += E_a_diff ** 2
        
        # Beta regularization
        beta_diff = physics_params['beta_AM'].mean() - self.beta_target
        loss += beta_diff ** 2
        
        # Gamma regularization
        gamma_diff = physics_params['gamma_AM'].mean() - self.gamma_target
        loss += gamma_diff ** 2
        
        return self.lambda_reg * loss


# Module Tests

if __name__ == '__main__':
    print("="*70)
    print("PINN Physics Module - Unit Tests")
    print("="*70)
    
    # Test PhysicsParameterNetwork
    print("\n[1] Testing PhysicsParameterNetwork...")
    batch_size = 8
    input_dim = 15
    
    param_net = PhysicsParameterNetwork(input_dim=input_dim)
    x = torch.randn(batch_size, input_dim)
    params = param_net(x)
    
    print(f"  Input shape: {x.shape}")
    print(f"  Output parameters:")
    for name, val in params.items():
        print(f"    {name}: shape={val.shape}, range=[{val.min():.4f}, {val.max():.4f}]")
    
    # Verify bounds
    assert (params['E_a_SEI'] >= PHYSICS.E_a_SEI_min).all(), "E_a below minimum"
    assert (params['E_a_SEI'] <= PHYSICS.E_a_SEI_max).all(), "E_a above maximum"
    assert (params['beta_AM'] >= 1.0).all(), "beta below minimum"
    assert (params['beta_AM'] <= 2.0).all(), "beta above maximum"
    print("   All parameter bounds verified")
    
    # Test SEI equation
    print("\n[2] Testing SEI Growth Equation...")
    sei_eq = SEIGrowthEquation()
    
    time = torch.linspace(1, 12, batch_size).unsqueeze(1)  # 1-12 months
    temp = torch.ones(batch_size, 1) * 298.15  # 25°C
    soc = torch.ones(batch_size, 1) * 0.8  # 80% SOC
    
    Q_SEI = sei_eq(
        time=time,
        temperature=temp,
        soc=soc,
        E_a=params['E_a_SEI'],
        k=params['k_SEI']
    )
    
    print(f"  Time range: {time.min():.1f} - {time.max():.1f} months")
    print(f"  Temperature: {temp[0].item():.1f} K ({temp[0].item() - 273.15:.1f}°C)")
    print(f"  SOC: {soc[0].item():.1%}")
    print(f"  Q_SEI range: [{Q_SEI.min():.6f}, {Q_SEI.max():.6f}]")
    print("   SEI equation working")
    
    # Test plating equation
    print("\n[3] Testing Lithium Plating Equation...")
    plating_eq = LithiumPlatingEquation()
    
    temp_cold = torch.ones(batch_size, 1) * 268.15  # -5°C
    c_rate = torch.ones(batch_size, 1) * 2.0  # 2C charging
    
    Q_plating = plating_eq(
        temperature=temp_cold,
        c_rate=c_rate,
        k=params['k_plating'],
        alpha=params['alpha_plating']
    )
    
    print(f"  Temperature: {temp_cold[0].item():.1f} K ({temp_cold[0].item() - 273.15:.1f}°C)")
    print(f"  C-rate: {c_rate[0].item():.1f}C")
    print(f"  Q_plating range: [{Q_plating.min():.6f}, {Q_plating.max():.6f}]")
    print("   Plating equation working")
    
    # Test AM loss equation
    print("\n[4] Testing Active Material Loss Equation...")
    am_eq = ActiveMaterialLossEquation()
    
    c_rate_cycling = torch.linspace(0.5, 3.0, batch_size).unsqueeze(1)
    n_cycles = torch.ones(batch_size, 1) * 500
    
    Q_AM = am_eq(
        c_rate=c_rate_cycling,
        n_cycles=n_cycles,
        k=params['k_AM'],
        beta=params['beta_AM'],
        gamma=params['gamma_AM']
    )
    
    print(f"  C-rate range: {c_rate_cycling.min():.1f} - {c_rate_cycling.max():.1f}C")
    print(f"  Cycles: {n_cycles[0].item():.0f}")
    print(f"  Q_AM range: [{Q_AM.min():.6f}, {Q_AM.max():.6f}]")
    print("   AM loss equation working")
    
    # Test combined physics loss
    print("\n[5] Testing Combined Physics Residual Loss...")
    physics_loss = PhysicsResidualLoss()
    
    predictions = {
        'sei': Q_SEI + torch.randn_like(Q_SEI) * 0.001,  # Add noise
        'plating': Q_plating + torch.randn_like(Q_plating) * 0.001,
        'am_loss': Q_AM + torch.randn_like(Q_AM) * 0.001,
    }
    
    context = {
        'temperature': temp,
        'c_rate': c_rate,
        'soc': soc,
        'time': time,
        'n_cycles': n_cycles,
    }
    
    total_loss, losses_dict = physics_loss(predictions, params, context)
    
    print(f"  Total physics loss: {total_loss.item():.6f}")
    for name, val in losses_dict.items():
        print(f"    {name}: {val.item():.6f}")
    print("   Combined loss working")
    
    # Test parameter regularization
    print("\n[6] Testing Parameter Regularization...")
    reg_loss = PhysicsParameterRegularization()
    L_reg = reg_loss(params)
    print(f"  Regularization loss: {L_reg.item():.6f}")
    print("   Regularization working")
    
    print("\n" + "="*70)
    print(" All PINN physics module tests passed!")
    print("="*70)
