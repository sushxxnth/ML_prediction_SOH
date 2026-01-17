"""
Advanced Feature Engineering for Battery SOH Prediction
Novel feature extraction techniques including:
1. Wavelet transforms for multi-resolution analysis
2. Statistical features (skewness, kurtosis, etc.)
3. Frequency domain features (FFT)
4. Degradation rate estimation
5. Health indicators from charge/discharge curves
"""
import numpy as np
import pandas as pd
from typing import Dict, List
from scipy import signal
from scipy.stats import skew, kurtosis


def extract_wavelet_features(series: np.ndarray, wavelet: str = 'db4', levels: int = 3) -> Dict[str, float]:
    """
    Extract wavelet decomposition features
    """
    try:
        import pywt
        coeffs = pywt.wavedec(series, wavelet, level=levels)
        features = {}
        
        # Energy in each level
        for i, coeff in enumerate(coeffs):
            features[f'wavelet_energy_level_{i}'] = np.sum(coeff**2)
            features[f'wavelet_std_level_{i}'] = np.std(coeff) if len(coeff) > 0 else 0.0
        
        # Total energy
        features['wavelet_total_energy'] = np.sum([np.sum(c**2) for c in coeffs])
        
        return features
    except ImportError:
        # Fallback if pywt not available
        return {}


def extract_frequency_features(series: np.ndarray, sampling_rate: float = 1.0) -> Dict[str, float]:
    """
    Extract frequency domain features using FFT
    """
    if len(series) < 4:
        return {}
    
    # Remove NaN and inf
    clean_series = series[np.isfinite(series)]
    if len(clean_series) < 4:
        return {}
    
    # FFT
    fft_vals = np.fft.fft(clean_series)
    fft_freq = np.fft.fftfreq(len(clean_series), 1.0 / sampling_rate)
    
    # Power spectral density
    psd = np.abs(fft_vals)**2
    
    # Dominant frequency
    dominant_freq_idx = np.argmax(psd[1:]) + 1  # Skip DC component
    dominant_freq = np.abs(fft_freq[dominant_freq_idx])
    
    features = {
        'fft_dominant_freq': dominant_freq,
        'fft_total_power': np.sum(psd),
        'fft_mean_power': np.mean(psd),
        'fft_max_power': np.max(psd),
        'fft_power_std': np.std(psd),
    }
    
    # Spectral centroid
    if np.sum(psd) > 0:
        features['fft_spectral_centroid'] = np.sum(fft_freq * psd) / np.sum(psd)
    else:
        features['fft_spectral_centroid'] = 0.0
    
    return features


def extract_statistical_features(series: np.ndarray) -> Dict[str, float]:
    """
    Extract advanced statistical features
    """
    clean_series = series[np.isfinite(series)]
    if len(clean_series) < 3:
        return {}
    
    features = {
        'mean': np.mean(clean_series),
        'std': np.std(clean_series),
        'min': np.min(clean_series),
        'max': np.max(clean_series),
        'median': np.median(clean_series),
        'skewness': float(skew(clean_series)) if len(clean_series) > 2 else 0.0,
        'kurtosis': float(kurtosis(clean_series)) if len(clean_series) > 2 else 0.0,
        'q25': np.percentile(clean_series, 25),
        'q75': np.percentile(clean_series, 75),
        'iqr': np.percentile(clean_series, 75) - np.percentile(clean_series, 25),
    }
    
    # Coefficient of variation
    if np.abs(features['mean']) > 1e-6:
        features['cv'] = features['std'] / np.abs(features['mean'])
    else:
        features['cv'] = 0.0
    
    return features


def extract_degradation_features(group: pd.DataFrame) -> pd.DataFrame:
    """
    Extract degradation-related features from cycle data
    """
    g = group.sort_values('cycle_index').copy()
    
    # Capacity fade rate (exponential model: Q = Q0 * exp(-k*n))
    if 'Capacity_f' in g.columns and g['Capacity_f'].notna().sum() > 5:
        cap_clean = g['Capacity_f'].dropna()
        cycles_clean = g.loc[cap_clean.index, 'cycle_index'].values
        
        if len(cycles_clean) > 5:
            # Fit exponential decay: log(Q) = log(Q0) - k*n
            log_cap = np.log(cap_clean.values + 1e-6)
            # Linear fit: log_cap ~ a + b*cycle
            coeffs = np.polyfit(cycles_clean, log_cap, 1)
            fade_rate = -coeffs[0]  # Negative slope = fade rate
            
            g['degradation_rate_cap'] = fade_rate
        else:
            g['degradation_rate_cap'] = np.nan
    else:
        g['degradation_rate_cap'] = np.nan
    
    # IR growth rate
    if 'IR_f' in g.columns and g['IR_f'].notna().sum() > 5:
        ir_clean = g['IR_f'].dropna()
        cycles_clean = g.loc[ir_clean.index, 'cycle_index'].values
        
        if len(cycles_clean) > 5:
            # Fit linear growth: IR = IR0 + k*n
            coeffs = np.polyfit(cycles_clean, ir_clean.values, 1)
            growth_rate = coeffs[0]
            
            g['degradation_rate_ir'] = growth_rate
        else:
            g['degradation_rate_ir'] = np.nan
    else:
        g['degradation_rate_ir'] = np.nan
    
    # Rolling statistics for trend detection
    for window in [5, 10, 20]:
        if 'Capacity_f' in g.columns:
            g[f'cap_trend_{window}'] = g['Capacity_f'].rolling(window, min_periods=1).apply(
                lambda x: np.polyfit(range(len(x)), x.values, 1)[0] if len(x) > 1 else 0.0
            )
        
        if 'IR_f' in g.columns:
            g[f'ir_trend_{window}'] = g['IR_f'].rolling(window, min_periods=1).apply(
                lambda x: np.polyfit(range(len(x)), x.values, 1)[0] if len(x) > 1 else 0.0
            )
    
    # Acceleration of degradation (second derivative)
    if 'degradation_rate_cap' in g.columns:
        g['degradation_accel_cap'] = g['degradation_rate_cap'].diff()
    
    if 'degradation_rate_ir' in g.columns:
        g['degradation_accel_ir'] = g['degradation_rate_ir'].diff()
    
    return g


def extract_advanced_features(summary: pd.DataFrame) -> pd.DataFrame:
    """
    Extract all advanced features for the dataset
    """
    summary = summary.copy()
    
    # Group by cell for cell-specific features
    enhanced_frames = []
    
    for cell_id, group in summary.groupby('cell_id'):
        g = group.sort_values('cycle_index').copy()
        
        # Degradation features
        g = extract_degradation_features(g)
        
        # Rolling window features for IR and Capacity
        for col in ['IR_f', 'Capacity_f']:
            if col in g.columns:
                for window in [3, 5, 10]:
                    g[f'{col}_rolling_mean_{window}'] = g[col].rolling(window, min_periods=1).mean()
                    g[f'{col}_rolling_std_{window}'] = g[col].rolling(window, min_periods=1).std()
        
        # Statistical features over rolling windows
        for col in ['IR_f', 'Capacity_f']:
            if col in g.columns and g[col].notna().sum() > 10:
                window = 10
                rolling_stats = []
                for i in range(len(g)):
                    window_data = g[col].iloc[max(0, i-window+1):i+1].dropna()
                    if len(window_data) >= 3:
                        stats = extract_statistical_features(window_data.values)
                        rolling_stats.append(stats)
                    else:
                        rolling_stats.append({})
                
                # Add key statistics as features
                if rolling_stats:
                    for stat_name in ['skewness', 'kurtosis', 'cv']:
                        values = [s.get(stat_name, np.nan) for s in rolling_stats]
                        g[f'{col}_{stat_name}_rolling'] = values
        
        enhanced_frames.append(g)
    
    summary_enhanced = pd.concat(enhanced_frames, ignore_index=True)
    
    # Fill NaN values
    numeric_cols = summary_enhanced.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        summary_enhanced[col] = summary_enhanced[col].fillna(summary_enhanced[col].median())
        summary_enhanced[col] = summary_enhanced[col].replace([np.inf, -np.inf], summary_enhanced[col].median())
    
    return summary_enhanced

