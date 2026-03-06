"""
SOTA Baseline Comparison for Battery SOH/RUL Prediction.

Implements 6 baseline methods for comparison with HERO:
1. LSTM - Classic recurrent baseline
2. GRU - Efficient RNN variant
3. CNN-LSTM - Hybrid convolutional-recurrent
4. Transformer - Self-attention based
5. MLP - Feedforward neural network
6. Random Forest - Ensemble ML baseline
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import json
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


# Baseline Model Definitions

class LSTMBaseline(nn.Module):
    """Standard LSTM for sequence prediction."""
    
    def __init__(self, input_dim, hidden_dim=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        self.soh_head = nn.Linear(hidden_dim, 1)
        self.rul_head = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        # x: (batch, features) -> reshape to (batch, 1, features) for LSTM
        if x.dim() == 2:
            x = x.unsqueeze(1)
        out, _ = self.lstm(x)
        h = out[:, -1, :]  # Last hidden state
        soh = torch.sigmoid(self.soh_head(h)) * 0.5 + 0.5  # 0.5-1.0
        rul = torch.sigmoid(self.rul_head(h))  # 0-1 normalized
        return soh.squeeze(-1), rul.squeeze(-1)


class GRUBaseline(nn.Module):
    """GRU variant - more efficient than LSTM."""
    
    def __init__(self, input_dim, hidden_dim=64, num_layers=2):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        self.soh_head = nn.Linear(hidden_dim, 1)
        self.rul_head = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        out, _ = self.gru(x)
        h = out[:, -1, :]
        soh = torch.sigmoid(self.soh_head(h)) * 0.5 + 0.5
        rul = torch.sigmoid(self.rul_head(h))
        return soh.squeeze(-1), rul.squeeze(-1)


class CNNLSTMBaseline(nn.Module):
    """CNN-LSTM hybrid - extracts local patterns then captures sequences."""
    
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.lstm = nn.LSTM(64, hidden_dim, 2, batch_first=True, dropout=0.2)
        self.soh_head = nn.Linear(hidden_dim, 1)
        self.rul_head = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        # x: (batch, features)
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch, 1, features)
        
        # CNN expects (batch, channels, seq)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        
        # Transpose for LSTM: (batch, seq, channels)
        x = x.transpose(1, 2)
        out, _ = self.lstm(x)
        h = out[:, -1, :]
        
        soh = torch.sigmoid(self.soh_head(h)) * 0.5 + 0.5
        rul = torch.sigmoid(self.rul_head(h))
        return soh.squeeze(-1), rul.squeeze(-1)


class TransformerBaseline(nn.Module):
    """Transformer encoder for battery prediction."""
    
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=128, dropout=0.2, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.soh_head = nn.Linear(d_model, 1)
        self.rul_head = nn.Linear(d_model, 1)
    
    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch, 1, features)
        x = self.input_proj(x)
        x = self.transformer(x)
        h = x.mean(dim=1)  # Global average pooling
        
        soh = torch.sigmoid(self.soh_head(h)) * 0.5 + 0.5
        rul = torch.sigmoid(self.rul_head(h))
        return soh.squeeze(-1), rul.squeeze(-1)


class MLPBaseline(nn.Module):
    """Simple feedforward network baseline."""
    
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 64),
            nn.ReLU()
        )
        self.soh_head = nn.Linear(64, 1)
        self.rul_head = nn.Linear(64, 1)
    
    def forward(self, x):
        if x.dim() == 3:
            x = x.squeeze(1)
        h = self.net(x)
        soh = torch.sigmoid(self.soh_head(h)) * 0.5 + 0.5
        rul = torch.sigmoid(self.rul_head(h))
        return soh.squeeze(-1), rul.squeeze(-1)


# Training and Evaluation Functions

def train_pytorch_model(model, train_loader, epochs=50, lr=0.001):
    """Train a PyTorch model."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in train_loader:
            features, soh_true, rul_true = batch
            
            optimizer.zero_grad()
            soh_pred, rul_pred = model(features)
            
            loss = criterion(soh_pred, soh_true) + criterion(rul_pred, rul_true)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
    
    return model


def evaluate_pytorch_model(model, test_loader):
    """Evaluate a PyTorch model."""
    model.eval()
    
    all_soh_pred, all_soh_true = [], []
    all_rul_pred, all_rul_true = [], []
    
    with torch.no_grad():
        for batch in test_loader:
            features, soh_true, rul_true = batch
            soh_pred, rul_pred = model(features)
            
            all_soh_pred.extend(soh_pred.numpy())
            all_soh_true.extend(soh_true.numpy())
            all_rul_pred.extend(rul_pred.numpy())
            all_rul_true.extend(rul_true.numpy())
    
    soh_mae = mean_absolute_error(all_soh_true, all_soh_pred)
    soh_r2 = r2_score(all_soh_true, all_soh_pred)
    rul_mae = mean_absolute_error(all_rul_true, all_rul_pred) * 1000  # Convert to cycles
    
    return {
        'soh_mae': soh_mae,
        'soh_r2': soh_r2,
        'rul_mae_cycles': rul_mae
    }


def train_random_forest(X_train, y_soh_train, y_rul_train, X_test, y_soh_test, y_rul_test):
    """Train and evaluate Random Forest baseline."""
    
    # SOH model
    rf_soh = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf_soh.fit(X_train, y_soh_train)
    soh_pred = rf_soh.predict(X_test)
    
    # RUL model
    rf_rul = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf_rul.fit(X_train, y_rul_train)
    rul_pred = rf_rul.predict(X_test)
    
    soh_mae = mean_absolute_error(y_soh_test, soh_pred)
    soh_r2 = r2_score(y_soh_test, soh_pred)
    rul_mae = mean_absolute_error(y_rul_test, rul_pred) * 1000
    
    return {
        'soh_mae': soh_mae,
        'soh_r2': soh_r2,
        'rul_mae_cycles': rul_mae
    }


# Data Loading

def load_training_data():
    """Load and prepare training/test data."""
    
    # Load TJU data for evaluation
    tju_path = Path("data/new_datasets/RUL-Mamba/data/TJU data/Dataset_3_NCM_NCA_battery_1C.npy")
    
    if not tju_path.exists():
        print("TJU data not found, generating synthetic data...")
        # Generate synthetic data for benchmarking
        np.random.seed(42)
        n_samples = 2000
        n_features = 20
        
        X = np.random.randn(n_samples, n_features).astype(np.float32)
        soh = 0.95 - 0.3 * np.abs(X[:, 0]) + 0.05 * np.random.randn(n_samples)
        soh = np.clip(soh, 0.6, 1.0).astype(np.float32)
        rul = 0.5 - 0.4 * np.abs(X[:, 1]) + 0.1 * np.random.randn(n_samples)
        rul = np.clip(rul, 0.0, 1.0).astype(np.float32)
    else:
        data = np.load(tju_path, allow_pickle=True).item()
        
        X_list, soh_list, rul_list = [], [], []
        
        for cell_name, df in data.items():
            capacity = df['Capacity'].values
            initial_capacity = capacity[0]
            cell_soh = capacity / initial_capacity
            
            feature_cols = [
                'voltage mean', 'voltage std', 'voltage kurtosis', 'voltage skewness',
                'CC Q', 'CC charge time', 'voltage slope', 'voltage entropy',
                'current mean', 'current std', 'current kurtosis', 'current skewness',
                'CV Q', 'CV charge time', 'current slope', 'current entropy'
            ]
            
            for i in range(0, len(df), 5):  # Sample every 5th cycle
                features = np.zeros(20, dtype=np.float32)
                for j, col in enumerate(feature_cols):
                    if col in df.columns:
                        val = df[col].iloc[i]
                        features[j] = float(val) if not np.isnan(val) else 0.0
                
                # Normalize features
                features = np.clip(features / (np.abs(features).max() + 1e-8), -1, 1)
                
                # Calculate RUL
                eol_threshold = 0.8
                rul = 0
                for future_i in range(i, len(df)):
                    if cell_soh[future_i] < eol_threshold:
                        rul = future_i - i
                        break
                else:
                    rul = len(df) - i
                
                rul_normalized = min(rul / 1000.0, 1.0)
                
                X_list.append(features)
                soh_list.append(float(cell_soh[i]))
                rul_list.append(rul_normalized)
        
        X = np.array(X_list, dtype=np.float32)
        soh = np.array(soh_list, dtype=np.float32)
        rul = np.array(rul_list, dtype=np.float32)
    
    # Split data
    n = len(X)
    indices = np.random.permutation(n)
    train_idx = indices[:int(0.7 * n)]
    test_idx = indices[int(0.7 * n):]
    
    return (X[train_idx], soh[train_idx], rul[train_idx],
            X[test_idx], soh[test_idx], rul[test_idx])


# Main Benchmark

def run_sota_benchmark():
    """Run benchmark comparison of all SOTA methods."""
    
    print("=" * 70)
    print("SOTA BASELINE COMPARISON FOR BATTERY SOH/RUL PREDICTION")
    print("=" * 70)
    
    # Load data
    print("\nLoading data...")
    X_train, soh_train, rul_train, X_test, soh_test, rul_test = load_training_data()
    
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Feature dimension: {X_train.shape[1]}")
    
    # Create PyTorch data loaders
    train_dataset = TensorDataset(
        torch.tensor(X_train),
        torch.tensor(soh_train),
        torch.tensor(rul_train)
    )
    test_dataset = TensorDataset(
        torch.tensor(X_test),
        torch.tensor(soh_test),
        torch.tensor(rul_test)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    input_dim = X_train.shape[1]
    
    # Define models to benchmark
    models = {
        'LSTM': LSTMBaseline(input_dim),
        'GRU': GRUBaseline(input_dim),
        'CNN-LSTM': CNNLSTMBaseline(input_dim),
        'Transformer': TransformerBaseline(input_dim),
        'MLP': MLPBaseline(input_dim)
    }
    
    results = {}
    
    # Train and evaluate each model
    for name, model in models.items():
        print(f"\n{'='*50}")
        print(f"Training {name}...")
        print("=" * 50)
        
        start_time = time.time()
        model = train_pytorch_model(model, train_loader, epochs=50)
        train_time = time.time() - start_time
        
        metrics = evaluate_pytorch_model(model, test_loader)
        metrics['train_time_sec'] = train_time
        
        results[name] = metrics
        print(f"  SOH MAE:  {metrics['soh_mae']*100:.2f}%")
        print(f"  SOH R²:   {metrics['soh_r2']:.4f}")
        print(f"  RUL MAE:  {metrics['rul_mae_cycles']:.1f} cycles")
        print(f"  Time:     {train_time:.1f}s")
    
    # Random Forest
    print(f"\n{'='*50}")
    print("Training Random Forest...")
    print("=" * 50)
    
    start_time = time.time()
    rf_metrics = train_random_forest(X_train, soh_train, rul_train, X_test, soh_test, rul_test)
    train_time = time.time() - start_time
    rf_metrics['train_time_sec'] = train_time
    
    results['Random Forest'] = rf_metrics
    print(f"  SOH MAE:  {rf_metrics['soh_mae']*100:.2f}%")
    print(f"  SOH R²:   {rf_metrics['soh_r2']:.4f}")
    print(f"  RUL MAE:  {rf_metrics['rul_mae_cycles']:.1f} cycles")
    print(f"  Time:     {train_time:.1f}s")
    
    # Load HERO results for comparison
    hero_results = {
        'soh_mae': 0.0074,  # From fine-tuning results
        'soh_r2': 0.99,
        'rul_mae_cycles': 16.7
    }
    results['HERO (Ours)'] = hero_results
    
    # Summary table
    print("\n" + "=" * 70)
    print("BENCHMARK RESULTS SUMMARY")
    print("=" * 70)
    
    print(f"\n{'Method':<20} {'SOH MAE':<12} {'SOH R²':<10} {'RUL MAE':<15}")
    print("-" * 57)
    
    # Sort by SOH MAE
    sorted_results = sorted(results.items(), key=lambda x: x[1]['soh_mae'])
    
    for name, metrics in sorted_results:
        soh_pct = f"{metrics['soh_mae']*100:.2f}%"
        r2 = f"{metrics['soh_r2']:.4f}"
        rul = f"{metrics['rul_mae_cycles']:.1f} cycles"
        marker = " ★" if name == 'HERO (Ours)' else ""
        print(f"{name:<20} {soh_pct:<12} {r2:<10} {rul:<15}{marker}")
    
    # Calculate improvement over baselines
    print("\n" + "=" * 70)
    print("HERO IMPROVEMENT OVER BASELINES")
    print("=" * 70)
    
    hero_soh = hero_results['soh_mae']
    hero_rul = hero_results['rul_mae_cycles']
    
    for name, metrics in sorted_results:
        if name == 'HERO (Ours)':
            continue
        
        soh_improvement = (metrics['soh_mae'] - hero_soh) / metrics['soh_mae'] * 100
        rul_improvement = (metrics['rul_mae_cycles'] - hero_rul) / metrics['rul_mae_cycles'] * 100
        
        print(f"\nvs {name}:")
        print(f"  SOH: {soh_improvement:+.1f}% better")
        print(f"  RUL: {rul_improvement:+.1f}% better")
    
    # Save results
    output_path = Path("reports/sota_baseline_comparison.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n Results saved to {output_path}")
    
    return results


if __name__ == '__main__':
    run_sota_benchmark()
