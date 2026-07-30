import time
import torch
import numpy as np
import sys
import os

# Add src to path so we can import models
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from models.pinn_causal_attribution import PINNCausalAttributionModel

def test_pinn_inference():
    print("Initializing Hybrid PINN for inference test...")
    # Initialize model with standard parameters matching the paper
    model = PINNCausalAttributionModel(
        input_dim=15, 
        hidden_dim=64, 
        latent_dim=32, 
        num_mechanisms=5
    )
    
    # Put in eval mode to disable dropout, etc.
    model.eval()
    
    # Create a batch of dummy data (batch size 1, like real-time BMS inference)
    # The expected shape is (batch_size, seq_len, input_dim) based on the forward pass
    # Using seq_len=50 as per the 50-cycle history mentioned in the paper
    dummy_input = torch.randn(1, 50, 15)
    
    print("\nRunning warm-up passes...")
    for _ in range(3):
        with torch.no_grad():
            _, _, _, _ = model(dummy_input)
            
    print("\nMeasuring inference time (100 iterations)...")
    times = []
    
    for _ in range(100):
        start_time = time.perf_counter()
        with torch.no_grad():
            output, params, mechanisms, total_loss = model(dummy_input)
        end_time = time.perf_counter()
        times.append(end_time - start_time)
        
    avg_time = np.mean(times)
    std_time = np.std(times)
    
    print(f"\nResults (Batch Size 1):")
    print(f"Average time per forward pass: {avg_time:.4f} seconds ({avg_time*1000:.2f} ms)")
    print(f"Standard deviation: {std_time:.4f} seconds")
    
    print("\nChecking parameter count...")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total model parameters: {total_params:,}")

if __name__ == "__main__":
    test_pinn_inference()
