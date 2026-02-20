import torch
from src.models.pinn_causal_attribution import PINNCausalAttributionModel
from src.optimization.counterfactual_intervention import BatteryState, CausalAttribution, CounterfactualSimulator, InterventionOptimizer

def test_integration():
    model = PINNCausalAttributionModel()
    # Try to load weights if they exist
    try:
        model.load_state_dict(torch.load("reports/pinn_causal/pinn_causal_retrained.pt", map_location='cpu'))
        print("Loaded retrained weights.")
    except Exception as e:
        print("Could not load weights, using untrained model.", e)

    state = BatteryState(soc=0.3, temperature=10.0, current=2.5, voltage=3.7, cycle_count=100, c_rate=1.25, capacity=2.0)
    current_attribution = CausalAttribution(sei_growth=0.15, lithium_plating=0.65, active_material_loss=0.10, electrolyte_loss=0.05, corrosion=0.05)
    
    # Test with injected model
    print("\n--- Testing with Injected PINN ---")
    simulator = CounterfactualSimulator(hybrid_pinn_model=model)
    optimizer = InterventionOptimizer(simulator)
    
    recs = optimizer.optimize(state, current_attribution, top_k=2)
    for r in recs:
        from src.optimization.counterfactual_intervention import format_recommendation
        r['current_attribution'] = current_attribution
        print(format_recommendation(r))

if __name__ == "__main__":
    test_integration()
