#!/usr/bin/env python3
"""
Profile different noise injection methods in DeterministicTreeSCM
"""
import numpy as np
import torch
import time
import sys
import os

# Add the source directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from tabicl.prior.deterministic_tree_scm import DeterministicTreeLayer


def profile_noise_injection(n_samples=5000, n_features=50, n_runs=10):
    """Profile different noise injection methods."""
    
    # Generate sample data
    X = torch.randn(n_samples, n_features)
    
    noise_configs = [
        ("No noise", {"noise_type": "swap", "swap_prob": 0.0}),
        ("Low swap (10%)", {"noise_type": "swap", "swap_prob": 0.1}),
        ("Low corrupt (15%)", {"noise_type": "corrupt", "swap_prob": 0.15}),
        ("Moderate mixed (30%)", {"noise_type": "mixed", "swap_prob": 0.3}),
        ("High boundary blur (60%)", {"noise_type": "boundary_blur", "swap_prob": 0.6}),
    ]
    
    results = []
    
    for name, config in noise_configs:
        print(f"\nProfiling: {name}")
        
        # Create layer with specific noise configuration
        layer = DeterministicTreeLayer(
            tree_model="xgboost",
            max_depth=3,
            n_estimators=10,
            out_dim=10,
            transform_type="polynomial",
            device="cpu",
            **config
        )
        
        # Profile the forward pass
        times = []
        fit_times = []
        noise_times = []
        
        for run in range(n_runs):
            # Time the entire forward pass
            start_total = time.time()
            
            # Generate deterministic targets
            y_det = layer._generate_deterministic_targets(X)
            
            # Time just the noise injection
            start_noise = time.time()
            y_noisy = layer._inject_noise(y_det)
            noise_time = time.time() - start_noise
            
            # Time the model fitting
            start_fit = time.time()
            layer.model.fit(X.numpy(), y_noisy)
            fit_time = time.time() - start_fit
            
            total_time = time.time() - start_total
            
            times.append(total_time)
            fit_times.append(fit_time)
            noise_times.append(noise_time)
            
            if run == 0:
                print(f"  Run {run+1}: Total={total_time:.3f}s, Fit={fit_time:.3f}s, Noise={noise_time:.3f}s")
        
        avg_total = np.mean(times)
        avg_fit = np.mean(fit_times)
        avg_noise = np.mean(noise_times)
        
        print(f"  Average: Total={avg_total:.3f}s, Fit={avg_fit:.3f}s, Noise={avg_noise:.3f}s")
        print(f"  Fit percentage: {avg_fit/avg_total*100:.1f}%")
        print(f"  Noise percentage: {avg_noise/avg_total*100:.1f}%")
        
        results.append({
            "name": name,
            "total_time": avg_total,
            "fit_time": avg_fit,
            "noise_time": avg_noise,
            "fit_pct": avg_fit/avg_total*100,
            "noise_pct": avg_noise/avg_total*100
        })
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{'Configuration':25} {'Total(s)':>10} {'Fit(s)':>10} {'Noise(s)':>10} {'Fit%':>8} {'Noise%':>8}")
    print("-"*60)
    
    baseline_time = results[0]["total_time"]
    for r in results:
        speedup = baseline_time / r["total_time"] if r["total_time"] > 0 else 0
        print(f"{r['name']:25} {r['total_time']:10.3f} {r['fit_time']:10.3f} "
              f"{r['noise_time']:10.3f} {r['fit_pct']:8.1f} {r['noise_pct']:8.1f}")
    
    # Analyze the bottleneck
    print("\n" + "="*60)
    print("ANALYSIS")
    print("="*60)
    
    for i in range(1, len(results)):
        r = results[i]
        baseline = results[0]
        slowdown = r["total_time"] / baseline["total_time"]
        fit_increase = r["fit_time"] - baseline["fit_time"]
        noise_increase = r["noise_time"] - baseline["noise_time"]
        
        print(f"\n{r['name']} vs No noise:")
        print(f"  Slowdown: {slowdown:.2f}x")
        print(f"  Fit time increase: {fit_increase:.3f}s ({fit_increase/r['total_time']*100:.1f}% of total)")
        print(f"  Noise time increase: {noise_increase:.3f}s ({noise_increase/r['total_time']*100:.1f}% of total)")
        
        if fit_increase > noise_increase:
            print(f"  Main bottleneck: Model fitting")
        else:
            print(f"  Main bottleneck: Noise injection")


if __name__ == "__main__":
    print("Profiling noise injection methods...")
    profile_noise_injection()