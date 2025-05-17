#!/bin/bash
# Profile script for the new deterministic tree SCM with varying noise levels

echo "Profiling Deterministic Tree SCM with different noise types..."

# Create profile output directory
mkdir -p ./profiles

# Very clean data (no noise)
echo "Test 1: No noise (0%) - Profiling..."
python -m cProfile -o ./profiles/no_noise.prof generate_data.py \
    --n_datasets 3 \
    --num_gpus 1 \
    --prior deterministic_tree_scm \
    --min_features 20 \
    --max_features 50 \
    --min_seq 1000 \
    --max_seq 1500 \
    --min_classes 2 \
    --max_classes 3 \
    --out_dir ./synth/profile_test/no_noise \
    --inner_bsz 32 \
    --no_causal \
    --num_layers 1 \
    --min_swap_prob 0.0 \
    --max_swap_prob 0.0 \
    --transform_type polynomial \
    --noise_type swap \
    --noise_std 0.001

# Low noise with corruption
echo "Test 2: Low corruption (15%) - Profiling..."
python -m cProfile -o ./profiles/low_corrupt.prof generate_data.py \
    --n_datasets 3 \
    --num_gpus 1 \
    --prior deterministic_tree_scm \
    --min_features 20 \
    --max_features 50 \
    --min_seq 1000 \
    --max_seq 1500 \
    --min_classes 2 \
    --max_classes 3 \
    --out_dir ./synth/profile_test/low_corrupt \
    --inner_bsz 32 \
    --no_causal \
    --num_layers 1 \
    --min_swap_prob 0.1 \
    --max_swap_prob 0.15 \
    --transform_type polynomial \
    --noise_type corrupt \
    --noise_std 0.001

# Moderate mixed noise
echo "Test 3: Moderate mixed noise (20-35%) - Profiling..."
python -m cProfile -o ./profiles/moderate_mixed.prof generate_data.py \
    --n_datasets 3 \
    --num_gpus 1 \
    --prior deterministic_tree_scm \
    --min_features 20 \
    --max_features 50 \
    --min_seq 1000 \
    --max_seq 1500 \
    --min_classes 2 \
    --max_classes 3 \
    --out_dir ./synth/profile_test/moderate_mixed \
    --inner_bsz 32 \
    --no_causal \
    --num_layers 2 \
    --min_swap_prob 0.2 \
    --max_swap_prob 0.35 \
    --transform_type mixed \
    --noise_type mixed \
    --noise_std 0.001

# High boundary blur noise
echo "Test 4: High boundary blur (50-70%) - Profiling..."
python -m cProfile -o ./profiles/high_blur.prof generate_data.py \
    --n_datasets 3 \
    --num_gpus 1 \
    --prior deterministic_tree_scm \
    --min_features 20 \
    --max_features 50 \
    --min_seq 1000 \
    --max_seq 1500 \
    --min_classes 2 \
    --max_classes 3 \
    --out_dir ./synth/profile_test/high_blur \
    --inner_bsz 32 \
    --no_causal \
    --num_layers 3 \
    --min_swap_prob 0.5 \
    --max_swap_prob 0.7 \
    --transform_type mixed \
    --noise_type boundary_blur \
    --noise_std 0.001

echo "Profiling complete!"
echo "Analyzing profiles..."

# Create Python script to analyze profiles
cat > analyze_profiles.py << 'EOF'
import pstats
import sys

def analyze_profile(filename, noise_type):
    stats = pstats.Stats(filename)
    print(f"\n{'='*60}")
    print(f"Profile Analysis: {noise_type}")
    print(f"{'='*60}")
    
    # Print top 20 time-consuming functions
    print("\nTop 20 time-consuming functions:")
    stats.sort_stats('tottime')
    stats.print_stats(20)
    
    # Focus on DeterministicTreeSCM and noise injection methods
    print(f"\n{'='*60}")
    print(f"DeterministicTreeSCM and noise-related functions:")
    print(f"{'='*60}")
    stats.print_stats('deterministic_tree_scm')
    stats.print_stats('_inject_noise')
    stats.print_stats('_swap_targets')
    stats.print_stats('_generate_deterministic_targets')
    stats.print_stats('fit')
    
    return stats

# Analyze each profile
profiles = [
    ("./profiles/no_noise.prof", "No Noise (0%)"),
    ("./profiles/low_corrupt.prof", "Low Corruption (15%)"),
    ("./profiles/moderate_mixed.prof", "Moderate Mixed (20-35%)"),
    ("./profiles/high_blur.prof", "High Boundary Blur (50-70%)")
]

stats_list = []
for filename, noise_type in profiles:
    try:
        stats = analyze_profile(filename, noise_type)
        stats_list.append((noise_type, stats))
    except Exception as e:
        print(f"Error analyzing {filename}: {e}")

# Compare total times
print(f"\n{'='*60}")
print("Total Time Comparison:")
print(f"{'='*60}")
for noise_type, stats in stats_list:
    total_time = stats.total_tt
    print(f"{noise_type:30} {total_time:10.3f} seconds")
EOF

python analyze_profiles.py