# Class Imbalance Control in TabICL

This document explains how to generate synthetic datasets with controlled class imbalance in TabICL.

## Overview

The `generate_data.py` script now supports generating datasets with controlled class imbalance through two new flags:

- `--min_imbalance_ratio`: Minimum ratio between the largest and smallest class sizes
- `--max_imbalance_ratio`: Maximum ratio between the largest and smallest class sizes

## Usage Examples

### Balanced Classes (Default)
```bash
python generate_data.py --n_datasets 1000 --out_dir data/balanced
```
This generates datasets with balanced classes (all classes have approximately equal representation).

### Fixed Imbalance Ratio
```bash
python generate_data.py --n_datasets 1000 --out_dir data/imbalanced_3x \
    --min_imbalance_ratio 3.0 --max_imbalance_ratio 3.0
```
This generates datasets where the largest class is exactly 3x the size of the smallest class.

### Variable Imbalance Ratio
```bash
python generate_data.py --n_datasets 1000 --out_dir data/imbalanced_2_10x \
    --min_imbalance_ratio 2.0 --max_imbalance_ratio 10.0
```
This generates datasets where the imbalance ratio varies between 2x and 10x.

### High Imbalance
```bash
python generate_data.py --n_datasets 1000 --out_dir data/highly_imbalanced \
    --min_imbalance_ratio 10.0 --max_imbalance_ratio 50.0
```
This generates highly imbalanced datasets where the largest class can be 10-50x larger than the smallest.

## Implementation Details

The imbalance is implemented by:

1. Calculating class proportions based on a geometric progression
2. Adjusting class boundaries to achieve the desired proportions
3. Ensuring both train and test splits maintain the class distribution

The imbalance ratio is defined as:
```
imbalance_ratio = size_of_largest_class / size_of_smallest_class
```

## Notes

- An imbalance ratio of 1.0 means perfectly balanced classes
- Higher ratios create more imbalanced distributions
- The actual imbalance may vary slightly due to randomness and rounding
- Both SCM-based and dummy priors support imbalanced generation