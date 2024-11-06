# Genetic Auto Piecewise Linear Approximation 
This toolkit provides utilities for approximating activation functions using a piecewise linear method. The toolkit can:

1. Automatically find the best split points for an activation function (autopwl).
2. Store the parameters (split points, coefficients, biases) for offline usage (offline_pwlstore).
3. Compute the errors between the original and approximated functions (compute_errors).

## Prerequisites
Ensure you have installed the required Python packages:
```python
pip install numpy matplotlib scipy deap
```

## Usage
### 1. autopwl
Find the best split points for a specified activation function.
```python
python genetic_auto_hardpwl.py autopwl --act_func [ACTIVATION_FUNCTION] [--num_splits SPLIT_COUNT] [--decimal_bit DECIMAL_BIT]
```
**Parameters:**
- [ACTIVATION_FUNCTION]: Name of the activation function (e.g., hswish).
- SPLIT_COUNT: Number of split points you want (optional, default is 7).
- DECIMAL_BIT: Decimal precision (optional, default is 5).
**Example:**
```python
python genetic_auto_hardpwl.py autopwl --act_func 'hswish' --num_splits 7 --decimal_bit 5
```
### 2. offline_pwlstore
Store parameters for offline usage.
```python
python genetic_auto_hardpwl.py offline_pwlstore --act_func [ACTIVATION_FUNCTION] [--decimal_bit_range BIT_RANGE]
```
**Parameters:**
- [ACTIVATION_FUNCTION]: Name of the activation function (e.g., hswish).
- BIT_RANGE: Maximum bit range for storage (optional, default is 16).
  
**Example:**
```python
python genetic_auto_hardpwl.py offline_pwlstore --act_func hswish --decimal_bit_range 13 --num_splits 7 --total_iters 100 --random
```

### 3. compute errors
Compute the errors between the original and approximated activation functions.
```python
python genetic_auto_hardpwl.py compute_errors --act_func [ACTIVATION_FUNCTION] --split_points SPLIT_POINT_1 SPLIT_POINT_2 ...
```
**Parameters:**
- [ACTIVATION_FUNCTION]: Name of the activation function (e.g., hswish).
- SPLIT_POINT_1, SPLIT_POINT_2, ...: Split points you want to use for error computation.
  
**Example:**
```python
python genetic_auto_hardpwl.py compute_errors --act_func hswish --split_points -3 -2 -1 0 1 2 3 --coeff --bias
```