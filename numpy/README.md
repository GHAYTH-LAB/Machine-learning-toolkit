# NumPy — Day 1 through Day 4

This README documents the examples and notes for `day1.py`, `day2.py`, `day3.py`, and `day4.py` in the `numpy` folder.

---

## Day 1 — NumPy Basics

What I learned:

- Python lists and how to access elements by index.
- `NumPy` stands for Numeric Python.
- Creating arrays with `np.array()`.
- Using `np.arange()` to generate ranges of values.
- Creating arrays of zeros with `np.zeros()`.
- Creating full arrays with `np.full()`.
- Accessing array shape with `.shape`.
- Indexing a NumPy array with square brackets.

Code (examples):

```python
# Reminder of python lists
List = [1, 2, 3, 4]
print(List)
# we access elements by index (first element index is 0)
print(List[0])

import numpy as np
array = np.array([0,1,2,3,4,5,6,7,8,9])
print(array)
print(array.shape)
array2 = np.arange(10)
print(array2)
array3 = np.arange(4,11,2)
print(array3)
New_array = np.zeros(10)
print(New_array)
New_array1 = np.zeros((3,4))
print(New_array1)
New_array2 = np.full((10),6)
print(New_array2)
New_array3 = np.full((5,4),8)
print(New_array3)
print(array3[3])
```

Example output:

```text
[1, 2, 3, 4]
1
[0 1 2 3 4 5 6 7 8 9]
(10,)
[0 1 2 3 4 5 6 7 8 9]
[ 4  6  8 10]
[0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
[[0. 0. 0. 0.]
 [0. 0. 0. 0.]
 [0. 0. 0. 0.]]
[6 6 6 6 6 6 6 6 6 6]
[[8 8 8 8]
 [8 8 8 8]
 [8 8 8 8]
 [8 8 8 8]
 [8 8 8 8]]
10
```

Key takeaways:

- NumPy arrays are more powerful than Python lists for numeric operations.
- `np.arange(start, stop, step)` creates a sequence of numbers in a range.
- `np.zeros(shape)` creates arrays filled with zeros.
- `np.full(shape, value)` creates arrays filled with the same value.
- `array.shape` tells you the dimensions of the array.
- You can index NumPy arrays using standard Python indexing.

---

## Day 2 — NumPy: Arrays and Slicing

This file documents the purpose, usage, and examples for `day2.py` in the `numpy` folder.

## Summary

`day2.py` demonstrates basic NumPy array creation, common slicing techniques, and simple 2D-array indexing. It includes examples for:

- 1D array creation (`array`, `arange`, `zeros`, `full`)
- Slicing and indexing (positive, negative indices, steps, reversing)
- Basic 2D array indexing and subarray selection

> Note: the file imports `pandas` but the script doesn't use it — you can safely remove that import.

## Requirements

- Python 3.7+
- NumPy

Install NumPy with:

```bash
pip install numpy
```

## Run

From the workspace root run:

```bash
python "numpy/day2.py"
```

## What the script does (high level)

- Creates several arrays and prints them
- Shows multiple slicing examples on a 1D array
- Demonstrates extracting rows, columns, and submatrices from a 2D array

## Key slicing examples used in `day2.py`

- `arr[start:stop]` — slice a contiguous range (stop is exclusive)
- `arr[start:]` — slice from `start` to the end
- `arr[-k:]` — last `k` elements
- `arr[::step]` — take every `step`-th element
- `arr[::-1]` — reverse the array
- `arr2d[row_indices, col_indices]` — 2D array row/column selection

## Sample output

(The script prints each result in order; approximate output below)

```
[1 3 5 7 9]
[10  9  8  7  6  5  4  3  2  1]
[0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
[6 6 6 6 6 6 6 6 6 6]
[5 7 8 4]
[8 4 3]
[8 4]
[8 4 3]
[1 7 4]
[3 4 8 7 5 1]
[1 2 3 4]
[ 2  6 10]
[[1 2]
 [5 6]]
```

---

## Day 3 — NumPy Views, Copies, Math, and Reshape

`day3.py` explores NumPy array views and copies, elementwise math, absolute values, reversal, and reshaping.

### Summary

`day3.py` includes:

- Creating a 1D integer array with `np.arange()`
- Demonstrating `view()` vs `copy()` behavior
- Using `np.sqrt()`, `np.max()`, `np.min()`, `np.abs()`, and array reversal
- Reshaping a 1D array into a 2D matrix with `.reshape()`

### Run

From the workspace root run:

```bash
python "numpy/day3.py"
```

### What the script does

- Creates an array from odd values using `np.arange(1, 12, 2)`
- Creates a view and a copy from the same source array
- Shows the effect of modifying the original array on the view but not on the copy
- Computes the square root, maximum, and minimum of a constant array
- Reverses a range array, takes the absolute value, and reshapes a 1D array into 2 rows by 3 columns

### Sample output

```
[41  3  5  7  9 11]
[1 5 7 8 9 11]
[16. 16. 16. 16. 16. 16.]
16
16
[15 13 11  9  7  3]
[[ 1  5  7]
 [ 8  9 11]]
```

---

## Day 4 — NumPy Iteration and Elementwise Operations

`day4.py` demonstrates array creation, reversing, absolute values, and elementwise iteration using `np.nditer()`.

### Summary
`day4.py` includes:

- Creating a 1D odd-number array with `np.arange()`
- Reversing and taking absolute values of a NumPy array
- Iterating over arrays with `np.nditer()`
- Demonstrating iteration over a 2D array in row-major order

### Run
From the workspace root run:

```bash
python "numpy/day4.py"
```

### What the script does

- Creates a 1D odd integer array with `np.arange(1, 9, 2)`
- Builds a second array, reverses it, and applies `np.abs()` 
- Iterates over the resulting 1D array using `np.nditer()` and prints values separated by `/`
- Creates a 2D array of ones and iterates through it with `np.nditer()` as well
- Builds and prints a 2D array, then iterates over it in row-major order

### Sample output

```
1/3/7/9/

1/1/1/1/1/1/1/1/1/1/1/1/

[[1 3 5]
 [2 4 6]]
1/3/5/2/4/6/
```
