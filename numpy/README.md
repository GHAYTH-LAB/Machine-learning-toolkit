# NumPy — Day 1 & Day 2

This README documents the examples and notes for `day1.py` and `day2.py` in the `numpy` folder.

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

