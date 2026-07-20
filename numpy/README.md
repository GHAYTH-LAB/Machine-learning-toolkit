# NumPy — Day 1 through Day 7

This README documents the examples and notes for `day1.py`, `day2.py`, `day3.py`, `day4.py`, `day5.py`, `day6.py`, and `day7.py` in the `numpy` folder.

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
print(array3)
print(New_array)
New_array1 = np.zeros((3,4))
print(New_array1)
New_array2 = np.full((10),6)
print(New_array2)
New_array3 = np.full((5,4),8)
print(New_array3)
print(array3[3])
```
[1, 2, 3, 4]
1
[0 1 2 3 4 5 6 7 8 9]
(10,)
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
- You can index NumPy arrays using standard Python indexing.

---

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


```bash
python "numpy/day2.py"
```

## What the script does (high level)


## Sample output

(The script prints each result in order; approximate output below)

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

`day3.py` explores NumPy array views and copies, elementwise math, absolute values, reversal, and reshaping.

### Summary

`day3.py` includes:

- Creating a 1D integer array with `np.arange()`
### Run

From the workspace root run:

```bash
python "numpy/day3.py"
```

### What the script does


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
- Iterating over arrays with `np.nditer()`
- Demonstrating iteration over a 2D array in row-major order

### Run
From the workspace root run:

```bash
python "numpy/day4.py"
```

- Builds a second array, reverses it, and applies `np.abs()` 
- Iterates over the resulting 1D array using `np.nditer()` and prints values separated by `/`
- Creates a 2D array of ones and iterates through it with `np.nditer()` as well
- Builds and prints a 2D array, then iterates over it in row-major order

### Sample output

```

1/1/1/1/1/1/1/1/1/1/1/1/

[[1 3 5]
 [2 4 6]]
1/3/5/2/4/6/
```

`day5.py` shows how to search arrays and select elements based on conditions using `np.where()`.

### Summary
`day5.py` includes:

- Creating a 1D NumPy array with `np.array()`
- Searching for values with `np.where()`
- Selecting odd elements using a modulo condition

### Run
### What the script does

- Creates a 1D array with repeated values
- Uses `np.where(np1 % 2 == 1)` to locate odd elements
- Prints the original array and the matching indices for odd values

### Sample output

```
[ 1  2  3  4  5  6  7  8  9 10  3]
[0 2 4 6 8 10]
```

---


`day6.py` demonstrates sorting of numeric, string, and boolean arrays, plus descending order sorting using slicing.

### Summary
`day6.py` includes:

- Sorting a numeric array with `np.sort()`
- Sorting a string array alphabetically with `np.sort()`
- Sorting a boolean array with `np.sort()`

```bash
python "numpy/day6.py"
```

### What the script does

- Creates a numeric array and prints it
- Sorts the numeric array and prints the sorted result without modifying the original
- Creates a string array and sorts it alphabetically

### Sample output

```
[6 7 4 9 0 2 1]
[0 1 2 4 6 7 9]
['John' 'Tina' 'Aaron' 'Zed']
['Aaron' 'John' 'Tina' 'Zed']
[ True False False  True]
[False False  True  True]
[6 7 4 9 0 2 1]
[0 1 2 4 6 7 9]
[9 7 6 4 2 1 0]

---

## Day 7 — Boolean Indexing and Filtering

`day7.py` demonstrates boolean masking (indexing) to filter NumPy arrays using both an explicit loop and a vectorized expression.

### Summary

`day7.py` includes:

- Creating a 1D numeric array with `np.array()`
- Building a boolean mask using a Python loop
- Using the vectorized shortcut `np1 % 2 == 0` to create a mask
- Applying the boolean mask to select elements from the array

### Run
From the workspace root run:

```bash
python "numpy/day7.py"
```

### What the script does

- Builds a boolean list by iterating over the array and appending `True`/`False` according to a condition (even numbers)
- Uses that list as a mask to select elements from the original array
- Demonstrates the concise, vectorized approach `np1 % 2 == 0` which yields the same result

### Sample output

```
[1 2 3 4 5 6 7 8 9 10]
[False, True, False, True, False, True, False, True, False, True]
[2 4 6 8 10]
[1 2 3 4 5 6 7 8 9 10]
[False  True False  True False  True False  True False  True]
[2 4 6 8 10]
```

---

## Where I learned

- Playlist: [NumPy tutorial playlist](https://youtube.com/playlist?list=PLCC34OHNcOtpalASMlX2HHdsLNipyyhbK&si=z2IYHM3CvY9O2CVg)

- [![NumPy playlist thumbnail](https://img.youtube.com/vi/gnKbAAVUzro/hqdefault.jpg)](https://youtube.com/playlist?list=PLCC34OHNcOtpalASMlX2HHdsLNipyyhbK&si=z2IYHM3CvY9O2CVg)
