# NumPy Day 1

## What I learned

- Python lists and how to access elements by index.
- `NumPy` stands for Numeric Python.
- Creating arrays with `np.array()`.
- Using `np.arange()` to generate ranges of values.
- Creating arrays of zeros with `np.zeros()`.
- Creating full arrays with `np.full()`.
- Accessing array shape with `.shape`.
- Indexing a NumPy array with square brackets.

## Code

```python
# Reminder of python lists
List=[1,2,3,4]
print(List)
# we acces elements by index(first element index is 0)
print(List[0])
# Numpy=Numeric python
# ndarray=n-dimensional array()
import numpy as np
array=np.array([0,1,2,3,4,5,6,7,8,9])
print(array)
print(array.shape)
array2=np.arange(10)
print(array2)
array3=np.arange(4,11,2)
print(array3)
New_array=np.zeros(10)
print(New_array)
New_array1=np.zeros((3,4))
print(New_array1)
New_array2=np.full((10),6)
print(New_array2)
New_array3=np.full((5,4),8)
print(New_array3)
print(array3[3])
```

## Example output

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

## Notes / Key takeaways

- NumPy arrays are more powerful than Python lists for numeric operations.
- `np.arange(start, stop, step)` creates a sequence of numbers in a range.
- `np.zeros(shape)` creates arrays filled with zeros.
- `np.full(shape, value)` creates arrays filled with the same value.
- `array.shape` tells you the dimensions of the array.
- You can index NumPy arrays using standard Python indexing.
