## Day 1 of Learning pandas
***Date:*** July 11 2026

### What I learned
- Importing the `pandas` library
- Creating a `pd.Series(data)` object
- Creating a `pd.Series(data, index=[...])` object
- Selecting data with `.loc[...]`

### Code
![My attempt](assets/pandas/Importing_pandas_library_day1-corrected.png)

### Example_output
![Execution](assets/pandas/execution_day1_pandas.png)

### Notes / Key takeaways
- `pd.Series(data)` creates a one-dimensional labeled data structure.
- `pd.Series(data, index=[...])` sets custom index labels.
- `.loc[index]` selects rows or values by index label.
- The `Series` constructor uses an uppercase `S`.
-  to View some exercices and some other implementations non existant in the "master" branch you can visit revision branch via:
--------------------- `git switch revision` ----------------------- 
---
## Day 2 of Learning pandas
***Date:*** July 12 2026
# What i Learned 
## WHAT IS THE DIFFERENCE BTWEEN `.loc` and `.iloc` ?
- `.loc[]` selects by label(index_name),While `.iloc[]` selects by integer position(0,1,2,3,..........) regardless of what labels are.
- `Data[Data<200]` filters the Data Series and only gardes the elements that satisfy the condition between [...]
- `Data.index=[.......]` is the way to acces to the Series indexes (lables)
- A Python `dict` stores `key: value` pairs, where keys must be hashable (e.g. `str`, `int`, `tuple`) but not necessarily strings, and passing a `dict` into `pd.Series()` automatically uses its keys as the index.
### code
![code-day-2](assets/pandas/ray-so-export.png)
### Example_output :
![Execution-day-2](assets/pandas/EXECUTION_DAY2%20_PANDAS.png)
### Notes / Key takeaways:
- `pd.Series(data, index=[...])` lets you assign custom labels to each item instead of relying on default 0,1,2 positions.
- `.loc[label]` selects an item by its **index label** (e.g. `Data.loc["index_1"]`).
- `.iloc[position]` selects an item by its **integer position**, regardless of what the labels are (position always starts at 0).
- You can **reassign the index** of an existing Series anytime with `Data.index = [...]`, as long as the new list has the same length as the Series.
- `Data[condition]` filters the Series, keeping only the items that satisfy the condition inside `[...]` (e.g. `Data[Data != "B"]`).
- Passing a `dict` into `pd.Series()` automatically uses the dict's **keys as the index** and its **values as the data** — no need to set `index=[...]` manually.
- `.loc[]`, `.iloc[]`, and filtering with `[...]` all work the same way whether the Series was built from a list or a dict.
## Day 3 of learning pandas
