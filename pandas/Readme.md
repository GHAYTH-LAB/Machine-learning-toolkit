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

---

## Day 3 of learning pandas:
***Date:*** July 13 2026
### What I learned:
-a `DataFrame` is a two dimentional labled data structure
-`pd.DataFrame` creates a DataFrame structure 
-we can acces to a line by `Data.loc(Label_Name)` or by `Data.iloc(index)`
-we can add `a line` by creating a new line by creating a new `DataFrame` and then use `pd.concat([old DataFrame,New DataFrame])`
-we can add a column `Data[.......]=......`
### code:
![My attempt](assets/pandas/Code_day3.png)
### Example_output
![IMPLEMENTATION_Day3](assets/pandas/Implementation_day_3.png)

### Notes / Key takeaways:
- A `DataFrame` is a two-dimensional table made of rows and columns.
- `pd.DataFrame(data)` converts a dictionary into a DataFrame.
- You can rename row labels with `df.index = [...]` to make data access easier.
- New columns are created by assigning a list to a column name, such as `df["married"] = [...]`.
- `.loc["label"]` is used to select a row by its index label.
- To add a new row, create a one-row DataFrame and combine it with the existing one using `pd.concat([df, new_row])`.
- The new row must include the same column names as the original DataFrame, otherwise missing values will appear as `NaN`.

---

## Day 4 of Learning Pandas
***Date:*** July 14 2026
### What I Learned ?
-`pd.read_csv(r"file path")` reads a `csv` file and converts it into a DataFrame same as `pd.read_json()` (it reads a `Json` file )
-- **`index_col`**: Sets a column as the DataFrame's index instead of using the default numeric index.
### code:
![My attempt](assets/pandas/Code_day_4_Learning_Pandas.png)
### Example_output:
![Execution](assets/pandas/Execution_day_4.png)

### Notes/Key Takeaways:
- `pd.read_csv(r"file_path")` reads a CSV file and converts it into a DataFrame.
- `pd.read_json(r"file_path")` reads a JSON file and converts it into a DataFrame.
- **`index_col`** parameter sets a specific column as the DataFrame's index (e.g., `pd.read_csv(..., index_col="City")`).
- **Column Selection:**
  - Single column: `df["column_name"]` returns a Series.
  - Multiple columns: `df[["col1", "col2", "col3"]]` returns a DataFrame with only those columns.
- **Row Selection:**
  - `.loc[index_label]` selects a row by its index label.
  - `.iloc[position]` selects a row by its integer position (0, 1, 2, ...).
  - `.loc[start:end]` selects multiple rows by index label range.
  - `.iloc[start:end]` selects multiple rows by position range.
- **Combined Selection:** `df.loc[row_range, ["col1", "col2"]]` selects specific rows and columns together.
- `.to_string()` converts a DataFrame to a formatted string for better readability when printing.
- You can filter DataFrame columns using the same syntax: `df[["col1", "col2"]]` to get a subset of columns.
