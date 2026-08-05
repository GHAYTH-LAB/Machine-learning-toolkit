# Seaborn: Essential Visualisation Practice

This three-day mini-course covers the core Seaborn workflow: set a theme, pass a tidy pandas DataFrame with `data=...`, map columns to visual properties, and display the chart with Matplotlib.

The examples use Seaborn's built-in datasets. On a machine without a cached copy, `sns.load_dataset()` may need internet access the first time it runs.

## Day 1 — Categorical plots and themes

File: `day1.py`

- Set a reusable chart theme with `sns.set_theme()`.
- Use `countplot` to compare the number of observations in categories.
- Use `boxplot` to show a numeric distribution by category, including median, quartiles, and potential outliers.
- Split a category with `hue` and arrange several figures using Matplotlib subplots.

## Day 2 — Numeric relationships

File: `day2.py`

- Use `scatterplot` when two numeric variables may be related.
- Map additional columns to colour (`hue`), marker (`style`), and point size (`size`) without manually creating groups.
- Use `regplot` to add a linear fit and its confidence interval.
- Adjust plot-specific details through `scatter_kws` and `line_kws`.

## Day 3 — Many variables at once

File: `day3.py`

- Remove missing rows before pairwise plotting with `dropna()`.
- Use `pairplot` to inspect distributions and relationships across several numeric columns, grouped by a class.
- Select numeric columns before calculating a correlation matrix.
- Use `heatmap` with annotations, a centred diverging colour map, and formatted coefficients to interpret correlations.

## Practical rules

- Prefer tidy data: one row per observation and one column per variable.
- Add a clear title and axis labels whenever the default labels are not enough.
- Use `hue` only for a small number of meaningful categories; too many colours make a chart difficult to read.
- A correlation shows association, not causation.
