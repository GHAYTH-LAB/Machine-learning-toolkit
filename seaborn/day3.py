"""Day 3: Multivariable distributions and correlation heatmaps."""
import matplotlib.pyplot as plt
import seaborn as sns
penguins = sns.load_dataset("penguins").dropna()
sns.set_theme(style="white")
pair_grid = sns.pairplot(
    data=penguins,
    vars=["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"],
    hue="species",
    corner=True,
    diag_kind="hist",
)
pair_grid.fig.suptitle("Penguin measurements by species", y=1.02)
plt.show()
numeric_columns = penguins.select_dtypes(include="number")
correlation = numeric_columns.corr()
plt.figure(figsize=(8, 6))
sns.heatmap(
    correlation,
    annot=True,
    fmt=".2f",
    cmap="vlag",
    center=0,
    square=True,
    linewidths=0.5,
)
plt.title("Correlation between penguin measurements")
plt.tight_layout()
plt.show()
