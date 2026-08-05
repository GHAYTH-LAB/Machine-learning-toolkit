
import matplotlib.pyplot as plt
import seaborn as sns
tips = sns.load_dataset("tips")
sns.set_theme(style="ticks", palette="colorblind")
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
sns.scatterplot(
    data=tips,
    x="total_bill",
    y="tip",
    hue="time",
    style="smoker",
    size="size",
    sizes=(30, 180),
    ax=axes[0],
)
axes[0].set(title="Tips versus bill size", xlabel="Total bill ($)", ylabel="Tip ($)")
sns.regplot(
    data=tips,
    x="total_bill",
    y="tip",
    scatter_kws={"alpha": 0.55},
    line_kws={"color": "crimson"},
    ax=axes[1],
)
axes[1].set(title="Linear relationship between bill and tip", xlabel="Total bill ($)", ylabel="Tip ($)")
fig.tight_layout()
plt.show()
