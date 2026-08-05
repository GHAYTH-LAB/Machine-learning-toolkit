"""Day 1: Seaborn foundations and categorical distributions."""
import matplotlib.pyplot as plt
import seaborn as sns
tips = sns.load_dataset("tips")
sns.set_theme(style="whitegrid", context="notebook", palette="deep")
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
# Count observations in each category.
sns.countplot(data=tips, x="day", hue="sex", ax=axes[0])
axes[0].set(title="Restaurant visits by day", xlabel="Day", ylabel="Number of bills")
sns.boxplot(data=tips, x="day", y="total_bill", hue="sex", ax=axes[1])
axes[1].set(title="Bill distribution by day", xlabel="Day", ylabel="Total bill ($)")
fig.tight_layout()
plt.show()
