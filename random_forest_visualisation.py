#this visualisation is taken from kaggle(you need to install first graphiz )
import pandas as pd
import graphviz

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import export_graphviz

# Load a dataset
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train a Random Forest
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Visualize the 11th tree
dot_data = export_graphviz(
    model.estimators_[10],
    out_file=None,
    feature_names=X.columns,
    class_names=["0", "1"],
    filled=True,
    rounded=True,
    special_characters=True,
    impurity=True
)

graph = graphviz.Source(dot_data)
graph.render("tree")   # Saves tree.pdf
graph.view()           # Opens the file