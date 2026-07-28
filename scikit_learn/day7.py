import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import OneHotEncoder, QuantileTransformer, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, recall_score

df = pd.read_csv(r"C:\Users\abidli\Desktop\Machine learning toolkit\datasets\train DIGIT RECOGNIZER.csv")
df1 = pd.read_csv(r"C:\Users\abidli\Desktop\Machine learning toolkit\datasets\TEST DIGIT RECOGNIZER.csv")

# --- feature engineering: train ---
new_cols_train = pd.DataFrame({
    "pixel_multiplied1": df["pixel126"] * df["pixel127"] * df["pixel128"],
    "pixel_multilied2": df["pixel155"] * df["pixel156"] * df["pixel326"],
    "pixel_multiplied3": df["pixel236"] * df["pixel235"] * df["pixel290"],
    "pixel_multiplied4": df["pixel353"] * df["pixel354"] * df["pixel381"] * df["pixel382"] * df["pixel380"],
})
df = pd.concat([df, new_cols_train], axis=1)
# --- feature engineering: test ---
new_cols_test = pd.DataFrame({
    "pixel_multiplied1": df1["pixel126"] * df1["pixel127"] * df1["pixel128"],
    "pixel_multilied2": df1["pixel155"] * df1["pixel156"] * df1["pixel326"],
    "pixel_multiplied3": df1["pixel236"] * df1["pixel235"] * df1["pixel290"],
    "pixel_multiplied4": df1["pixel353"] * df1["pixel354"] * df1["pixel381"] * df1["pixel382"] * df1["pixel380"],
})
df1 = pd.concat([df1, new_cols_test], axis=1)
scaler = StandardScaler()
y_train = df["label"]
X_train = df.drop(columns="label")
X_test = df1
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
voting = VotingClassifier(
    estimators=[
        ("rf", RandomForestClassifier(random_state=42)),
        ("knn", KNeighborsClassifier()),
        ("sv", SVC())
    ]
)
Grid = GridSearchCV(
    estimator=voting,
    param_grid={
        "rf__n_estimators": [300, 350, 400],
        "rf__max_depth": [None, 5],
        "rf__min_samples_leaf": [5, 10, 15],
        "rf__min_samples_split": [5, 10],
        "knn__n_neighbors": [5, 7],
        "sv__C": [1, 10, 15]
    },
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)
Grid.fit(X_train, y_train)
predictions = Grid.predict(X_test)
print(Grid.best_params_)
submission = pd.DataFrame({
    "ImageId": range(1, len(predictions) + 1),
    "Label": predictions
})
submission.to_csv("submission_Digit_Recongnizer.csv", index=False)