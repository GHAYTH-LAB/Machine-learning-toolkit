# Machine Learning Toolkit

> **“Grind every day—small, consistent steps become the projects you once thought were out of reach.”**

This repository is my personal **30-day challenge to learn machine learning and deep learning with Python**. It captures how far I had come as of **6 August 2026**: hands-on tutorials, daily practice files, feature-engineering experiments, neural-network work, and selected code used for Kaggle submissions.

The goal is to learn by building, testing ideas, and improving one day at a time. I hope to create as many projects as I can, turning each new concept into something practical.

## Project structure

```text
Machine learning toolkit/
│
├── numpy/                 # NumPy practice and fundamentals
├── pandas/                # Data manipulation and analysis practice
├── seaborn/               # Three-day visualisation practice course
│   ├── day1.py            # Categorical plots and themes
│   ├── day2.py            # Numeric relationships and regression plots
│   ├── day3.py            # Pair plots and correlation heatmaps
│   └── Readme.md          # Seaborn learning notes
│
├── scikit_learn/          # Classical machine-learning tutorials and projects
│   ├── day*.py            # Daily scikit-learn practice
│   ├── solution *.py      # Kaggle solution code
│   └── README.md
│
├── pytorch/               # Deep-learning practice with PyTorch
│   ├── day1.py ... day12.py
│   └── README.md          # Day-by-day PyTorch learning journey
│
├── datasets/              # Local datasets used by the scripts (Git-ignored)
├── assets/                # Project assets
├── catboost_info/         # CatBoost training output
├── random_forest_visualisation.py
├── .gitignore             # Excludes datasets and generated submissions
└── README.md              # This project overview
```

## What this repository includes

- Daily Python files that document my learning path.
- Practice with NumPy, pandas, Seaborn, scikit-learn, and PyTorch.
- Data cleaning, preprocessing, visualisation, feature engineering, classification, evaluation, and neural networks.
- Python source code for selected Kaggle submissions and tutorials.

## What I have learned

### NumPy

- Creating arrays and inspecting their shape.
- Indexing, slicing, reshaping, copying, reversing, and iterating over arrays.
- Applying vectorised arithmetic instead of Python loops when working with numeric data.
- Filtering with boolean masks, searching with `np.where()`, and sorting arrays.

### pandas

- Creating and selecting data with `Series` and `DataFrame` objects.
- Using `.loc` for label-based selection and `.iloc` for position-based selection.
- Loading CSV files, inspecting data, cleaning column names, handling missing values, and removing duplicates.
- Filtering rows, creating features, working with categorical columns, and preparing tabular data for models.

### Matplotlib

- Creating figures and subplots with `plt.subplots()`.
- Setting readable chart titles, axis labels, figure sizes, and layouts.
- Combining Matplotlib layout tools with Seaborn plots and displaying charts with `plt.show()`.

### Seaborn

- Setting visual themes with `sns.set_theme()`.
- Exploring categories and distributions with count plots and box plots.
- Visualising numeric relationships with scatter plots and regression lines.
- Exploring several variables with pair plots and interpreting correlations with annotated heatmaps.

### scikit-learn

- Splitting data into training and test sets with `train_test_split`.
- Scaling numeric features and encoding categorical features with tools such as `StandardScaler`, `MinMaxScaler`, `QuantileTransformer`, and `OneHotEncoder`.
- Training regression and classification models, including linear/logistic regression, K-nearest neighbours, random forests, and voting ensembles.
- Tuning models with `GridSearchCV` and validating classification work with stratified folds.
- Evaluating models with accuracy, precision, recall, F1 score, RMSE, R², and log loss.

### PyTorch

- Creating tensors, checking their shape, type, and device, and selecting values with tensor operations.
- Building fully connected neural networks by subclassing `nn.Module` and using linear layers with ReLU, Leaky ReLU, and tanh activations.
- Training with `TensorDataset`, `DataLoader`, Adam, gradient backpropagation, and mini-batches.
- Handling binary and multiclass tasks with the appropriate target types, losses, and prediction rules.
- Using CPU/GPU devices, evaluating with `model.eval()` and `torch.no_grad()`, and preparing neural-network Kaggle submissions.

## Learning approach

Each folder follows a learn-by-doing approach: study a concept, write code, train a model or create a visualisation, then record the next step. The README files inside the `seaborn/` and `pytorch/` folders explain the progress day by day without repeating previous lessons.

## Note on data and submissions

Datasets and generated Kaggle submission CSV files are intentionally excluded from version control. The source code remains here to show the workflow and the ideas behind each experiment.

---

*This is only the beginning—more experiments, projects, and lessons are on the way.*
