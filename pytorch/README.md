# PyTorch Learning Journey

This folder records a day-by-day progression through PyTorch. Each entry focuses on what was added on that day, so concepts introduced earlier are not repeated in later sections.

## Day 1 — First end-to-end multiclass project

Built a classifier for the Ghouls, Goblins, and Ghosts dataset.

- Loaded CSV data with pandas and normalized column names.
- Created threshold-based and interaction features from the measurements.
- Transformed numeric features with `QuantileTransformer` and categorical features with `OneHotEncoder`.
- Converted NumPy arrays to `float32` feature tensors and integer (`long`) class-label tensors.
- Packaged training data with `TensorDataset` and `DataLoader`.
- Defined a fully connected `nn.Module` using `nn.Linear` layers and ReLU activations.
- Trained with the PyTorch loop: `zero_grad()`, forward pass, loss, `backward()`, and `step()`.
- Used `CrossEntropyLoss` and `argmax` to predict one of three classes.
- Switched between `train()` and `eval()` modes and created a Kaggle submission.

## Day 2 — Tensor inspection and indexing

Focused on basic tensor operations rather than modelling.

- Created tensors from a pandas DataFrame and NumPy data.
- Inspected tensor device, shape, data type, values, and means along a chosen dimension.
- Created random tensors matching an existing tensor with `torch.randint_like`.
- Selected position-specific values with `torch.gather`.
- Checked whether CUDA is available.

## Day 3 — Preparing a binary-classification dataset

Started working with scikit-learn’s breast-cancer dataset.

- Loaded features and labels directly with `load_breast_cancer`.
- Split the data into training and test sets with a fixed random seed.
- Fit `StandardScaler` on the training features and applied it to the test features.

## Day 4 — First binary neural network

Completed a binary classifier on the breast-cancer data.

- Shaped binary targets as one-column tensors with `unsqueeze(1)`.
- Used a sigmoid output layer, `BCELoss`, and a 0.5 decision threshold.
- Mixed ReLU and Leaky ReLU hidden activations.
- Calculated test loss and accuracy directly from tensor comparisons.
- Achieved approximately 97.4% test accuracy.

## Day 5 — Deeper binary network and min–max scaling

Applied a larger architecture to `numerical_dataset.csv`.

- Replaced standardization with `MinMaxScaler` to map input values to a shared range.
- Experimented with a six-layer, widening-then-narrowing network.
- Trained for many epochs while tracking mean batch loss.

## Day 6 — Architecture simplification experiment

Revisited the same numerical classification problem with a smaller network.

- Compared a compact four-layer architecture with the deeper design from Day 5.
- Changed preprocessing back to `StandardScaler`.
- Adjusted the Adam learning rate to compare training behaviour.

## Day 7 — Multiclass forest-cover classification with validation

Built a seven-class model for the Forest Cover Type dataset.

- Engineered domain-inspired features for elevation, slope, hydrology, sunlight, roads, and fire points.
- Split the labelled data again to obtain a validation set before preparing the Kaggle test set.
- Mapped cover types from 1–7 to PyTorch’s 0–6 class indices, then mapped predictions back for submission.
- Moved tensors and the model to a dynamically selected CPU/GPU device.
- Measured validation loss and multiclass accuracy.

## Day 8 — Logits and training accuracy

Improved the forest-cover training workflow.

- Used `MinMaxScaler` for the engineered forest features.
- Kept the final layer as raw logits, which is the correct input form for `CrossEntropyLoss`.
- Computed training accuracy incrementally from each mini-batch.
- Reduced console noise by reporting loss and accuracy every ten epochs.
- Produced a neural-network Kaggle submission using the competition column names.

## Day 9 — Mixed-type employee attrition data

Applied binary classification to a realistic employee dataset.

- Automatically separated numeric and categorical columns with `select_dtypes`.
- Combined standardized numeric features and one-hot encoded categories with `np.hstack`.
- Encoded text attrition labels as numeric targets.
- Evaluated the model with scikit-learn’s `accuracy_score`.

## Day 10 — Numerically stable binary loss

Refined attrition modelling by separating logits from probabilities.

- Replaced a sigmoid output plus `BCELoss` with raw outputs plus `BCEWithLogitsLoss`.
- Applied `torch.sigmoid` only after inference to obtain probabilities and thresholded labels.
- Added weighted F1 score to assess performance beyond accuracy.
- Began importing `skorch` as a bridge between PyTorch models and scikit-learn-style workflows.

## Day 11 — Loan approval classification on GPU

Transferred the mixed-data binary workflow to loan approval prediction.

- Used `QuantileTransformer` to handle non-uniform numeric distributions.
- Moved both feature/target tensors and the model to the selected device.
- Reported weighted F1 alongside accuracy and loss after moving evaluation tensors back to CPU for scikit-learn.

## Day 12 — Multiclass prediction from housing data

Used housing attributes to classify `ocean proximity` categories.

- Checked duplicate rows and missing values before training.
- Imputed missing `total bedrooms` values with the median.
- Treated a categorical column as the multiclass target and encoded it as class indices.
- Explored a new hidden activation (`tanh`) alongside ReLU and Leaky ReLU.
- Evaluated multiclass output with accuracy and weighted precision.

## Key implementation notes

- `CrossEntropyLoss` expects raw logits and integer class indices; do not add a final softmax layer before this loss.
- `BCEWithLogitsLoss` expects raw logits; use `torch.sigmoid` only when converting outputs into probabilities.
- Fit preprocessors on training data only, then use `transform` for validation, test, and submission data.
- Keep model and tensors on the same device, and use `model.eval()` with `torch.no_grad()` for inference.
