Electric Vehicle Purchase Interest — Kaggle Competition
Introduction

This notebook is my submission for the Kaggle Playground Series — Season 6, Episode 9 competition: Predicting Electric Vehicle Interest.

The goal is to predict, for each customer, the probability that they are interested in buying an electric vehicle (Will_Buy_EV: Yes/No), based on demographic data (age, gender, income), travel habits (daily commute, current car type), and EV-adoption factors (home charging access, subsidy availability, range anxiety level, proximity to charging stations, environmental concern, etc.). This is a binary classification problem, evaluated on AUC-ROC.

Notebook outline
Data loading and overview
Importing libraries (pandas, numpy, matplotlib, seaborn, scikit-learn, XGBoost)
Loading the training dataset and a first look at it (.info())
Exploratory Data Analysis (EDA)
Checking missing values (none found) and duplicate rows (none found)
Distribution of Age and Annual_Income_USD, including a check on customers with income stuck at exactly $30,000
Breakdown of Current_Car_Type, City_Type, Range_Anxiety_Level, Number_of_Cars_Owned, Gender
Correlation heatmap of the numerical features
Feature preparation
Manual encoding of Home_Charging_Possible and Subsidy_Available (Yes/No → 1/0)
Manual ordinal encoding of Range_Anxiety_Level (Low/Medium/High → 1/2/3), to preserve the natural order of the variable
Splitting features (X) from the target (Will_Buy_EV, encoded Yes/No → 1/0)
Preprocessing and model training
Train/validation split (train_test_split)
One-hot encoding of the remaining categorical columns (Gender, City_Type, Current_Car_Type) via ColumnTransformer, with remainder='passthrough' to keep every other (already numeric/mapped) column untouched
Model: XGBClassifier (n_estimators=600, learning_rate=0.03, max_depth=5, subsample=0.8, colsample_bytree=0.8, min_child_weight=5, reg_lambda=1.0), wrapped together with the preprocessing step in a single Pipeline
Fitting the pipeline and checking validation performance
Test data preprocessing and submission
Loading the test dataset and applying the same manual encoding used on the training data
Generating prediction probabilities with predict_proba(X_test)[:, 1]
Exporting the submission.csv file in the format expected by Kaggle
Conclusion

Final leaderboard score: AUC ≈ 0.941.

Two mistakes came up while building this notebook, both worth noting since they're easy to miss and can silently wreck a model:

ColumnTransformer without remainder='passthrough': by default (remainder='drop'), any column not explicitly listed in transformers gets silently dropped before reaching the model — including numerical features that turned out to carry most of the signal. This first version scored an AUC of only ~0.53 on validation (barely above random) while still showing a "normal-looking" ~82% accuracy, which is what made the bug hard to notice at first.
Computing AUC on predict() instead of predict_proba(): roc_auc_score needs continuous probabilities to properly measure a model's ranking ability. Computing it on already-thresholded class predictions (predict()) understates the real score substantially (~0.81 vs. the actual ~0.94 measured with probabilities).

Model selection here was limited to a single train/validation split with manually chosen XGBoost hyperparameters — no systematic hyperparameter search (e.g. GridSearchCV/Optuna) and no k-fold cross-validation were used, so the reported validation score should be read as a single-split estimate rather than a fully robust one. Both would be natural next steps, along with trying alternative models (LightGBM, CatBoost) for comparison.
