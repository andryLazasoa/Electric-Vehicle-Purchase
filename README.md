# Spaceship Titanic — Kaggle Competition

## Introduction

This notebook is my submission for the Kaggle **[Spaceship Titanic](https://www.kaggle.com/competitions/spaceship-titanic)** competition.

The (fictional) context set up by the competition: in the year 2912, the *Spaceship Titanic* was carrying almost 13,000 passengers toward three newly habitable exoplanets. While crossing a spacetime anomaly hidden in a dust cloud, about half of the passengers were "transported" to another dimension. The goal of the competition is to predict, based on passengers' personal data and onboard spending (home planet, cabin, expenses, etc.), which ones were transported (`Transported`: `True`/`False`). This is a **binary classification** problem, evaluated on prediction accuracy.

## Notebook outline

1. **Data loading and overview**
   - Importing libraries (pandas, numpy, scikit-learn)
   - Loading the training dataset and a first look at it (`.info()`)

2. **Exploratory Data Analysis (EDA)**
   - Checking missing values and duplicates
   - Analysis of `HomePlanet`, `CryoSleep`, `Destination`, `VIP`
   - Analysis of `Age`, `RoomService`, `FoodCourt`, `ShoppingMall`, `Spa`, `VRDeck`
   - Splitting the `Cabin` column into three sub-columns: `Deck`, `Num`, `Side`, followed by analysis of these new columns
   - Summary of findings: which columns have missing values, distributions, planned imputation strategy, and no strong correlation found between numerical variables (so no identified risk of data leakage)

3. **Data preprocessing and model training**
   - Extracting extra information from `PassengerId` (group number)
   - Splitting features / target (`Transported`) and train/validation split
   - Separating categorical and numerical columns
   - Imputing missing values (`SimpleImputer`: `most_frequent` strategy for categorical columns, `median`/`fillna(0)` for numerical columns depending on the column)
   - Encoding categorical variables with `OneHotEncoder`
   - Merging all transformed columns into final training and validation sets
   - Comparing two models (`RandomForestClassifier` and `GradientBoostingClassifier`) via 5-fold cross-validation across several `n_estimators` values, with performance curves plotted
   - Selecting the best model

4. **Test data preprocessing and submission**
   - Loading the test dataset and applying the same transformations used on the training data (via a `Pipeline` combining imputation, encoding, and the model)
   - Fitting the final pipeline on the full training data
   - Generating predictions on the test set
   - Exporting the `submission.csv` file in the format expected by Kaggle

## Conclusion

The two models tested via cross-validation gave the following results (mean accuracy across 5 folds):

- **RandomForestClassifier**: best accuracy around **0.796** (`n_estimators=400`)
- **GradientBoostingClassifier**: best accuracy around **0.806** (`n_estimators=300`)

The `GradientBoostingClassifier` (`n_estimators=300`) was chosen as the final model and trained on the full dataset to generate the predictions submitted to the competition.

Model selection here was limited to comparing `n_estimators` for both algorithms via cross-validation; the models were **not further hyperparameter-tuned** (e.g. `learning_rate`, `max_depth`, `min_samples_split`, etc.) beyond that, mainly because I didn't yet have enough knowledge and understanding of how these parameters interact to tune them properly.

This is my **very first Kaggle competition**, and more broadly my **first Machine Learning experience outside of a class exercise**. The notebook reflects that: it isn't always very polished in form (organization, naming, comments), and there is certainly a lot of room for improvement — both in feature engineering (e.g. making better use of the `Deck`/`Num`/`Side` columns or onboard spending), and in properly tuning the models or trying other algorithms. I'm publishing it as is, as a record of this starting point.
