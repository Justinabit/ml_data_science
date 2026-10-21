# Depression Prediction – Group Data Science Project

## Project goal
Build a machine learning model that predicts whether a student is depressed
(`Depression` = 1) or not (`Depression` = 0) using the survey dataset `depression.csv`.

## Files
| File | Purpose |
|------|---------|
| `depression.csv` | Dataset (27,901 rows, 18 columns) |
| `datasci.py` | Our original notebook/script (needs fixing) |
| `datasci_fixed.py` | Reference version of the fixes. Read it for ideas, but do the corrections yourselves and understand each one |
| `README.md` | This guide |

## Course requirements (must all be met)
1. Build a learning model based on our assigned dataset.
2. Perform basic exploratory data analysis (EDA), basic data cleaning and feature preparation.
3. Train with an **80-20 split** using **`random_state=42`** (also use 42 for any model that takes a random_state).
4. Use a **confusion matrix** to evaluate the model.
5. Submit **complete documentation** and the **Python code** showing the ML process.

## Main problem in the original code
`Depression` is a **0/1 label**, so this is a **classification** task. The original code used
regression models (`LinearRegression`, `RandomForestRegressor`) and regression metrics (MSE, R²).
That does not fit the problem and must be changed.

## Tasks to fix (checklist)

### A. Setup
- [ ] Put `depression.csv` in the same folder as the script and load it with a **relative path**: `pd.read_csv('depression.csv')` (not `'/depression.csv'`).

### B. Exploratory data analysis (requirement 2)
- [ ] Show `df.shape`, `df.info()`, `df.describe()`.
- [ ] Check missing values (`df.isna().sum()`) and duplicate rows (`df.duplicated().sum()`).
- [ ] Check class balance of `Depression` (`value_counts`) and plot it.
- [ ] Plot histograms of the numeric columns and bar charts of key categorical columns.
- [ ] Look at the unique values of `Gender`, `Sleep Duration`, `Dietary Habits`. Note odd categories such as "Others".
- [ ] Write 3–5 short observations from the EDA for the documentation.

### C. Data cleaning and feature preparation (requirement 2)
- [ ] Drop the `id` column (it is only an identifier).
- [ ] Handle missing values. Report how many rows were dropped, or justify any imputation.
- [ ] Remove the leftover `fillna(0)` step; it does nothing after `dropna()`.
- [ ] Decide what to do with high-cardinality columns (`City`, `Degree`) and near-constant columns (`Work Pressure`, `Job Satisfaction`). Explain your decision.
- [ ] Encode categorical columns (one-hot) and scale numeric columns **using a Pipeline / ColumnTransformer**, so preprocessing is fitted on training data only.
- [ ] Discuss whether to keep `Have you ever had suicidal thoughts ?`. It is very close to a symptom of the target. (Optional: compare results with and without it.)

### D. Train/test split (requirement 3)
- [ ] `train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)`
- [ ] Use `random_state=42` in **every** model that has that parameter.

### E. Models
- [ ] Replace `LinearRegression` with `LogisticRegression(max_iter=1000)`.
- [ ] Replace `RandomForestRegressor(max_depth=2)` with `RandomForestClassifier` and a larger depth (e.g. 8–15).
- [ ] (Optional) Tune hyperparameters with `GridSearchCV` or `RandomizedSearchCV`.
- [ ] (Optional) Run 5-fold cross-validation for a more reliable score.

### F. Evaluation (requirement 4)
- [ ] Plot a **confusion matrix** for each model on the **test set** (`ConfusionMatrixDisplay`).
- [ ] Report accuracy, precision, recall, F1 and ROC-AUC for train and test sets.
- [ ] Compare train vs. test scores to check for overfitting.
- [ ] Explain which model is better and why. Recall matters for depression screening, because missing a depressed student is the costly mistake.

### G. Visualization fixes
- [ ] **Do not reuse `x` and `y` as variable names** in the plotting cell. The original code overwrote the feature matrix and the target.
- [ ] Remove the leftover "Experimental LogS / Predict LogS" scatter plot. It was copied from another tutorial and does not apply here.
- [ ] Add a feature-importance chart from the random forest (top 15 features).
- [ ] Call `plt.show()` after each plot.

### H. Documentation (requirement 5)
Write a report that includes:
1. Introduction and objective
2. Dataset description (rows, columns, target, class balance)
3. EDA findings (with charts)
4. Data cleaning and feature preparation steps, with justification
5. Models used and why
6. Results: metrics table and confusion matrices
7. Discussion: best model, limitations (e.g. self-reported survey data), and ethical note (this is a student project, **not** a clinical diagnostic tool)
8. Conclusion
9. Member contributions (3 members: lead/documentation, code part 1, code part 2)

## Work split (3 members)
| Member | Role | Tasks |
|--------|------|-------|
| Member 1 (project lead) | Coordination, documentation, review | Section H (write the report), compare the teammates' final code against the reference version, make sure all requirements are met |
| Member 2 | Code fixer, part 1 | Sections A–C (load data, EDA, cleaning, feature preparation) and Section D (80-20 split with `random_state=42`) |
| Member 3 | Code fixer, part 2 | Sections E–G (models, confusion matrix and metrics, visualization fixes) |

Members 2 and 3 work in the **same script**. Member 2 finishes Sections A–D first
and shares the file. Member 3 then builds on it. Agree on the variable names
(`X_train`, `X_test`, `y_train`, `y_test`) before starting.

Member 1 collects the EDA findings, results and discussion from Members 2 and 3
for the documentation. Members 2 and 3 should send short notes on what they
changed and why, as the report needs this.

## Rules for the team
- Everyone uses the same dataset file and `random_state=42` so results match.
- Make sure the final script runs from top to bottom with no errors (Colab: *Runtime → Restart and run all*).
- Comment your code and explain *why* you made each choice, not just what the code does.
- Send the final fixed script to Member 1, who will review it before submission.
- Submit: the final `.py` (or `.ipynb`) file and the documentation.
