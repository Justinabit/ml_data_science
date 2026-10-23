# Depression Prediction – Group Data Science Project

Predict whether a student is depressed (`Depression` = 1) or not (`Depression` = 0)
from the survey answers in `depression.csv`. Because the target is a 0/1 label, this is a
**supervised binary classification** task.

> This is a student project, **not** a clinical diagnostic tool.

## Files
| File | Purpose |
|------|---------|
| `depression.csv` | Dataset: 27,901 student survey rows, 18 columns |
| `main.py` | The full ML process: EDA → cleaning → 80-20 split → pipeline → models → confusion matrix |
| `requirements.txt` | Python packages needed to run the script |

## How to run
```bash
pip install -r requirements.txt
python main.py          # run from this folder so the relative path to depression.csv works
```
In Google Colab, upload `depression.csv` and paste `main.py` into a cell (or run `!python main.py`).

## How the code meets the requirements
| Requirement | Where in `main.py` |
|-------------|--------------------|
| 1. Build a learning model on the assigned dataset | Section 5: logistic regression and random forest classifiers |
| 2. EDA, basic cleaning, feature preparation | Sections 2–3 (EDA, cleaning) and the `Pipeline` in section 5 (encoding and scaling) |
| 3. 80-20 split, `random_state=42` | Section 4: `train_test_split(test_size=0.2, random_state=42, stratify=y)`; every model also uses `random_state=42` |
| 4. Confusion matrix evaluation | Section 5: a confusion matrix for each model, plus accuracy, precision, recall and F1 |
| 5. Code and documentation | This repository and the group report |

**Features (13):** Age, Gender, City, Degree, Academic Pressure, CGPA, Study Satisfaction,
Sleep Duration, Dietary Habits, Suicidal Thoughts, Work/Study Hours, Financial Stress,
Family History of Mental Illness. **Target:** Depression.

## EDA findings
- Classes are mildly imbalanced: 58.5% depressed and 41.5% not depressed. No duplicate rows were found, and only 3 values are missing (all in `Financial Stress`).
- Students who reported suicidal thoughts: 79.0% are depressed, compared with 23.2% of those who did not.
- The share of depressed students rises with academic pressure (86.1% at level 5, compared with 19.4% at level 1) and with financial stress (81.3% at level 5, compared with 31.9% at level 1).
- Sleeping less than 5 hours: 64.5% are depressed, compared with 50.9% of those sleeping more than 8 hours.
- `Profession`, `Work Pressure` and `Job Satisfaction` have almost only one value, and some `City` values are names or typos.

## Cleaning and feature preparation
| Step | Rows / columns affected | Why |
|------|-------------------------|-----|
| Drop `id` | 1 column | Row identifier only, no information |
| Drop rows with missing values | 3 rows | Too few to be worth imputing |
| Drop invalid `City` values (fewer than 100 occurrences) | 26 rows | Entries like "Saanvi", "M.Tech" or "3" are typos or names, not cities |
| Drop `Others` in Sleep Duration / Dietary Habits | 30 rows | Unclear answer that cannot be placed in a category |
| Drop `Profession`, `Work Pressure`, `Job Satisfaction` | 3 columns | One value fills more than 99% of rows (almost everyone is a student), so these columns cannot separate the classes |
| One-hot encode categorical columns, scale numeric columns | — | Done inside a `Pipeline`, so the encoder and scaler are **fitted on the training set only** (no test-data leakage) |

That leaves 27,842 rows, split into 22,273 training rows and 5,569 test rows.

## Results (test set)
| Model | Train accuracy | Test accuracy | Precision | Recall | F1 |
|-------|---------------:|--------------:|----------:|-------:|---:|
| Baseline (always "depressed") | 0.586 | 0.586 | 0.586 | 1.000 | 0.739 |
| **Logistic regression** | 0.849 | **0.851** | **0.862** | 0.889 | **0.875** |
| Random forest (max_depth=10) | 0.854 | 0.832 | 0.835 | 0.889 | 0.861 |

Confusion matrices on the test set (rows = true label, columns = predicted label):

| Model | True negatives | False positives | False negatives | True positives |
|-------|---------------:|----------------:|----------------:|---------------:|
| Baseline | 0 | 2,308 | 0 | 3,261 |
| Logistic regression | 1,842 | 466 | 362 | 2,899 |
| Random forest | 1,734 | 574 | 363 | 2,898 |

- Both models clearly beat the baseline, which simply predicts the majority class.
- **Logistic regression is the chosen model.** It has the best test accuracy, precision and F1, and its train and test accuracy are almost equal (0.849 vs. 0.851), so it does not overfit. The random forest has the same recall but more false alarms, and it does better on training data than on test data (0.854 vs. 0.832), which is a sign of mild overfitting.
- Recall matters most here, because missing a depressed student (a false negative) is the costly mistake. Logistic regression misses 362 of the 3,261 depressed students in the test set.

## Limitations
- **Suicidal thoughts are close to a symptom of depression.** Without this feature, logistic regression drops to 0.800 accuracy and 0.851 recall. The model still works, but much of its strength comes from that one question.
- The data is self-reported survey data and may contain honest mistakes or biased answers. Respondents are almost all students in Indian cities, so the results may not generalise to other groups.
- The data has no time dimension, so the model shows associations, not causes.
- The model uses the default 0.5 decision threshold. A screening tool would probably lower it to catch more depressed students, at the cost of more false alarms.

## Report outline (requirement 5)
1. Introduction and objective
2. Dataset description
3. EDA findings with charts
4. Cleaning and feature preparation, with justification
5. Models used and why
6. Results: metrics table and confusion matrices
7. Discussion: best model, limitations, ethical note
8. Conclusion
9. Member contributions
