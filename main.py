# Depression prediction - supervised classification
# Target  : Depression (1 = depressed, 0 = not depressed)
# Features: the other survey answers (age, gender, city, degree, academic pressure,
#           CGPA, study satisfaction, sleep, diet, suicidal thoughts,
#           work/study hours, financial stress, family history of mental illness)

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, ConfusionMatrixDisplay)

# ---------- 1. Load data ----------
df = pd.read_csv('depression.csv')   # relative path: keep the CSV next to this script
df = df.rename(columns={'Have you ever had suicidal thoughts ?': 'Suicidal Thoughts'})

# ---------- 2. Exploratory data analysis ----------
print(df.shape)
df.info()
print(df.describe().round(2).T.to_string())   # .T turns the table sideways so it fits
print("Missing values:\n", df.isna().sum())
print("Duplicate rows:", df.duplicated().sum())
print(df['Depression'].value_counts(normalize=True))   # class balance

df['Depression'].value_counts().plot(kind='bar', title='Depression (0 = no, 1 = yes)')
plt.ylabel('Number of students')
plt.show()

df.drop(columns=['id', 'Depression']).hist(figsize=(12, 8))
plt.suptitle('Distribution of each numeric column (y-axis = number of students)')
plt.tight_layout()
plt.show()

# Look for odd or useless values
for col in ['Gender', 'Profession', 'Sleep Duration', 'Dietary Habits',
            'Work Pressure', 'Job Satisfaction', 'City']:
    print(df[col].value_counts(), "\n")

# % of depressed students in each group of the key columns
for col in ['Academic Pressure', 'Financial Stress', 'Sleep Duration', 'Suicidal Thoughts']:
    rate = df.groupby(col)['Depression'].mean() * 100
    print(rate.round(1), "\n")
    rate.plot(kind='bar', title=f'% depressed by {col}', ylim=(0, 100))
    plt.ylabel('% depressed')
    plt.show()

# EDA findings:
# - 58.5% of students are depressed, 41.5% are not (mildly imbalanced).
# - Only 3 missing values (Financial Stress) and no duplicate rows.
# - Suicidal thoughts: 79% depressed vs. 23% without.
# - The % depressed rises steadily with Academic Pressure and Financial Stress.
# - Profession, Work Pressure and Job Satisfaction have almost one value only
#   (nearly everyone is a student), and some City values are names or typos.

# ---------- 3. Data cleaning ----------
# Every step prints how many rows are left, so each deletion can be traced
print("Rows before cleaning:", len(df))

df = df.drop(columns=['id'])   # only a row number, not a feature

df = df.dropna()               # 3 rows with missing Financial Stress
print("After removing missing values:", len(df))

# Cities that appear fewer than 100 times are typos or names (e.g. 'Saanvi', '3')
city_counts = df['City'].value_counts()
valid_cities = city_counts[city_counts >= 100].index
df = df[df['City'].isin(valid_cities)]
print("After removing invalid cities:", len(df))

# 'Others' is an unclear answer for sleep and diet (only 30 rows)
df = df[df['Sleep Duration'] != 'Others']
df = df[df['Dietary Habits'] != 'Others']
print("After removing 'Others' answers:", len(df))

# One value fills more than 99% of these columns, so they cannot help the model
df = df.drop(columns=['Profession', 'Work Pressure', 'Job Satisfaction'])

print("Shape after cleaning:", df.shape)
print("Missing values left:", df.isna().sum().sum())

# ---------- 4. Features (X), target (y) and 80-20 split ----------
X = df.drop(columns=['Depression'])
y = df['Depression']

# stratify=y keeps the same depressed / not-depressed ratio in train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

num_cols = X.select_dtypes(include='number').columns.tolist()
cat_cols = X.select_dtypes(exclude='number').columns.tolist()
print("Numeric features:", num_cols)
print("Categorical features:", cat_cols)

# ---------- 5. Feature check (training data only) ----------
# Numeric features: correlation with Depression
corr = X_train[num_cols].corrwith(y_train).sort_values()
print(corr.round(3))
corr.plot(kind='barh', title='Correlation with Depression (training data)')
plt.xlabel('Pearson correlation')
plt.show()

# Categorical features: % depressed in each group
for col in cat_cols:
    print((y_train.groupby(X_train[col]).mean() * 100).round(1), "\n")

# ---------- 6. Train and evaluate ----------
models = {
    'Baseline (most frequent)': DummyClassifier(strategy='most_frequent'),
    'Logistic regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random forest': RandomForestClassifier(n_estimators=300, max_depth=10, random_state=42),
}

results = []
fitted = {}   # keep the trained pipelines for the error analysis
for name, model in models.items():
    # One-hot encode text columns and scale numeric columns. Inside a Pipeline
    # these steps are fitted on the training data only, so no test data leaks in.
    preprocess = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols),
        ('num', StandardScaler(), num_cols),
    ])
    pipe = Pipeline([('prep', preprocess), ('model', model)])
    pipe.fit(X_train, y_train)
    fitted[name] = pipe

    train_pred = pipe.predict(X_train)
    test_pred = pipe.predict(X_test)

    results.append({
        'Model': name,
        'Train accuracy': accuracy_score(y_train, train_pred),   # compare with test accuracy
        'Test accuracy': accuracy_score(y_test, test_pred),      # to check for overfitting
        'Precision': precision_score(y_test, test_pred, zero_division=0),
        'Recall': recall_score(y_test, test_pred),
        'F1': f1_score(y_test, test_pred),
    })

    print(f"\n{name} - confusion matrix (test data):")
    print(confusion_matrix(y_test, test_pred))
    ConfusionMatrixDisplay.from_predictions(
        y_test, test_pred, display_labels=['Not depressed', 'Depressed'], cmap='Blues')
    plt.title(name)
    plt.show()

print(pd.DataFrame(results).round(3).to_string())

# ---------- 7. Error analysis (logistic regression, test data) ----------
# Which depressed students does the model miss?
test_data = X_test.copy()
test_data['Actual'] = y_test
test_data['Predicted'] = fitted['Logistic regression'].predict(X_test)

caught = test_data[(test_data['Actual'] == 1) & (test_data['Predicted'] == 1)]
missed = test_data[(test_data['Actual'] == 1) & (test_data['Predicted'] == 0)]
false_alarms = test_data[(test_data['Actual'] == 0) & (test_data['Predicted'] == 1)]

for label, group in [('Caught (TP)', caught), ('Missed (FN)', missed),
                     ('False alarm (FP)', false_alarms)]:
    print(f"{label}: {len(group)} students | "
          f"avg Academic Pressure {group['Academic Pressure'].mean():.2f} | "
          f"avg Financial Stress {group['Financial Stress'].mean():.2f} | "
          f"suicidal thoughts {(group['Suicidal Thoughts'] == 'Yes').mean() * 100:.0f}%")

# ---------- 8. Limitation check: model without 'Suicidal Thoughts' ----------
# This question is very close to a symptom of depression. Does the model still
# work without it?
cat_cols_no_st = [col for col in cat_cols if col != 'Suicidal Thoughts']
preprocess = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols_no_st),
    ('num', StandardScaler(), num_cols),
])
pipe = Pipeline([('prep', preprocess),
                 ('model', LogisticRegression(max_iter=1000, random_state=42))])
pipe.fit(X_train, y_train)
test_pred = pipe.predict(X_test)

print("\nLogistic regression WITHOUT Suicidal Thoughts")
print("Test accuracy:", round(accuracy_score(y_test, test_pred), 3))
print("Recall:", round(recall_score(y_test, test_pred), 3))
print(confusion_matrix(y_test, test_pred))
