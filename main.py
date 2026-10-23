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
print(df.describe())
print("Missing values:\n", df.isna().sum())
print("Duplicate rows:", df.duplicated().sum())
print(df['Depression'].value_counts(normalize=True))   # class balance

df['Depression'].value_counts().plot(kind='bar', title='Depression (0 = no, 1 = yes)')
plt.ylabel('Number of students')
plt.show()

df.drop(columns=['id', 'Depression']).hist(figsize=(12, 8))
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
print("Rows before cleaning:", len(df))

df = df.drop(columns=['id'])   # only a row number, not a feature
df = df.dropna()               # 3 rows with missing Financial Stress

# Cities that appear fewer than 100 times are typos or names (e.g. 'Saanvi', '3')
city_counts = df['City'].value_counts()
valid_cities = city_counts[city_counts >= 100].index
df = df[df['City'].isin(valid_cities)]

# 'Others' is an unclear answer for sleep and diet (only 30 rows)
df = df[df['Sleep Duration'] != 'Others']
df = df[df['Dietary Habits'] != 'Others']

# One value fills more than 99% of these columns, so they cannot help the model
df = df.drop(columns=['Profession', 'Work Pressure', 'Job Satisfaction'])

print("Rows after cleaning:", len(df))

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

# ---------- 5. Train and evaluate ----------
models = {
    'Baseline (most frequent)': DummyClassifier(strategy='most_frequent'),
    'Logistic regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random forest': RandomForestClassifier(n_estimators=300, max_depth=10, random_state=42),
}

results = []
for name, model in models.items():
    # One-hot encode text columns and scale numeric columns. Inside a Pipeline
    # these steps are fitted on the training data only, so no test data leaks in.
    preprocess = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols),
        ('num', StandardScaler(), num_cols),
    ])
    pipe = Pipeline([('prep', preprocess), ('model', model)])
    pipe.fit(X_train, y_train)

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

# ---------- 6. Limitation check: model without 'Suicidal Thoughts' ----------
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
