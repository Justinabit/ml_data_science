import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, chi2_contingency

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             ConfusionMatrixDisplay, RocCurveDisplay)

# ---------- 1. Load ----------
df = pd.read_csv('depression.csv')   # use a relative path, not '/depression.csv'
df = df.rename(columns={'Have you ever had suicidal thoughts ?': 'Suicidal Thoughts'})

print(df.shape)
print(df.isna().sum())
print(df['Depression'].value_counts(normalize=True))   # check class balance

# ---------- 1b. Basic EDA ----------
df.info()
print(df.describe())
print("Duplicate rows:", df.duplicated().sum())

df['Depression'].value_counts().sort_index().plot(
    kind='bar', title='Depression class counts (0 = not depressed, 1 = depressed)')
plt.ylabel('Number of students')
plt.show()

df.select_dtypes(include='number').drop(columns=['id']).hist(figsize=(12, 8))
plt.tight_layout()
plt.show()

# % depressed in each group of the key columns
sleep_order = ['Less than 5 hours', '5-6 hours', '7-8 hours', 'More than 8 hours']
fig, axes = plt.subplots(1, 4, figsize=(18, 4))
for ax, col in zip(axes, ['Academic Pressure', 'Financial Stress', 'Sleep Duration', 'Suicidal Thoughts']):
    rate = df.groupby(col)['Depression'].mean() * 100
    if col == 'Sleep Duration':
        rate = rate.reindex(sleep_order)        # natural order, not alphabetical
    rate.plot(kind='bar', ax=ax, title=f'Depression by {col}', rot=20)
    ax.set_ylabel('% depressed')
    ax.set_ylim(0, 100)
plt.tight_layout()
plt.show()

# EDA findings (numbers come from the data)
rate = lambda mask: df.loc[mask, 'Depression'].mean() * 100
print(f"1. Suicidal thoughts: {rate(df['Suicidal Thoughts'] == 'Yes'):.1f}% depressed vs "
      f"{rate(df['Suicidal Thoughts'] == 'No'):.1f}% without")
print(f"2. Academic Pressure 4-5: {rate(df['Academic Pressure'] >= 4):.1f}% depressed vs "
      f"{rate(df['Academic Pressure'] <= 2):.1f}% at 0-2")
print(f"3. Financial Stress 4-5: {rate(df['Financial Stress'] >= 4):.1f}% depressed vs "
      f"{rate(df['Financial Stress'] <= 2):.1f}% at 1-2")
print(f"4. Sleep < 5 hours: {rate(df['Sleep Duration'] == 'Less than 5 hours'):.1f}% depressed vs "
      f"{rate(df['Sleep Duration'] == 'More than 8 hours'):.1f}% for > 8 hours")

# ---------- 2. Clean ----------
# Inspect odd categories (e.g. 'Others' in Sleep Duration / Dietary Habits)
for col in ['Gender', 'Sleep Duration', 'Dietary Habits']:
    print(col, df[col].unique())

# id is only a row number, not a feature
df = df.drop(columns=['id'])

before = len(df)
df = df.dropna()
print(f"Dropped {before - len(df)} rows with missing values")

# City values that appear fewer than 100 times look like typing mistakes
city_counts = df['City'].value_counts()
before = len(df)
df = df[df['City'].isin(city_counts[city_counts >= 100].index)]
print(f"Dropped {before - len(df)} rows with invalid City values")

# Near-constant columns: one value fills more than 99% of rows -> no information
for col in ['Profession', 'Work Pressure', 'Job Satisfaction']:
    print(f"{col}: most common value fills {df[col].value_counts(normalize=True).iloc[0]:.2%} of rows")
df = df.drop(columns=['Profession', 'Work Pressure', 'Job Satisfaction'])
print("Rows left after cleaning:", len(df))

# ---------- 3. Split X / y ----------
y = df['Depression']
X = df.drop(columns=['Depression'])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# ---------- 3b. Feature selection (TRAINING data only) ----------
num_cols = X_train.select_dtypes(include='number').columns.tolist()
cat_cols = X_train.select_dtypes(exclude='number').columns.tolist()

# Number columns -> Pearson correlation with Depression
for col in num_cols:
    r, p = pearsonr(X_train[col], y_train)
    print(f"{col:35s} r = {r:+.3f}   p = {p:.2e}")

# Text columns -> chi-square test (is the column linked to Depression?)
drop_cols = []
for col in cat_cols:
    chi2, p, _, _ = chi2_contingency(pd.crosstab(X_train[col], y_train))
    print(f"{col:35s} chi-square p = {p:.2e}")
    if p >= 0.05:
        drop_cols.append(col)
print("Dropped (no significant link with Depression):", drop_cols)   # -> Gender

X_train = X_train.drop(columns=drop_cols)
X_test = X_test.drop(columns=drop_cols)
cat_cols = [c for c in cat_cols if c not in drop_cols]

# Correlation heatmap of the number columns
corr = pd.concat([X_train[num_cols], y_train], axis=1).corr()
plt.figure(figsize=(8, 6))
plt.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)
plt.colorbar()
plt.xticks(range(len(corr)), corr.columns, rotation=45, ha='right')
plt.yticks(range(len(corr)), corr.columns)
for i in range(len(corr)):
    for j in range(len(corr)):
        plt.text(j, i, f"{corr.iloc[i, j]:.2f}", ha='center', va='center', fontsize=8)
plt.title('Correlation heatmap (training data)')
plt.tight_layout()
plt.show()

# ---------- 4. Preprocessing (fit on TRAIN only, via Pipeline) ----------
preprocess = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols),
    ('num', StandardScaler(), num_cols),
])

models = {
    'Baseline (most frequent)': DummyClassifier(strategy='most_frequent'),
    'Logistic regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random forest': RandomForestClassifier(
        n_estimators=300, max_depth=10, random_state=42, n_jobs=-1),
}

# ---------- 5. Train + evaluate ----------
rows = []
fitted = {}
for name, clf in models.items():
    pipe = Pipeline([('prep', preprocess), ('model', clf)])
    pipe.fit(X_train, y_train)
    fitted[name] = pipe

    for split, Xs, ys in [('Train', X_train, y_train), ('Test', X_test, y_test)]:
        pred = pipe.predict(Xs)
        proba = pipe.predict_proba(Xs)[:, 1]
        rows.append({
            'Method': name, 'Split': split,
            'Accuracy': accuracy_score(ys, pred),
            'Precision': precision_score(ys, pred),
            'Recall': recall_score(ys, pred),
            'F1': f1_score(ys, pred),
            'ROC AUC': roc_auc_score(ys, proba),
        })

    cv = cross_val_score(pipe, X_train, y_train, cv=5, scoring='f1')
    print(f"{name}: 5-fold CV F1 = {cv.mean():.3f} +/- {cv.std():.3f}")

results = pd.DataFrame(rows)
print(results.round(3))

# ---------- 6. Visualization ----------
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, (name, pipe) in zip(axes, fitted.items()):
    ConfusionMatrixDisplay.from_estimator(
        pipe, X_test, y_test, ax=ax, colorbar=False,
        display_labels=['Not depressed', 'Depressed'])
    ax.set_title(name)
plt.tight_layout()
plt.show()

# ROC curve
ax = plt.gca()
for name in ['Logistic regression', 'Random forest']:
    RocCurveDisplay.from_estimator(fitted[name], X_test, y_test, name=name, ax=ax)
ax.plot([0, 1], [0, 1], '--', color='gray', label='Random guess')
ax.legend()
plt.title('ROC curve (test data)')
plt.show()

# Feature importance from the random forest
rf_pipe = fitted['Random forest']
names = rf_pipe['prep'].get_feature_names_out()
imp = pd.Series(rf_pipe['model'].feature_importances_, index=names).sort_values()
imp.tail(15).plot(kind='barh', figsize=(10, 6), title='Top 15 features (Random forest)')
plt.tight_layout()
plt.show()

# ---------- 7. Error analysis (Logistic regression, test data) ----------
pred = fitted['Logistic regression'].predict(X_test)
errors = X_test.copy()
errors['Outcome'] = np.select(
    [(y_test == 1) & (pred == 1), (y_test == 1) & (pred == 0),
     (y_test == 0) & (pred == 0), (y_test == 0) & (pred == 1)],
    ['TP (caught)', 'FN (missed)', 'TN (correct)', 'FP (false alarm)'], default='')
errors['Suicidal Thoughts = Yes'] = (errors['Suicidal Thoughts'] == 'Yes') * 100

summary = errors.groupby('Outcome')[['Academic Pressure', 'Financial Stress',
                                     'Suicidal Thoughts = Yes']].mean()
summary['Students'] = errors['Outcome'].value_counts()
print(summary.round(2).to_string())
# Missed students (FN) have lower pressure and fewer suicidal thoughts than the
# depressed students the model caught -> they do not fit the usual pattern.

summary[['Academic Pressure', 'Financial Stress']].plot(
    kind='bar', title='Error analysis: average scores per outcome', rot=0)
plt.show()

# ---------- 8. Test without 'Suicidal Thoughts' ----------
# This question is almost a symptom of depression. Does the model still work without it?
cat_no_st = [c for c in cat_cols if c != 'Suicidal Thoughts']
pipe_no_st = Pipeline([
    ('prep', ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_no_st),
        ('num', StandardScaler(), num_cols),
    ])),
    ('model', LogisticRegression(max_iter=1000, random_state=42)),
])
pipe_no_st.fit(X_train, y_train)
pred_no_st = pipe_no_st.predict(X_test)
print("Without Suicidal Thoughts - accuracy:", round(accuracy_score(y_test, pred_no_st), 3),
      "| recall:", round(recall_score(y_test, pred_no_st), 3))
print(confusion_matrix(y_test, pred_no_st))
