"""
Depression Prediction - Python ML Project
-----------------------------------------
Goal : Predict whether a student is depressed (1) or not depressed (0).
Type : Supervised learning - Classification (the target has only 0 or 1).

Steps in this file:
    1. Load the data
    2. Explore the data (inspection, findings, charts)
    3. Clean the data and prepare the features
    4. Split 80-20, THEN encode (so test data never leaks into preprocessing)
    5. Build a baseline and two models
    6. Evaluate with a confusion matrix and other metrics
    7. Save the best model
    8. Predict for a new person
    9. Show the charts and the limitations
"""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "depression.csv"
MODEL_PATH = BASE_DIR / "depression_model.joblib"
EDA_CHART_PATH = BASE_DIR / "eda_charts.png"
RESULT_CHART_PATH = BASE_DIR / "model_evaluation.png"

TARGET_COLUMN = "Depression"
RANDOM_STATE = 42
TEST_SIZE = 0.2

OLD_SUICIDE_COLUMN = "Have you ever had suicidal thoughts ?"
NEW_SUICIDE_COLUMN = "Suicidal Thoughts"

BASELINE_NAME = "Baseline (Most Frequent)"

MIN_CITY_ROWS = 100
MIN_GROUP_ROWS = 100

def load_data(path):
    """Read the CSV file and give the awkward column a cleaner name."""
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find '{path.name}'. Put it in the same folder as main.py."
        )

    df = pd.read_csv(path)
    df = df.rename(columns={OLD_SUICIDE_COLUMN: NEW_SUICIDE_COLUMN})

    print(f"Data loaded: {df.shape[0]} rows and {df.shape[1]} columns")
    return df

def inspect_dataset(df):
    """Print a basic overview: sample rows, types, numbers, missing values, target."""
    print("\n===== Dataset overview =====")

    print("\nFirst 5 rows:")
    print(df.head().to_string())

    print("\nColumn types and non-empty counts:")
    df.info()

    print("\nSummary of the number columns:")
    print(df.describe().round(2).T.to_string())

    print("\nText columns and how many different values each has:")
    for column in df.select_dtypes(exclude="number").columns:
        unique_values = df[column].unique()
        if len(unique_values) <= 6:
            print(f"  {column}: {list(unique_values)}")
        else:
            print(f"  {column}: {len(unique_values)} different values")

    print("\nMissing values per column:")
    missing = df.isna().sum()
    print(missing[missing > 0].to_string() if missing.any() else "  None")

    print(f"\nDuplicate rows: {df.duplicated().sum()}")

    print("\nTarget column (what we want to predict):")
    counts = df[TARGET_COLUMN].value_counts()
    for value, label in [(0, "Not depressed"), (1, "Depressed")]:
        print(f"  {label} ({value}): {counts[value]} rows ({counts[value] / len(df):.1%})")


def depression_rate(df, condition):
    """Percent of depressed people among the rows that match the condition."""
    return df[condition][TARGET_COLUMN].mean() * 100


def print_findings(df):
    """Print what the data shows. Every number is calculated from the data."""
    print("\n===== Exploratory findings =====")

    overall = df[TARGET_COLUMN].mean() * 100
    print(f"1. {overall:.1f}% of the students are labelled depressed, "
          f"so the two classes are fairly balanced.")

    with_thoughts = depression_rate(df, df[NEW_SUICIDE_COLUMN] == "Yes")
    without_thoughts = depression_rate(df, df[NEW_SUICIDE_COLUMN] == "No")
    print(f"2. Students who had suicidal thoughts: {with_thoughts:.1f}% depressed, "
          f"compared with {without_thoughts:.1f}% for those who did not.")

    high_pressure = depression_rate(df, df["Academic Pressure"] >= 4)
    low_pressure = depression_rate(df, df["Academic Pressure"] <= 2)
    print(f"3. Academic Pressure of 4-5: {high_pressure:.1f}% depressed, "
          f"compared with {low_pressure:.1f}% at 0-2. Higher pressure goes with more depression.")

    high_stress = depression_rate(df, df["Financial Stress"] >= 4)
    low_stress = depression_rate(df, df["Financial Stress"] <= 2)
    print(f"4. Financial Stress of 4-5: {high_stress:.1f}% depressed, "
          f"compared with {low_stress:.1f}% at 1-2.")

    short_sleep = depression_rate(df, df["Sleep Duration"] == "Less than 5 hours")
    long_sleep = depression_rate(df, df["Sleep Duration"] == "More than 8 hours")
    print(f"5. Sleeping less than 5 hours: {short_sleep:.1f}% depressed, "
          f"compared with {long_sleep:.1f}% for more than 8 hours.")

    print("\nData quality problems found:")
    student_share = (df["Profession"] == "Student").mean() * 100
    print(f"  - {student_share:.1f}% of the rows are students, so 'Profession' tells the model very little.")

    work_pressure_used = (df["Work Pressure"] != 0).sum()
    job_satisfaction_used = (df["Job Satisfaction"] != 0).sum()
    print(f"  - 'Work Pressure' is not 0 in only {work_pressure_used} rows and "
          f"'Job Satisfaction' in only {job_satisfaction_used} rows.")

    city_counts = df["City"].value_counts()
    bad_cities = city_counts[city_counts < MIN_CITY_ROWS]
    print(f"  - {len(bad_cities)} 'City' values appear fewer than {MIN_CITY_ROWS} times "
          f"({bad_cities.sum()} rows) and look like typing mistakes, "
          f"for example: {list(bad_cities.index[:5])}")


def plot_rate_bar(ax, df, column, title):
    """Bar chart: the percent of depressed people in each group of one column."""
    group_sizes = df[column].value_counts()
    big_groups = group_sizes[group_sizes >= MIN_GROUP_ROWS].index
    rates = (
        df[df[column].isin(big_groups)]
        .groupby(column)[TARGET_COLUMN]
        .mean()
        .sort_index()
        * 100
    )

    labels = [str(int(v)) if isinstance(v, float) and v.is_integer() else str(v) for v in rates.index]

    ax.bar(labels, rates.values, color="#4C72B0")
    ax.set_title(title)
    ax.set_ylabel("% depressed")
    ax.set_ylim(0, 100)
    ax.tick_params(axis="x", rotation=20)


def plot_eda(df, path):
    """Draw 6 charts that show the main patterns in the data."""
    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    fig.suptitle("Exploratory Data Analysis", fontsize=14)

    counts = df[TARGET_COLUMN].value_counts().sort_index()
    axes[0, 0].bar(["Not depressed", "Depressed"], counts.values, color=["#55A868", "#C44E52"])
    axes[0, 0].set_title("Target: Depressed vs Not Depressed")
    axes[0, 0].set_ylabel("Number of students")

    axes[0, 1].hist(
        [df[df[TARGET_COLUMN] == 0]["Age"], df[df[TARGET_COLUMN] == 1]["Age"]],
        bins=15, label=["Not depressed", "Depressed"], color=["#55A868", "#C44E52"],
    )
    axes[0, 1].set_title("Age")
    axes[0, 1].set_xlabel("Age")
    axes[0, 1].legend()

    plot_rate_bar(axes[0, 2], df, "Sleep Duration", "Depression by Sleep Duration")
    plot_rate_bar(axes[1, 0], df, "Academic Pressure", "Depression by Academic Pressure")
    plot_rate_bar(axes[1, 1], df, "Financial Stress", "Depression by Financial Stress")
    plot_rate_bar(axes[1, 2], df, NEW_SUICIDE_COLUMN, "Depression by Suicidal Thoughts")

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    print(f"\nEDA charts saved to: {path.name}")


def explore_data(df):
    """Run the whole EDA: overview, findings, and charts."""
    inspect_dataset(df)
    print_findings(df)
    plot_eda(df, EDA_CHART_PATH)

def clean_data(df):
    """Remove rows that would harm the model."""
    print("\n===== Cleaning the data =====")
    rows_start = len(df)

    df = df.dropna()
    print(f"Rows removed because of missing values: {rows_start - len(df)}")

    rows_before_city = len(df)
    city_counts = df["City"].value_counts()
    valid_cities = city_counts[city_counts >= MIN_CITY_ROWS].index
    df = df[df["City"].isin(valid_cities)]
    print(f"Rows removed because of invalid City values: {rows_before_city - len(df)}")

    print(f"Rows left after cleaning: {len(df)} (started with {rows_start})")
    return df.reset_index(drop=True)


def prepare_data(df):
    """
    Separate the cleaned data into:
        x = the features (the clues the model learns from)
        y = the target  (the answer we want to predict)
    The text columns are NOT encoded yet. That happens after the split.
    """
    y = df[TARGET_COLUMN]

    x = df.drop(columns=[TARGET_COLUMN, "id"])
    return x, y


def split_data(x, y):
    """
    Split into training (80%) and testing (20%) sets.
    stratify=y keeps the same depressed / not depressed ratio in both sets.
    """
    return train_test_split(
        x, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )


def encode_features(x_train, x_test):
    """
    Turn text columns into 0/1 columns (one-hot encoding).

    The columns are decided by the TRAINING data only. The test data is then
    arranged to match them. This way, the test data never influences how the
    data is prepared (no data leakage).
    """
    x_train_encoded = pd.get_dummies(x_train, drop_first=True).astype(float)

    x_test_encoded = pd.get_dummies(x_test)
    x_test_encoded = x_test_encoded.reindex(columns=x_train_encoded.columns, fill_value=0)
    x_test_encoded = x_test_encoded.astype(float)

    print(f"\nFeatures after encoding: {x_train_encoded.shape[1]} columns")
    print(f"Training rows: {len(x_train_encoded)} | Testing rows: {len(x_test_encoded)}")
    return x_train_encoded, x_test_encoded

def build_baseline(x_train, y_train):
    """
    The baseline always guesses the most common answer.
    A real model must beat this, or it has learned nothing useful.
    """
    model = DummyClassifier(strategy="most_frequent")
    model.fit(x_train, y_train)
    return model


def build_logistic_regression(x_train, y_train):
    """
    Logistic Regression is the classification version of Linear Regression.
    StandardScaler puts all columns on a similar scale so the model learns well.
    (The scaler learns from the training data only.)
    """
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    )
    model.fit(x_train, y_train)
    return model


def build_random_forest(x_train, y_train):
    """
    Random Forest (classifier version).
    GridSearchCV tries several settings with 5-fold cross-validation
    and keeps the best one, so we do not have to guess the depth.
    """
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [5, 8, 12],
    }

    search = GridSearchCV(
        RandomForestClassifier(random_state=RANDOM_STATE),
        param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
    )
    search.fit(x_train, y_train)

    print(f"Best Random Forest settings: {search.best_params_}")
    return search.best_estimator_

def cross_validate_model(name, model, x_train, y_train):
    """Test the model on 5 different parts of the training data."""
    scores = cross_val_score(
        model, x_train, y_train, cv=5, scoring="accuracy", n_jobs=-1
    )
    print(f"{name} cross-validation accuracy: {scores.mean():.4f} (+/- {scores.std():.4f})")


def evaluate_model(name, model, x_train, y_train, x_test, y_test):
    """Print the detailed results of one model and return its scores."""
    train_pred = model.predict(x_train)
    test_pred = model.predict(x_test)
    test_probability = model.predict_proba(x_test)[:, 1]

    print(f"\n----- {name} -----")
    print("Confusion matrix (rows = actual, columns = predicted):")
    print(confusion_matrix(y_test, test_pred))
    print("\nClassification report (test data):")
    print(classification_report(
        y_test, test_pred, target_names=["Not depressed", "Depressed"], zero_division=0
    ))

    return {
        "Method": name,
        "Train Accuracy": accuracy_score(y_train, train_pred),
        "Test Accuracy": accuracy_score(y_test, test_pred),
        "Precision": precision_score(y_test, test_pred, zero_division=0),
        "Recall": recall_score(y_test, test_pred, zero_division=0),
        "F1": f1_score(y_test, test_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, test_probability),
    }


def print_baseline_comparison(results):
    """Show how much better each real model is than the baseline."""
    baseline = next(row for row in results if row["Method"] == BASELINE_NAME)

    print(f"\nBaseline accuracy (always guess the most common answer): "
          f"{baseline['Test Accuracy']:.1%}")
    for row in results:
        if row["Method"] == BASELINE_NAME:
            continue
        gain = (row["Test Accuracy"] - baseline["Test Accuracy"]) * 100
        print(f"  {row['Method']}: {row['Test Accuracy']:.1%} "
              f"({gain:+.1f} percentage points compared with the baseline)")


def pick_best_model(results):
    """Return the result row of the real model with the highest ROC-AUC."""
    real_models = [row for row in results if row["Method"] != BASELINE_NAME]
    return max(real_models, key=lambda row: row["ROC-AUC"])

def save_model(model, model_name, feature_columns, path):
    """
    Save the trained model to a file.
    We also save the feature column names, because a new person's data
    must be arranged in exactly the same columns the model learned from.
    """
    joblib.dump(
        {"model": model, "model_name": model_name, "feature_columns": feature_columns},
        path,
    )
    print(f"\nBest model ({model_name}) saved to: {path.name}")


def load_model(path):
    """Load a model that was saved by save_model()."""
    return joblib.load(path)


def predict_new_person(saved, person):
    """
    Predict for one new person.
    'person' is a dictionary using the same column names as the dataset.
    """
    new_data = pd.DataFrame([person])

    new_data = pd.get_dummies(new_data)
    new_data = new_data.reindex(columns=saved["feature_columns"], fill_value=0)
    new_data = new_data.astype(float)

    model = saved["model"]
    label = model.predict(new_data)[0]
    probability = model.predict_proba(new_data)[0, 1]
    return label, probability

def ask_number(prompt, low, high, whole_number):
    """Keep asking until the user types a valid number between low and high."""
    while True:
        answer = input(f"{prompt} ({low} to {high}): ").strip()
        try:
            value = int(answer) if whole_number else float(answer)
        except ValueError:
            print("  Please type a number.")
            continue

        if low <= value <= high:
            return value
        print(f"  Please type a value from {low} to {high}.")


def ask_choice(prompt, options):
    """
    Ask the user to pick one option.
    Short lists  -> numbered menu (type the number).
    Long lists   -> type the name; type '?' to see all the choices.
    """
    options = sorted(options)

    if len(options) <= 6:
        while True:
            print(f"\n{prompt}")
            for number, option in enumerate(options, start=1):
                print(f"  {number}. {option}")

            answer = input("Type the number of your choice: ").strip()
            if answer.isdigit() and 1 <= int(answer) <= len(options):
                return options[int(answer) - 1]
            print("  Invalid choice, please try again.")

    lookup = {option.lower(): option for option in options}
    while True:
        answer = input(f"\n{prompt} (type '?' to see the list): ").strip()
        if answer == "?":
            print("  " + ", ".join(options))
        elif answer.lower() in lookup:
            return lookup[answer.lower()]
        else:
            print("  Not found in the dataset. Type '?' to see the choices.")


def ask_yes_no(prompt):
    """Return True if the user answers yes, False if the user answers no."""
    while True:
        answer = input(f"{prompt} (yes/no): ").strip().lower()
        if answer in ("yes", "y"):
            return True
        if answer in ("no", "n"):
            return False
        print("  Please type yes or no.")


def ask_new_person(df):
    """
    Ask the user for every detail about one new person.
    The allowed choices and number ranges come from the cleaned dataset,
    so the user can only enter values the model has seen before.
    """
    print("\n===== Enter the new person's details =====")

    def choices(column):
        return df[column].unique().tolist()

    def low(column):
        return df[column].min()

    def high(column):
        return df[column].max()

    person = {}
    person["Gender"] = ask_choice("Gender:", choices("Gender"))
    person["Age"] = ask_number("Age", int(low("Age")), int(high("Age")), True)
    person["City"] = ask_choice("City:", choices("City"))
    person["Profession"] = ask_choice("Profession:", choices("Profession"))
    person["Academic Pressure"] = ask_number(
        "Academic Pressure", int(low("Academic Pressure")), int(high("Academic Pressure")), True
    )
    person["Work Pressure"] = ask_number(
        "Work Pressure", int(low("Work Pressure")), int(high("Work Pressure")), True
    )
    person["CGPA"] = ask_number("CGPA", float(low("CGPA")), float(high("CGPA")), False)
    person["Study Satisfaction"] = ask_number(
        "Study Satisfaction", int(low("Study Satisfaction")), int(high("Study Satisfaction")), True
    )
    person["Job Satisfaction"] = ask_number(
        "Job Satisfaction", int(low("Job Satisfaction")), int(high("Job Satisfaction")), True
    )
    person["Sleep Duration"] = ask_choice("Sleep Duration:", choices("Sleep Duration"))
    person["Dietary Habits"] = ask_choice("Dietary Habits:", choices("Dietary Habits"))
    person["Degree"] = ask_choice("Degree:", choices("Degree"))
    person[NEW_SUICIDE_COLUMN] = ask_choice(
        "Have you ever had suicidal thoughts?", choices(NEW_SUICIDE_COLUMN)
    )
    person["Work/Study Hours"] = ask_number(
        "Work/Study Hours", int(low("Work/Study Hours")), int(high("Work/Study Hours")), True
    )
    person["Financial Stress"] = float(
        ask_number("Financial Stress", int(low("Financial Stress")), int(high("Financial Stress")), True)
    )
    person["Family History of Mental Illness"] = ask_choice(
        "Family history of mental illness?", choices("Family History of Mental Illness")
    )
    return person


def show_prediction(saved, person):
    """Predict for one person and print the result in a readable way."""
    label, probability = predict_new_person(saved, person)
    result_text = "Depressed" if label == 1 else "Not depressed"

    print(f"\n----- Prediction ({saved['model_name']}) -----")
    print(f"Result              : {result_text}")
    print(f"Chance of depression: {probability:.1%}")
    print("(Demo only. This is not a medical diagnosis.)")

def plot_results(models, x_test, y_test, path):
    """Draw a confusion matrix for every model and one ROC curve chart."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Model Evaluation (test data)", fontsize=14)
    flat_axes = axes.flatten()

    for ax, (name, model) in zip(flat_axes[:3], models.items()):
        ConfusionMatrixDisplay.from_estimator(
            model, x_test, y_test,
            display_labels=["Not depressed", "Depressed"],
            cmap="Blues", colorbar=False, ax=ax,
        )
        ax.set_title(f"{name}\nConfusion Matrix")

    roc_ax = flat_axes[3]
    for name, model in models.items():
        if name != BASELINE_NAME:
            RocCurveDisplay.from_estimator(model, x_test, y_test, name=name, ax=roc_ax)
    roc_ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random guess")
    roc_ax.set_title("ROC Curve Comparison")
    roc_ax.legend()

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    print(f"Model charts saved to: {path.name}")

def print_limitations(df, best):
    """Print what this project cannot tell us. Numbers come from the results."""
    student_share = (df["Profession"] == "Student").mean() * 100
    missed = (1 - best["Recall"]) * 100
    false_alarms = (1 - best["Precision"]) * 100

    print("\n===== Limitations =====")
    print(f"1. Narrow group: {student_share:.1f}% of the rows are students and all the "
          f"cities are in India. The results may not apply to workers or other countries.")
    print("2. The labels come from a survey. We do not know if 'Depression' means a "
          "clinical diagnosis or a self-reported answer.")
    print(f"3. The best model ({best['Method']}) still makes mistakes. It misses about "
          f"{missed:.0f}% of the depressed students, and about {false_alarms:.0f}% of the "
          f"students it flags as depressed are not.")
    print("4. 'Suicidal Thoughts' is strongly linked to the label (see the EDA findings). "
          "A real screening tool may not be able to rely on such a sensitive question.")
    print("5. The model finds patterns, not causes. High pressure going with depression "
          "does not prove that pressure causes it.")
    print("6. Weak columns (Profession, Work Pressure, Job Satisfaction) are kept in the "
          "model. They hold almost no information and only add noise.")
    print("7. Only one dataset, one 80-20 split and 5-fold cross-validation were used. "
          "There is no separate outside data to confirm the results.")
    print("8. This is a learning project. It must not be used to diagnose anyone.")

def main():
    df_raw = load_data(DATA_PATH)

    explore_data(df_raw)

    df = clean_data(df_raw)
    x, y = prepare_data(df)

    x_train_raw, x_test_raw, y_train, y_test = split_data(x, y)
    x_train, x_test = encode_features(x_train_raw, x_test_raw)

    print("\nTraining the models...")
    models = {
        BASELINE_NAME: build_baseline(x_train, y_train),
        "Logistic Regression": build_logistic_regression(x_train, y_train),
        "Random Forest": build_random_forest(x_train, y_train),
    }

    print("\n===== Cross-validation =====")
    for name, model in models.items():
        cross_validate_model(name, model, x_train, y_train)

    print("\n===== Test results =====")
    results = [
        evaluate_model(name, model, x_train, y_train, x_test, y_test)
        for name, model in models.items()
    ]

    print("\n===== Model comparison =====")
    print(pd.DataFrame(results).round(4).to_string(index=False))
    print_baseline_comparison(results)

    best = pick_best_model(results)
    save_model(models[best["Method"]], best["Method"], list(x_train.columns), MODEL_PATH)

    saved = load_model(MODEL_PATH)
    while ask_yes_no("\nDo you want to predict for a new person?"):
        person = ask_new_person(df)
        show_prediction(saved, person)

    plot_results(models, x_test, y_test, RESULT_CHART_PATH)
    print_limitations(df_raw, best)
    plt.show()


if __name__ == "__main__":
    main()