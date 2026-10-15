
import os
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

DATA_PATH = "StudentsPerformance.csv"
MODEL_PATH = "student_model.pkl"

CATEGORICAL = [
    "gender",
    "race",
    "parent_education",
    "lunch",
    "test_preparation",
]

NUMERICAL = [
    "math_score",
    "reading_score",
    "writing_score",
]

TARGET = "average_score"


def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"{DATA_PATH} not found. Put your CSV in the project folder."
        )

    df = pd.read_csv(DATA_PATH)

    # Support the original Kaggle dataset column names.
    aliases = {
        "race/ethnicity": "race",
        "parental level of education": "parent_education",
        "test preparation course": "test_preparation",
    }
    df.rename(columns=aliases, inplace=True)

    # Rename score columns to match the app.
    df.rename(
        columns={
            "math score": "math_score",
            "reading score": "reading_score",
            "writing score": "writing_score",
        },
        inplace=True,
    )

    missing = [c for c in CATEGORICAL + NUMERICAL if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required CSV columns: {missing}")

    for column in NUMERICAL:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=NUMERICAL)
    if df.empty:
        raise ValueError("The dataset contains no usable score records.")

    df[TARGET] = df[NUMERICAL].mean(axis=1)

    preprocessing = ColumnTransformer(
        transformers=[
            (
                "categorical",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OneHotEncoder(handle_unknown="ignore")),
                ]),
                CATEGORICAL,
            ),
            (
                "numerical",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                ]),
                NUMERICAL,
            ),
        ]
    )

    model = Pipeline([
        ("preprocessing", preprocessing),
        (
            "regressor",
            RandomForestRegressor(
                n_estimators=250,
                random_state=42,
                min_samples_leaf=2,
                n_jobs=-1,
            ),
        ),
    ])

    features = CATEGORICAL + NUMERICAL
    model.fit(df[features], df[TARGET])

    score_bins = pd.cut(
        df[TARGET],
        bins=[0, 20, 40, 60, 80, 100.1],
        right=False,
        labels=["0–19", "20–39", "40–59", "60–79", "80–100"],
    ).value_counts(sort=False).tolist()

    joblib.dump(
        {
            "model": model,
            "features": features,
            "dataset_averages": {
                column: round(float(df[column].mean()), 2)
                for column in NUMERICAL
            },
            "overall_average": round(float(df[TARGET].mean()), 2),
            "training_rows": int(len(df)),
            "dataset_distribution": score_bins,
        },
        MODEL_PATH,
    )

    print(f"Model saved successfully: {MODEL_PATH}")
    print(f"Training rows: {len(df)}")
    print(f"Dataset average: {df[TARGET].mean():.2f}")


if __name__ == "__main__":
    main()
