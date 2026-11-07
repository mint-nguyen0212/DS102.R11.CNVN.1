from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATA_PATH = Path("data/books_new.csv")

FEATURES = [
    "price_gbp",
    "title_length",
    "word_count",
    "avg_word_length",
    "uppercase_count",
    "digit_count",
]

POSITIVE_RATINGS = {4, 5}
NEGATIVE_RATINGS = {1, 2}

TEST_SIZE = 0.2
RANDOM_STATE = 42


def print_scores(scores):
    for name, value in scores.items():
        print(f"{name:<12} {value:.4f}")


def evaluate(y_true, y_pred, average="binary"):
    return {
        "Precision": precision_score(y_true, y_pred, average=average, zero_division=0),
        "Recall": recall_score(y_true, y_pred, average=average, zero_division=0),
        "F1-Score": f1_score(y_true, y_pred, average=average, zero_division=0),
    }


def part_a():
    df = pd.read_csv(DATA_PATH)
    df = df.drop_duplicates(subset="title").reset_index(drop=True)
    df = df[df["rating"].isin(POSITIVE_RATINGS | NEGATIVE_RATINGS)].reset_index(drop=True)

    X = df[FEATURES].to_numpy(dtype=np.float64)
    y = df["rating"].isin(POSITIVE_RATINGS).to_numpy(dtype=np.float64)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000, solver="lbfgs", random_state=RANDOM_STATE)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    scores = evaluate(y_test, y_pred, average="binary")

    print("=" * 40)
    print("Logistic Regression (Binary)")
    print("=" * 40)
    print_scores(scores)


def part_b():
    df = pd.read_csv(DATA_PATH)
    df = df.drop_duplicates(subset="title").reset_index(drop=True)

    X = df[FEATURES].to_numpy(dtype=np.float64)
    y = df["rating"].to_numpy(dtype=np.int64) - 1

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000, solver="lbfgs", random_state=RANDOM_STATE)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    scores = evaluate(y_test, y_pred, average="macro")

    print()
    print("=" * 40)
    print("Softmax Regression (Multiclass)")
    print("=" * 40)
    print_scores(scores)


def main():
    part_a()
    part_b()


if __name__ == "__main__":
    main()

