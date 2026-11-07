"""Assignment 2: Softmax Regression (NumPy) cho bai toan phan loai rating sach.

Target: phan loai 5 muc rating (1, 2, 3, 4, 5).
    label 0 = rating 1
    label 1 = rating 2
    label 2 = rating 3
    label 3 = rating 4
    label 4 = rating 5

Chay:  .venv\\Scripts\\python.exe assignment_2.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from softmax_regression_2 import SoftmaxRegression

DATA_PATH = Path("data/books_new.csv")
FIGURE_DIR = Path("figures")

FEATURES = [
    "price_gbp",
    "title_length",
    "word_count",
    "avg_word_length",
    "uppercase_count",
    "digit_count",
]

NUM_CLASSES = 5
LEARNING_RATE = 0.01
EPOCHS = 1000
TEST_SIZE = 0.2
RANDOM_STATE = 42


def load_data(path: Path = DATA_PATH):
    df = pd.read_csv(path)
    df = df.drop_duplicates(subset="title").reset_index(drop=True)

    X = df[FEATURES].to_numpy(dtype=np.float64)
    y = (df["rating"].to_numpy(dtype=np.int64)) - 1
    return X, y


def standardize(X_train: np.ndarray, X_test: np.ndarray):
    mu = X_train.mean(axis=0)
    sd = X_train.std(axis=0) + 1e-8
    return (X_train - mu) / sd, (X_test - mu) / sd


def plot_loss(loss_history, path: Path = FIGURE_DIR / "loss_curve_softmax.png"):
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, len(loss_history) + 1), loss_history, color="tab:blue", linewidth=1.5)
    ax.set_title("Ham loss trong qua trinh train (Softmax Regression)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-Entropy Loss")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

    print(f"Da luu bieu do loss: {path}")


def main():
    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    X_train, X_test = standardize(X_train, X_test)

    model = SoftmaxRegression(learning_rate=LEARNING_RATE, epochs=EPOCHS, num_classes=NUM_CLASSES)
    model.fit(X_train, y_train)

    z_train = X_train @ model.W + model.b
    final_loss = model.loss_fn(model.softmax(z_train), model.one_hot(y_train))

    y_pred = model.predict(X_test)
    scores = model.eval(y_pred, y_test)

    print()
    print("Training result:")
    print(f"{'Loss (train)':<12} {final_loss:.4f}")
    print("Testing and evaluation result:")
    for name, value in scores.items():
        print(f"{name:<12} {value:.4f}")

    plot_loss(model.loss_history)


if __name__ == "__main__":
    main()
