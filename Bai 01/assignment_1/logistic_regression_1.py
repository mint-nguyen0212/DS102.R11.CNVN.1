import numpy as np
from tqdm import tqdm
from sklearn.metrics import precision_score, f1_score, recall_score


class LogisticRegressionRebuild:
    def __init__(self, learning_rate: float = 0.01, epochs: int = 1000):
        self.w = None
        self.lr = learning_rate
        self.epochs = epochs
        self.loss_history = []

    def sigmoid(self, X: np.ndarray):
        """
            X: (N, D)
            tra ve xac suat (N,)
        """
        logits = X @ self.w
        return 1 / (1 + np.exp(-logits))

    def loss_fn(self, y_hat: np.ndarray, Y: np.ndarray):
        """Binary cross-entropy trung binh."""
        eps = 1e-12
        y_hat = np.clip(y_hat, eps, 1 - eps)
        return -((1 - Y) * np.log(1 - y_hat) + Y * np.log(y_hat)).mean()

    def fit(self, X: np.ndarray, Y: np.ndarray):
        N, D = X.shape
        self.w = np.zeros(D)
        self.loss_history = []

        pbar = tqdm(range(self.epochs), desc="Training")
        for _ in pbar:
            y_hat = self.sigmoid(X)
            loss = self.loss_fn(y_hat, Y)
            self.loss_history.append(loss)

            grad = X.T @ (y_hat - Y) / N
            self.w = self.w - self.lr * grad
            pbar.set_postfix(loss=f"{loss:.4f}")

    def predict(self, X: np.ndarray):
        probs = self.sigmoid(X)
        return np.where(probs >= 0.5, 1, 0)

    def eval(self, y_hat, y):
        return {
            "Precision": precision_score(y, y_hat),
            "Recall": recall_score(y, y_hat),
            "F1-Score": f1_score(y, y_hat),
        }
