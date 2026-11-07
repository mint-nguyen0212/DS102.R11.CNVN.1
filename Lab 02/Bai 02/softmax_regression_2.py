import numpy as np
from tqdm import tqdm
from sklearn.metrics import precision_score, f1_score, recall_score


class SoftmaxRegression:
    def __init__(self, learning_rate: float = 0.01, epochs: int = 1000, num_classes: int = 5):
        self.W = None
        self.b = None
        self.lr = learning_rate
        self.epochs = epochs
        self.num_classes = num_classes
        self.loss_history = []

    def softmax(self, z: np.ndarray):
        z_shifted = z - z.max(axis=1, keepdims=True)
        exp_z = np.exp(z_shifted)
        return exp_z / exp_z.sum(axis=1, keepdims=True)

    def one_hot(self, y: np.ndarray):
        N = y.shape[0]
        Y_oh = np.zeros((N, self.num_classes))
        Y_oh[np.arange(N), y.astype(int)] = 1
        return Y_oh

    def loss_fn(self, y_hat: np.ndarray, Y_oh: np.ndarray):
        eps = 1e-12
        y_hat = np.clip(y_hat, eps, 1 - eps)
        return -(Y_oh * np.log(y_hat)).sum(axis=1).mean()

    def fit(self, X: np.ndarray, y: np.ndarray):
        N, D = X.shape
        self.W = np.zeros((D, self.num_classes))
        self.b = np.zeros(self.num_classes)
        self.loss_history = []

        Y_oh = self.one_hot(y)

        pbar = tqdm(range(self.epochs), desc="Training")
        for _ in pbar:
            z = X @ self.W + self.b
            y_hat = self.softmax(z)

            loss = self.loss_fn(y_hat, Y_oh)
            self.loss_history.append(loss)

            diff = y_hat - Y_oh
            grad_W = X.T @ diff / N
            grad_b = diff.mean(axis=0)

            self.W -= self.lr * grad_W
            self.b -= self.lr * grad_b

            pbar.set_postfix(loss=f"{loss:.4f}")

    def predict(self, X: np.ndarray):
        z = X @ self.W + self.b
        probs = self.softmax(z)
        return np.argmax(probs, axis=1)

    def eval(self, y_pred, y_true):
        return {
            "Precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
            "Recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
            "F1-Score": f1_score(y_true, y_pred, average="macro", zero_division=0),
        }
