"""
Confidence-Calibrated SVM Classification for MASD (Section II-C, Eq. 9-10).

RBF-kernel SVM with temperature scaling:
    K(v_i, v_j) = exp(-gamma * ||v_i - v_j||^2)      (Eq. 9)
    P(y=1|v) = 1 / (1 + exp(-f(v) / T))               (Eq. 10)

Temperature T optimized via maximum likelihood on calibration set.
Achieves ECE = 1.1% (93% improvement over uncalibrated ECE = 15.1%).
"""

import numpy as np
from sklearn.svm import SVC
from scipy.optimize import minimize
from sklearn.metrics import log_loss


class CalibratedSVM:
    """
    Temperature-scaled RBF-kernel SVM.

    Args:
        C (float): Regularization parameter (default: 10).
        gamma (float): RBF kernel width (default: 0.01).
        class_weight: Class weighting for imbalance (default: 'balanced').
    """

    def __init__(self, C: float = 10.0, gamma: float = 0.01, class_weight="balanced"):
        self.svm = SVC(
            kernel="rbf",
            C=C,
            gamma=gamma,
            class_weight=class_weight,
            decision_function_shape="ovr",
        )
        self.temperature = 1.0  # Will be optimized

    def fit(self, X_train, y_train):
        """Train the SVM on fused features."""
        self.svm.fit(X_train, y_train)

    def calibrate(self, X_cal, y_cal):
        """
        Learn optimal temperature T on calibration set (Eq. 10).

        Minimizes negative log-likelihood using L-BFGS-B.

        Args:
            X_cal: Calibration features.
            y_cal: Calibration labels.
        """
        decision_values = self.svm.decision_function(X_cal)

        def nll(T):
            T = max(T[0], 1e-4)
            probs = 1.0 / (1.0 + np.exp(-decision_values / T))
            probs = np.clip(probs, 1e-7, 1 - 1e-7)
            return log_loss(y_cal, probs)

        result = minimize(nll, x0=[1.0], method="L-BFGS-B", bounds=[(0.01, 10.0)])
        self.temperature = result.x[0]

    def predict(self, X):
        """Predict class labels."""
        return self.svm.predict(X)

    def predict_proba(self, X):
        """
        Get calibrated probabilities using temperature scaling (Eq. 10).

        Returns:
            probs: (N, 2) array — [P(real), P(fake)] per sample.
        """
        decision_values = self.svm.decision_function(X)
        p_fake = 1.0 / (1.0 + np.exp(-decision_values / self.temperature))
        p_fake = np.clip(p_fake, 1e-7, 1 - 1e-7)
        return np.stack([1 - p_fake, p_fake], axis=-1)

    def predict_with_confidence(self, X):
        """
        Predict with calibrated confidence scores.

        Returns:
            predictions: (N,) — class labels.
            confidences: (N,) — max(P(real), P(fake)).
        """
        probs = self.predict_proba(X)
        predictions = np.argmax(probs, axis=1)
        confidences = np.max(probs, axis=1)
        return predictions, confidences

    def save(self, path):
        import pickle
        with open(path, "wb") as f:
            pickle.dump({"svm": self.svm, "temperature": self.temperature}, f)

    def load(self, path):
        import pickle
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.svm = data["svm"]
        self.temperature = data["temperature"]
