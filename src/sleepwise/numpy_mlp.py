"""A small feed-forward neural network written from scratch with NumPy.

Purpose: to make the mechanics of a neural network visible (forward pass, loss,
back-propagation, Adam updates) for study and for the viva. The deployed model uses
scikit-learn's ``MLPRegressor``; this class is compared against it.

Architecture:  inputs -> [Dense + ReLU] x L -> Dense (1 output, linear)
Loss:          0.5 * mean((y_hat - y)^2)  +  0.5 * l2 * sum(W^2)      (biases not penalised)
Optimiser:     mini-batch Adam
Init:          He normal (suited to ReLU)
Targets are standardised internally and un-standardised in ``predict``.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin


class NumpyMLPRegressor(RegressorMixin, BaseEstimator):
    def __init__(
        self,
        hidden_layer_sizes: tuple[int, ...] = (32, 16),
        learning_rate: float = 0.01,
        epochs: int = 300,
        batch_size: int = 32,
        l2: float = 1e-3,
        random_state: int | None = 42,
    ):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.l2 = l2
        self.random_state = random_state

    # ------------------------------------------------------------------ #
    # parameters are kept as one list: [W1, b1, W2, b2, ..., Wout, bout]
    # ------------------------------------------------------------------ #
    def _init_params(self, n_in: int, rng: np.random.Generator) -> list[np.ndarray]:
        sizes = [n_in, *self.hidden_layer_sizes, 1]
        params: list[np.ndarray] = []
        for fan_in, fan_out in zip(sizes[:-1], sizes[1:]):
            params.append(rng.normal(0.0, np.sqrt(2.0 / fan_in), size=(fan_in, fan_out)))  # He init
            params.append(np.zeros(fan_out))
        return params

    @staticmethod
    def _forward(params: list[np.ndarray], X: np.ndarray):
        """Return (prediction, cache). cache holds each layer's input and pre-activation."""
        activations = [X]
        pre_acts = []
        a = X
        n_layers = len(params) // 2
        for i in range(n_layers):
            W, b = params[2 * i], params[2 * i + 1]
            z = a @ W + b
            pre_acts.append(z)
            a = z if i == n_layers - 1 else np.maximum(z, 0.0)  # ReLU on hidden layers only
            activations.append(a)
        return a[:, 0], (activations, pre_acts)

    def _loss_and_grads(self, params: list[np.ndarray], X: np.ndarray, y: np.ndarray):
        """Loss and gradients via back-propagation."""
        n = X.shape[0]
        y_hat, (acts, pre) = self._forward(params, X)
        err = y_hat - y
        loss = 0.5 * np.mean(err**2)
        loss += 0.5 * self.l2 * sum(np.sum(params[i] ** 2) for i in range(0, len(params), 2))

        grads: list[np.ndarray] = [None] * len(params)  # type: ignore[list-item]
        delta = (err / n)[:, None]  # dLoss/dz at the (linear) output layer
        for i in reversed(range(len(params) // 2)):
            W = params[2 * i]
            grads[2 * i] = acts[i].T @ delta + self.l2 * W
            grads[2 * i + 1] = delta.sum(axis=0)
            if i > 0:
                delta = (delta @ W.T) * (pre[i - 1] > 0)  # chain rule through ReLU
        return loss, grads

    # ------------------------------------------------------------------ #
    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        if X.ndim != 2 or len(X) != len(y):
            raise ValueError("X must be 2-D with the same number of rows as y.")
        rng = np.random.default_rng(self.random_state)
        self.n_features_in_ = X.shape[1]
        self.y_mean_ = float(y.mean())
        self.y_std_ = float(y.std()) or 1.0
        ys = (y - self.y_mean_) / self.y_std_

        params = self._init_params(X.shape[1], rng)
        m = [np.zeros_like(p) for p in params]
        v = [np.zeros_like(p) for p in params]
        beta1, beta2, eps, step = 0.9, 0.999, 1e-8, 0
        n = len(X)
        batch = max(1, min(self.batch_size, n))
        self.loss_curve_: list[float] = []

        for _ in range(self.epochs):
            order = rng.permutation(n)
            for start in range(0, n, batch):
                idx = order[start:start + batch]
                _, grads = self._loss_and_grads(params, X[idx], ys[idx])
                step += 1
                for k, g in enumerate(grads):  # Adam update
                    m[k] = beta1 * m[k] + (1 - beta1) * g
                    v[k] = beta2 * v[k] + (1 - beta2) * g * g
                    m_hat = m[k] / (1 - beta1**step)
                    v_hat = v[k] / (1 - beta2**step)
                    params[k] -= self.learning_rate * m_hat / (np.sqrt(v_hat) + eps)
            self.loss_curve_.append(float(self._loss_and_grads(params, X, ys)[0]))

        self.params_ = params
        return self

    def predict(self, X):
        if not hasattr(self, "params_"):
            raise RuntimeError("Call fit before predict.")
        X = np.asarray(X, dtype=float)
        y_hat, _ = self._forward(self.params_, X)
        return y_hat * self.y_std_ + self.y_mean_
