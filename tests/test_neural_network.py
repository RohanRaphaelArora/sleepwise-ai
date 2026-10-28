import numpy as np

from sleepwise.numpy_mlp import NumpyMLPRegressor


def test_backpropagation_matches_numerical_gradients():
    """Analytic gradients from back-propagation must equal central finite differences."""
    rng = np.random.default_rng(0)
    X, y = rng.normal(size=(12, 5)), rng.normal(size=12)
    net = NumpyMLPRegressor(hidden_layer_sizes=(6, 4), l2=1e-2)
    params = net._init_params(5, rng)
    _, grads = net._loss_and_grads(params, X, y)
    worst = 0.0
    for k, p in enumerate(params):
        it = np.nditer(p, flags=["multi_index"])
        for _ in it:
            idx, old, h = it.multi_index, p[it.multi_index], 1e-6
            p[idx] = old + h
            plus = net._loss_and_grads(params, X, y)[0]
            p[idx] = old - h
            minus = net._loss_and_grads(params, X, y)[0]
            p[idx] = old
            numeric = (plus - minus) / (2 * h)
            worst = max(worst, abs(numeric - grads[k][idx]) / max(1e-8, abs(numeric) + abs(grads[k][idx])))
    assert worst < 1e-5


def test_numpy_network_learns_a_simple_function():
    rng = np.random.default_rng(1)
    X = rng.uniform(-1, 1, size=(200, 2))
    y = 3 * X[:, 0] - 2 * X[:, 1] + 1
    net = NumpyMLPRegressor(hidden_layer_sizes=(8,), epochs=150, random_state=0).fit(X, y)
    rmse = float(np.sqrt(np.mean((net.predict(X) - y) ** 2)))
    assert rmse < 0.1
    assert net.loss_curve_[-1] < net.loss_curve_[0]  # training loss decreases


def test_numpy_network_is_reproducible_with_a_seed():
    rng = np.random.default_rng(2)
    X, y = rng.normal(size=(40, 3)), rng.normal(size=40)
    a = NumpyMLPRegressor(epochs=20, random_state=7).fit(X, y).predict(X)
    b = NumpyMLPRegressor(epochs=20, random_state=7).fit(X, y).predict(X)
    assert np.allclose(a, b)
