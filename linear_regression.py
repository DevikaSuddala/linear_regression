"""
Linear Regression from Scratch (NumPy only)
===========================================
Implements the hypothesis, MSE cost, gradient and gradient descent by hand.
scikit-learn is used ONLY for: loading the dataset, train_test_split,
and the final LinearRegression sanity-check comparison.

Run:  python linear_regression_scratch.py
Outputs: cost_curve_multiple.png, lr_experiment.png, simple_regression_fit.png
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # works without a display; remove if you want pop-up windows
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression  # sanity check ONLY

np.random.seed(42)


# ----------------------------------------------------------------------
# 1. Core model: hypothesis, cost, gradient, gradient descent
# ----------------------------------------------------------------------
def add_bias(X):
    """Prepend a column of ones so theta[0] is the intercept."""
    return np.c_[np.ones(X.shape[0]), X]


def predict(X, theta):
    """Hypothesis h(X) = X @ theta  (X already includes the bias column)."""
    return X @ theta


def cost(X, y, theta):
    """MSE cost J(theta) = (1 / 2m) * sum((X theta - y)^2)."""
    m = len(y)
    err = predict(X, theta) - y
    return (err @ err) / (2 * m)


def gradient(X, y, theta):
    """dJ/dtheta = (1/m) * X^T (X theta - y)."""
    m = len(y)
    return (X.T @ (predict(X, theta) - y)) / m


def gradient_descent(X, y, alpha=0.1, n_iters=1000, tol=1e-10):
    """Batch gradient descent. Returns the final theta and the cost history."""
    theta = np.zeros(X.shape[1])
    history = [cost(X, y, theta)]
    for _ in range(n_iters):
        theta = theta - alpha * gradient(X, y, theta)
        history.append(cost(X, y, theta))
        if not np.isfinite(history[-1]):  # diverged
            break
        if abs(history[-2] - history[-1]) < tol:  # converged
            break
    return theta, np.array(history)


def numerical_gradient(X, y, theta, eps=1e-5):
    """Finite-difference gradient, used only to verify the analytic one."""
    grad = np.zeros_like(theta)
    for j in range(len(theta)):
        step = np.zeros_like(theta)
        step[j] = eps
        grad[j] = (cost(X, y, theta + step) - cost(X, y, theta - step)) / (2 * eps)
    return grad


# ----------------------------------------------------------------------
# 2. Metrics computed manually
# ----------------------------------------------------------------------
def mse_score(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def r2_score_manual(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return 1 - ss_res / ss_tot


# ----------------------------------------------------------------------
# 3. Data loading, cleaning, splitting, scaling
# ----------------------------------------------------------------------
def prepare(df, feature_cols, target_col, test_size=0.2):
    """Split FIRST, then standardize using training statistics only."""
    X = df[feature_cols].values.astype(float)
    y = df[target_col].values.astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    mu = X_train.mean(axis=0)
    sigma = X_train.std(axis=0)
    X_train_s = (X_train - mu) / sigma
    X_test_s = (X_test - mu) / sigma
    return X_train_s, X_test_s, y_train, y_test


def run_experiment(name, df, feature_cols, target_col, alpha, n_iters):
    print(f"\n{'=' * 60}\n{name}\n{'=' * 60}")
    X_train_s, X_test_s, y_train, y_test = prepare(df, feature_cols, target_col)
    Xb_train, Xb_test = add_bias(X_train_s), add_bias(X_test_s)

    # Verify the analytic gradient against the numerical one (on a random theta)
    theta_rand = np.random.randn(Xb_train.shape[1])
    diff = np.max(np.abs(gradient(Xb_train, y_train, theta_rand)
                         - numerical_gradient(Xb_train, y_train, theta_rand)))
    print(f"Gradient check, max |analytic - numerical| = {diff:.2e}  (should be ~1e-8 or less)")

    # Train
    theta, history = gradient_descent(Xb_train, y_train, alpha=alpha, n_iters=n_iters)
    print(f"Trained for {len(history) - 1} iterations, "
          f"cost {history[0]:.4f} -> {history[-1]:.4f}")

    # Evaluate on the TEST set (manual metrics)
    y_pred = predict(Xb_test, theta)
    mse = mse_score(y_test, y_pred)
    r2 = r2_score_manual(y_test, y_pred)
    print(f"[Scratch] Test MSE = {mse:.4f} | Test R2 = {r2:.4f}")

    # Sanity check against scikit-learn
    sk = LinearRegression().fit(X_train_s, y_train)
    sk_pred = sk.predict(X_test_s)
    sk_mse = mse_score(y_test, sk_pred)
    sk_r2 = r2_score_manual(y_test, sk_pred)
    print(f"[sklearn] Test MSE = {sk_mse:.4f} | Test R2 = {sk_r2:.4f}")

    comparison = pd.DataFrame({
        "feature": ["intercept"] + list(feature_cols),
        "scratch": theta,
        "sklearn": np.r_[sk.intercept_, sk.coef_],
    })
    comparison["abs_diff"] = (comparison["scratch"] - comparison["sklearn"]).abs()
    print("\nCoefficient comparison (standardized features):")
    print(comparison.to_string(index=False, float_format=lambda v: f"{v:.6f}"))
    print(f"Max coefficient difference: {comparison['abs_diff'].max():.2e}")

    return dict(
        Xb_train=Xb_train, Xb_test=Xb_test, y_train=y_train, y_test=y_test,
        theta=theta, history=history, mse=mse, r2=r2,
    )


# ----------------------------------------------------------------------
# 4. Main
# ----------------------------------------------------------------------
def main():
    # --- Load ---
    data = fetch_california_housing(as_frame=True)
    df = data.frame
    target = "MedHouseVal"
    print("Dataset shape:", df.shape)
    print("Missing values per column:\n", df.isna().sum().to_string())
    df = df.fillna(df.median())  # no-op for this dataset, needed for messy CSVs
    # To use your own CSV instead:
    #   df = pd.read_csv("your_file.csv").dropna()
    #   target = "your_target_column"

    all_features = [c for c in df.columns if c != target]

    # --- Multiple linear regression (all features) ---
    multi = run_experiment(
        "MULTIPLE LINEAR REGRESSION (8 features)",
        df, all_features, target, alpha=0.1, n_iters=2000,
    )

    plt.figure(figsize=(7, 4))
    plt.plot(multi["history"])
    plt.xlabel("Iteration")
    plt.ylabel("Cost J(theta)")
    plt.title("Gradient descent convergence (multiple regression, alpha=0.1)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("cost_curve_multiple.png", dpi=150)
    plt.close()

    # --- Learning-rate experiment ---
    plt.figure(figsize=(7, 4))
    for a in [0.001, 0.01, 0.1, 0.5]:
        _, h = gradient_descent(multi["Xb_train"], multi["y_train"],
                                alpha=a, n_iters=300, tol=0)
        plt.plot(h, label=f"alpha={a}")
    plt.xlabel("Iteration")
    plt.ylabel("Cost J(theta)")
    plt.title("Effect of learning rate")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("lr_experiment.png", dpi=150)
    plt.close()

    # --- Simple linear regression (single feature: MedInc) ---
    simple = run_experiment(
        "SIMPLE LINEAR REGRESSION (feature: MedInc)",
        df, ["MedInc"], target, alpha=0.1, n_iters=2000,
    )

    x_test = simple["Xb_test"][:, 1]
    order = np.argsort(x_test)
    plt.figure(figsize=(7, 4))
    plt.scatter(x_test, simple["y_test"], s=4, alpha=0.3, label="test data")
    plt.plot(x_test[order], predict(simple["Xb_test"], simple["theta"])[order],
             color="red", label="fitted line")
    plt.xlabel("MedInc (standardized)")
    plt.ylabel("MedHouseVal")
    plt.title("Simple linear regression fit")
    plt.legend()
    plt.tight_layout()
    plt.savefig("simple_regression_fit.png", dpi=150)
    plt.close()

    # --- Summary ---
    print("\n" + "=" * 60)
    print("SUMMARY (test set)")
    print("=" * 60)
    print(f"Simple   (MedInc only): MSE = {simple['mse']:.4f}, R2 = {simple['r2']:.4f}")
    print(f"Multiple (8 features) : MSE = {multi['mse']:.4f}, R2 = {multi['r2']:.4f}")
    print("\nSaved: cost_curve_multiple.png, lr_experiment.png, simple_regression_fit.png")


if __name__ == "__main__":
    main()