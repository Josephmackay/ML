from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib
import os
from src.logs.logger_config import get_logger
logger = get_logger(__name__)

def tune_logreg(X_train, y_train, cv=5, save_path="saved_models/log_reg_best.pkl"):
    """Perform extensive GridSearch for Logistic Regression to improve accuracy."""

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=5000, random_state=42))
    ])

    param_grid = {
        "clf__C": [0.01, 0.1, 1, 10, 50, 100],             # Regularization strength
        "clf__penalty": ["l1", "l2", "elasticnet", "none"],# Regularization type
        "clf__solver": ["liblinear", "saga"],              # solvers that support l1/l2/elasticnet
        "clf__class_weight": [None, "balanced"],          # handle imbalance
        "clf__l1_ratio": [0, 0.25, 0.5, 0.75, 1]          # only used for elasticnet
    }

    grid = GridSearchCV(pipe, param_grid, cv=cv, scoring="accuracy", n_jobs=-1, verbose=2)
    grid.fit(X_train, y_train)


    logger.info(f"Best Params: {grid.best_params_}")
    logger.info(f"Best Score: {grid.best_score_}")

    save_path = os.path.join(os.path.dirname(__file__), "saved_models", "log_reg_best.pkl")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(grid.best_estimator_, save_path)
    logger.info(f"Tuned model saved to {save_path}")

    return grid.best_estimator_
