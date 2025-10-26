from collections import Counter
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib  # For saving models
from src.logs.logger_config import get_logger
logger = get_logger(__name__)

def train_model(X, y, cv=5, save_path="saved_models/log_reg_model.pkl"):
    """Train logistic regression model with SMOTE resampling."""

    # Check class distribution before SMOTE
    logger.info(f"Before SMOTE: {Counter(y)}")

    smote = SMOTE(sampling_strategy='auto', random_state=42, k_neighbors=5)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    logger.info(f"After SMOTE: {Counter(y_resampled)}")

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
    )

    # Pipeline
    log_reg = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=42))
    ])

    # Cross-validation
    scores = cross_val_score(log_reg, X_train, y_train, cv=cv, scoring="accuracy")
    logger.info("\n=== Logistic Regression ===")
    logger.info(f"Cross-Validation Accuracy: {scores}")
    logger.info(f"Mean Accuracy: {scores.mean():.4f} (+/- {scores.std():.4f})")

    # Fit model
    log_reg.fit(X_train, y_train)

    # Evaluate
    y_pred = log_reg.predict(X_test)
    
    logger.info("\n=== Logistic Regression Test Results ===")
    logger.info(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    logger.info(f"Confusion Matrix:\n {confusion_matrix(y_test, y_pred)}")
    logger.info(f"Classification Report:\n {classification_report(y_test, y_pred)}")

    # Save model
    joblib.dump(log_reg, save_path)
    logger.info(f"Model saved to {save_path}")

    return log_reg, X_train, X_test, y_train, y_test
