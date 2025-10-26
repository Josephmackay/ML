from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
from src.logs.logger_config import get_logger
logger = get_logger(__name__)

def evaluate_model(model_path, X_test, y_test):
    """Evaluate a saved model."""

    model = joblib.load(model_path)
    y_pred = model.predict(X_test)

    logger.info(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    logger.info(f"Confusion Matrix:\n {confusion_matrix(y_test, y_pred)}")
    logger.info(f"Classification Report:\n {classification_report(y_test, y_pred)}")
