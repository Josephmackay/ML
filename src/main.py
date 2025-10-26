# src/main.py
from src.data.data_import import load_data
from src.feature.preprocess import preprocess_data
from src.feature.feature_engineering import create_features
from src.model.train import train_model
from src.model.tune import tune_logreg  # Optional if tuning separately
import joblib
import os
from src.data.data_import import load_data
from src.logs.logger_config import get_logger
logger = get_logger(__name__)

def main():
    logger.info("\nStarting Loan Default Prediction Pipeline...")

    # 1. Load dataset
    df = load_data("data/Loan_default.csv")
    logger.info(df.head())

    # 2. Create derived features
    df = create_features(df)

    # 3. Specify target column (replace with your actual target)
    target_col = "Default"

    # 4. Preprocess data (encode, keep X and y)
    df = preprocess_data(df)
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # 5. Train model (SMOTE, scaling, evaluation, saving handled inside)
    model_save_path = "src/model/saved_models/log_reg_model.pkl"
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    trained_model, X_train, X_test, y_train, y_test = train_model(X, y, save_path=model_save_path)

    logger.info("\nPipeline completed successfully!")

if __name__ == "__main__":
    main()
