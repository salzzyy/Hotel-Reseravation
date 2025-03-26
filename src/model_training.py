import os
import pandas as pd
import joblib
from sklearn.model_selection import RandomizedSearchCV
import lightgbm as lgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src.logger import get_logger
from src.custom_exception import CustomException
from config.paths_config import *
from config.model_params import *
from utils.common_functions import read_yaml, load_data
import mlflow
import mlflow.sklearn

logger = get_logger(__name__)

class ModelTraining:
    """
    This class handles model training for LightGBM, including:
    - Loading and splitting data
    - Hyperparameter tuning using RandomizedSearchCV
    - Model training
    - Model evaluation
    - Model saving
    """

    def __init__(self, train_path, test_path, model_output_path):
        """Initialize paths and parameters"""
        self.train_path = train_path
        self.test_path = test_path
        self.model_output_path = model_output_path
        self.params_dist = LIGHTGM_PARAMS  # Hyperparameter search space
        self.random_search_params = RANDOM_SEARCH_PARAMS  # RandomizedSearchCV parameters
    
    def load_and_split_data(self):
        """Loads processed training and test data, then splits them into features and labels."""
        try:
            logger.info(f"Loading data from {self.train_path}")
            train_df = load_data(self.train_path)

            logger.info(f"Loading data from {self.test_path}")
            test_df = load_data(self.test_path)

            # Splitting into features and target variable
            X_train = train_df.drop(columns=["booking_status"])
            y_train = train_df["booking_status"] 
            
            X_test = test_df.drop(columns=["booking_status"])
            y_test = test_df["booking_status"]

            logger.info("Data successfully loaded and split for model training")
            return X_train, y_train, X_test, y_test
        
        except Exception as e:
            logger.error(f"Error while loading data: {e}")
            raise CustomException("Failed to load data", e)
    
    def train_lgbm(self, X_train, y_train):
        """Trains a LightGBM model with hyperparameter tuning using RandomizedSearchCV."""
        try:
            logger.info("Initializing LightGBM model")
            lgbm_model = lgb.LGBMClassifier(random_state=self.random_search_params["random_state"])
            
            logger.info("Starting hyperparameter tuning")
            random_search = RandomizedSearchCV(
                estimator=lgbm_model,
                param_distributions=self.params_dist,
                n_iter=self.random_search_params["n_iter"],
                cv=self.random_search_params["cv"],
                n_jobs=self.random_search_params["n_jobs"],
                verbose=self.random_search_params["verbose"],  
                scoring=self.random_search_params["scoring"]  
            )
            
            logger.info("Training the model")
            random_search.fit(X_train, y_train)
            
            best_params = random_search.best_params_
            best_lgbm_model = random_search.best_estimator_
            
            logger.info(f"Best parameters found: {best_params}")
            return best_lgbm_model
        
        except Exception as e:
            logger.error(f"Error while training model: {e}")
            raise CustomException("Failed to train model", e)
    
    def evaluate_model(self, model, X_test, y_test):
        """Evaluates the trained model and logs the metrics."""
        try:
            logger.info("Evaluating the model")
            y_pred = model.predict(X_test)
            
            # Calculating performance metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)

            # Logging the metrics
            logger.info(f"Accuracy: {accuracy}")
            logger.info(f"Precision: {precision}")
            logger.info(f"Recall: {recall}")
            logger.info(f"F1 Score: {f1}")

            return {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1
            }
        
        except Exception as e:
            logger.error(f"Error while evaluating model: {e}")
            raise CustomException("Failed to evaluate model", e)
    
    def save_model(self, model):
        """Saves the trained model as a .pkl file."""
        try:
            os.makedirs(os.path.dirname(self.model_output_path), exist_ok=True)  
            logger.info("Saving the model")

            joblib.dump(model, self.model_output_path)
            logger.info(f"Model saved successfully at {self.model_output_path}")
        
        except Exception as e:
            logger.error(f"Error while saving model: {e}")
            raise CustomException("Failed to save model", e)
    
    def run(self):
        """Executes the full model training pipeline."""
        try:
            with mlflow.start_run():
                logger.info("Starting model training pipeline")

                # Log dataset paths
                if os.path.exists(self.train_path):
                    mlflow.log_artifact(self.train_path, artifact_path="datasets")
                if os.path.exists(self.test_path):
                    mlflow.log_artifact(self.test_path, artifact_path="datasets")

                X_train, y_train, X_test, y_test = self.load_and_split_data()
                best_lgbm_model = self.train_lgbm(X_train, y_train)
                
                # Evaluate only once
                metrics = self.evaluate_model(best_lgbm_model, X_test, y_test)

                # Save the model
                self.save_model(best_lgbm_model)

                # Log model parameters
                mlflow.log_params(best_lgbm_model.get_params())  
                mlflow.log_metrics(metrics)

                logger.info("Model training completed successfully")

        except Exception as e:
            logger.error(f"Error in model training pipeline: {e}")
            raise CustomException("Failed during model training", e)

if __name__ == "__main__":
    # Run the training pipeline
    trainer = ModelTraining(PROCESSED_TRAIN_DATA_PATH, PROCESSSED_TEST_DATA_PATH, MODEL_OUTPUT_PATH)
    trainer.run()
