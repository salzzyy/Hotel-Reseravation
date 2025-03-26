import os
import pandas as pd
import numpy as np
from src.logger import get_logger
from src.custom_exception import CustomException
from config.paths_config import *
from utils.common_functions import read_yaml, load_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

logger = get_logger(__name__)


class DataProcessor:
    def __init__(self, train_path, test_path, processed_dir, config_path):
        self.train_path = train_path
        self.test_path = test_path
        self.processed_dir = processed_dir
        self.config = read_yaml(config_path)

        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    def preprocessing_data(self, df):
        try:
            logger.info("Starting the Data Preprocessing step")

            logger.info("Dropping unnecessary columns")
            df.drop(columns=["Unnamed: 0", "Booking_ID"], errors="ignore", inplace=True)
            df.drop_duplicates(inplace=True)

            cat_cols = self.config["data_processing"]["categorical_columns"]
            num_cols = self.config["data_processing"]["numerical_columns"]

            logger.info("Applying Label Encoding")
            label_encoder = LabelEncoder()
            mappings = {}

            for col in cat_cols:
                df[col] = label_encoder.fit_transform(df[col])
                mappings[col] = {
                    label: code
                    for label, code in zip(
                        label_encoder.classes_,
                        label_encoder.transform(label_encoder.classes_),
                    )
                }

            logger.info("Label Encodings:")
            for col, mapping in mappings.items():
                logger.info(f"{col}: {mapping}")

            logger.info("Handling skewness in numerical features")
            skew_threshold = self.config["data_processing"]["skewness_threshold"]
            skewness = df[num_cols].apply(lambda x: x.skew())

            for column in skewness[skewness > skew_threshold].index:
                df[column] = np.log1p(df[column])

            return df

        except Exception as e:
            logger.error("Error during preprocessing step")
            raise CustomException("Error while preprocessing data", e)

    
    def balance_data(self, df):
        try:
            logger.info("Handling data imbalance using SMOTE")
            X = df.drop(columns=["booking_status"])
            y = df["booking_status"]

            #Ensure y is int
            y = y.astype(int)

            #Debug: Log class distribution
            logger.info(f"Class distribution before SMOTE:\n{y.value_counts()}")

            #Debug: Log data types
            logger.info(f"Before SMOTE, X dtypes:\n{X.dtypes}")

            #Check for categorical columns
            if X.select_dtypes(include=["object"]).shape[1] > 0:
                logger.error("Categorical features detected! Ensure all are encoded.")
                raise ValueError("Some categorical features were not encoded properly!")

            #Check for NaN values
            if X.isnull().sum().sum() > 0:
                logger.error(f"NaN values detected in X:\n{X.isnull().sum()}")
                raise ValueError("NaN values detected before SMOTE!")

            #Ensure X is a valid numeric matrix
            if not np.isfinite(X.to_numpy()).all():
                logger.error("Non-finite values detected in X! Check for Inf or NaN.")
                raise ValueError("X contains Inf or NaN values!")

            #Apply SMOTE
            smote = SMOTE(random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X, y)

            balanced_df = pd.DataFrame(X_resampled, columns=X.columns)
            balanced_df["booking_status"] = y_resampled

            logger.info("Data balancing successful")
            return balanced_df

        except Exception as e:
            logger.error(f"Error during data balancing: {str(e)}")
            raise CustomException("Error while balancing data", e)

    def select_feature(self, df):
        try:
            logger.info("Starting feature selection using RandomForest")
            X = df.drop(columns=["booking_status"])
            y = df["booking_status"]

            model = RandomForestClassifier(random_state=42)
            model.fit(X, y)

            feature_importance = model.feature_importances_
            feature_importance_df = pd.DataFrame(
                {"feature": X.columns, "importance": feature_importance}
            )

            feature_importance_df = feature_importance_df.sort_values(
                by="importance", ascending=False
            )
            num_features_to_select = self.config["data_processing"]["no_of_features"]
            selected_features = (
                feature_importance_df["feature"].head(num_features_to_select).values
            )

            selected_df = df[selected_features.tolist() + ["booking_status"]]

            logger.info("Feature selection completed successfully")
            return selected_df

        except Exception as e:
            logger.error("Error occurred while selecting features")
            raise CustomException("Error during feature selection", e)

    def save_data(self, df, file_path):
        try:
            logger.info(f"Saving processed data to {file_path}")
            df.to_csv(file_path, index=False)
            logger.info("Data saved successfully")

        except Exception as e:
            logger.error(f"Error during saving data: {e}")
            raise CustomException("Error while saving data", e)

    def process(self):
        try:
            logger.info("Loading data from raw directory")

            train_df = load_data(self.train_path)
            test_df = load_data(self.test_path)

            train_df = self.preprocessing_data(train_df)
            test_df = self.preprocessing_data(test_df)

            train_df = self.balance_data(train_df)

            train_df = self.select_feature(train_df)
            test_df = test_df[train_df.columns]

            self.save_data(train_df, PROCESSED_TRAIN_DATA_PATH)
            self.save_data(test_df, PROCESSSED_TEST_DATA_PATH)

            logger.info("Data preprocessing completed successfully")

        except Exception as e:
            logger.error("Error during preprocessing pipeline")
            raise CustomException("Error while preprocessing", e)


if __name__ == "__main__":
    processor = DataProcessor(
        TRAIN_FILE_PATH, TEST_FILE_PATH, PROCESSED_DIR, CONFIG_PATH
    )
    processor.process()
