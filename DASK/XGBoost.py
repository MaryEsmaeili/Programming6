import time
import pandas as pd
from xgboost import XGBClassifier, DMatrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


class ModelTrainer:
    def __init__(self, X_train, y_train, X_test, y_test):
        """
        Initialize the trainer with training and testing datasets.
        Automatically preprocesses data.
        """
        # Convert Dask Arrays to NumPy if necessary
        if hasattr(X_train, 'compute'):
            X_train = X_train.compute()
        if hasattr(y_train, 'compute'):
            y_train = y_train.compute()
        if hasattr(X_test, 'compute'):
            X_test = X_test.compute()
        if hasattr(y_test, 'compute'):
            y_test = y_test.compute()

        # Ensure target variables are 1D
        self.y_train = y_train.ravel()
        self.y_test = y_test.ravel()

        # Preprocess the features
        self.X_train = self.preprocess_features(X_train)
        self.X_test = self.preprocess_features(X_test)

        # Initialize the model
        self.model = XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)

        print("\nModelTrainer Initialized Successfully!")

    def preprocess_features(self, X):
        """
        Convert categorical variables to numeric and ensure all columns are float.
        """
        if isinstance(X, pd.DataFrame):
            # Identify categorical columns
            categorical_cols = X.select_dtypes(include=["string[pyarrow]", "object"]).columns

            if len(categorical_cols) > 0:
                print(f"Found categorical columns: {list(categorical_cols)}")

                # Apply Label Encoding for categorical columns
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                for col in categorical_cols:
                    X[col] = le.fit_transform(X[col].astype(str))

        # Convert all columns to float32
        X = X.astype("float32")

        print(f"Feature preprocessing completed. Shape: {X.shape}")
        return X

    def train_model(self):
        """
        Train the XGBoost model using the preprocessed training data.
        """
        start_time = time.time()
        self.model.fit(self.X_train, self.y_train)
        training_time = time.time() - start_time

        print(f"Training completed in {training_time:.4f} seconds")
        return training_time

    # Compute evaluation metrics
    def evaluate_model(self):
        print("\nEvaluating Model...")
        start = time.time()

        # Predict probabilities
        probs = self.model.predict(self.X_test)

        # Convert to binary predictions
        preds = (probs > 0.5).astype(int)
        end = time.time()
        print(f"Inference completed in {end - start:.4f} seconds")

        # Compute metrics
        accuracy = accuracy_score(self.y_test, preds)
        precision = precision_score(self.y_test, preds)
        recall = recall_score(self.y_test, preds)
        f1 = f1_score(self.y_test, preds)
        auc = roc_auc_score(self.y_test, probs)

        print("\n-- Model Evaluation Metrics --")
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1 Score:  {f1:.4f}")
        print(f"AUC-ROC:   {auc:.4f}")

        return {
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
            "AUC-ROC": auc
        }
