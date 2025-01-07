import sqlite3
import pandas as pd
import json
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
import miceforest as mf
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, precision_recall_curve,precision_score, accuracy_score, recall_score, f1_score
import numpy as np


class ModelRunner:
    def __init__(self, X, y, categorical_columns=None, test_size=0.2, random_state=42):
        # Perform train-test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        self.categorical_columns = categorical_columns
        self.encoder = OneHotEncoder()
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=0.95)
        self.results = {}

    def preprocess(self, X_train, X_val):
        if self.categorical_columns:
            # One-hot encode categorical features
            encoded_train = self.encoder.fit_transform(X_train[self.categorical_columns])
            encoded_val = self.encoder.transform(X_val[self.categorical_columns])
            
            column_names = self.encoder.get_feature_names_out(self.categorical_columns)
            encoded_train_df = pd.DataFrame(encoded_train.toarray(), columns=column_names, index=X_train.index)
            encoded_val_df = pd.DataFrame(encoded_val.toarray(), columns=column_names, index=X_val.index)

            X_train = pd.concat([X_train.drop(columns=self.categorical_columns), encoded_train_df], axis=1)
            X_val = pd.concat([X_val.drop(columns=self.categorical_columns), encoded_val_df], axis=1)

        X_train.reset_index(drop=True, inplace=True)
        X_val.reset_index(drop=True, inplace=True)

        # Apply MICE imputation
        kernel = mf.ImputationKernel(
            X_train,
            num_datasets=4,
            random_state=1
        )
        kernel.mice(iterations=2)
        X_train_filled = kernel.complete_data(dataset=3)
        X_val_filled = kernel.transform(X_val)

        # Standardize numerical features
        X_train_scaled = self.scaler.fit_transform(X_train_filled)
        X_val_scaled = self.scaler.transform(X_val_filled)

        # Apply PCA for dimensionality reduction
        X_train_pca = self.pca.fit_transform(X_train_scaled)
        X_val_pca = self.pca.transform(X_val_scaled)

        return X_train_pca, X_val_pca

    def train_and_evaluate(self, model, model_name):
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        acc, prec, rec, f1 = [], [], [], []

        for train_idx, val_idx in kf.split(self.X_train):
            X_train_fold, X_val_fold = self.X_train.iloc[train_idx], self.X_train.iloc[val_idx]
            y_train_fold, y_val_fold = self.y_train.iloc[train_idx], self.y_train.iloc[val_idx]

            # Preprocess the data
            X_train_pca, X_val_pca = self.preprocess(X_train_fold, X_val_fold)

            # Train the model
            model.fit(X_train_pca, y_train_fold)

            # Predict and evaluate
            preds = model.predict(X_val_pca)
            acc.append(accuracy_score(y_val_fold, preds))
            prec.append(precision_score(y_val_fold, preds, average='micro'))
            rec.append(recall_score(y_val_fold, preds, average='micro'))
            f1.append(f1_score(y_val_fold, preds, average='micro'))

        # Store results
        self.results[model_name] = {
            'accuracy': sum(acc) / len(acc),
            'precision': sum(prec) / len(prec),
            'recall': sum(rec) / len(rec),
            'f1_score': sum(f1) / len(f1)
        }

    def test_model(self, model):
        # Preprocess test data
        X_test_pca, _ = self.preprocess(self.X_test, self.X_test)

        # Predict and evaluate on test data
        preds = model.predict(X_test_pca)
        test_results = {
            'accuracy': accuracy_score(self.y_test, preds),
            'precision': precision_score(self.y_test, preds, average='micro'),
            'recall': recall_score(self.y_test, preds, average='micro'),
            'f1_score': f1_score(self.y_test, preds, average='micro')
        }
        return test_results

    def get_results(self):
        return self.results

# Example usage
if __name__ == "__main__":
    # Path to the merged database
    clean_db_path = "data/cleaned_data.db"

    # Establish a connection to the database
    conn = sqlite3.connect(clean_db_path)

    # Read the merged data into a DataFrame
    cleaned_df = pd.read_sql_query("SELECT * FROM cleaned_data", conn)

    # Close the connection
    conn.close()

    # Load category metadata
    with open("data/category_metadata.json", "r") as f:
        category_metadata = json.load(f)

    # Convert columns back to categorical
    for column, categories in category_metadata.items():
        cleaned_df[column] = pd.Categorical(cleaned_df[column], categories=categories, ordered=True)

    # Convert ordinal categorical columns to their numerical codes
    cleaned_df = cleaned_df.apply(lambda col: col.cat.codes if col.dtype.name == 'category' else col)

    cleaned_df['date'] = pd.to_datetime(cleaned_df['date'])
    cleaned_df['month'] = cleaned_df['date'].dt.month

    X = cleaned_df.drop(['Daily Solar Panel Efficiency',  
                            'date', 
                            'data_ref', 
                            'psi_north', 'psi_south', 'psi_east','psi_west', 'psi_central', 
                            'pm25_north', 'pm25_south', 'pm25_east','pm25_west',
                            'Wet Bulb Temperature (deg F)', 'Air Pressure (hPa)', 'Dew Point Category'
                            ], axis=1)

    y = cleaned_df['Daily Solar Panel Efficiency']

    # Assuming X and y are already defined and preprocessed
    categorical_columns = ['Wind Direction']  # Replace with your actual categorical columns

    # Instantiate the ModelRunner
    runner = ModelRunner(X, y, categorical_columns)

    # Define models to test
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "SVM": SVC(random_state=42)
    }

    # Train and evaluate each model
    for model_name, model in models.items():
        runner.train_and_evaluate(model, model_name)
        print(f"Train Results for {model_name}:")
        print(runner.get_results()[model_name])

#Train Results for Logistic Regression:
#{'accuracy': 0.7237141811609897, 'precision': 0.7237141811609897, 'recall': 0.7237141811609897, 'f1_score': 0.7237141811609897}
#Train Results for Decision Tree:
#{'accuracy': 0.6510802595908978, 'precision': 0.6510802595908978, 'recall': 0.6510802595908978, 'f1_score': 0.6510802595908978}
#Train Results for SVM:
#{'accuracy': 0.7573303062664765, 'precision': 0.7573303062664765, 'recall': 0.7573303062664765, 'f1_score': 0.7573303062664765}