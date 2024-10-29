import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance
import xgboost as xgb
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTENC

class SolarEfficiencyModel:
    def __init__(self, data, target_column='Daily Solar Panel Efficiency', timestamp_column='date'):
        self.data = data
        self.target_column = target_column
        self.timestamp_column = timestamp_column
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.selected_features = None
        self.models = {}
        self.best_models = {}
        self.meta_model = None
        self.preprocessor = None
        self.results = {}
        self.label_encoder = LabelEncoder()

    def prepare_data(self):

        # Encode target variable
        self.y = self.label_encoder.fit_transform(self.data[self.target_column])
        
        # Drop unnecessary columns
        columns_to_drop = [self.target_column, 'data_ref', self.timestamp_column, 'date']
        self.X = self.data.drop(columns_to_drop, axis=1)
        
        # Identify numeric and categorical columns
        numeric_features = self.X.select_dtypes(include=['int64', 'float64']).columns
        categorical_features = self.X.select_dtypes(include=['object']).columns
        categorical_features_idx = [self.X.columns.get_loc(col) for col in categorical_features]

        # Create preprocessing steps for numeric and categorical data
        numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        # Combine preprocessing steps
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])

        # Apply SMOTENC to tackle unbalanced data
        smote = SMOTENC(categorical_features=categorical_features_idx, random_state=42)
        self.X, self.y = smote.fit_resample(self.X, self.y)

        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )

        


    def feature_selection(self, n_features=20):
        # Fit the preprocessor and transform the data
        X_preprocessed = self.preprocessor.fit_transform(self.X)
        
        # Get feature names after preprocessing
        feature_names = (self.preprocessor.named_transformers_['num'].get_feature_names_out().tolist() +
                         self.preprocessor.named_transformers_['cat'].get_feature_names_out().tolist())

        # Perform feature selection on preprocessed data
        selector = SelectKBest(f_classif, k=n_features)
        X_selected = selector.fit_transform(X_preprocessed, self.y)
        
        # Get selected feature names
        selected_mask = selector.get_support()
        self.selected_features = [feature for feature, selected in zip(feature_names, selected_mask) if selected]
        
        print("Top 20 selected features:")
        print(self.selected_features)
        
        return self.selected_features
    


    def hyperparameter_tuning(self):
        X_train_preprocessed = self.preprocessor.fit_transform(self.X_train)

        param_grids = {
            'RandomForest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'SVM': {
                'C': [0.1, 1, 10],
                'kernel': ['rbf', 'linear'],
                'gamma': ['scale', 'auto']
            },
            'XGBoost': {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.3],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0]
            }
        }

        models = {
            'RandomForest': RandomForestClassifier(random_state=42),
            'SVM': SVC(probability=True, random_state=42),
            'XGBoost': xgb.XGBClassifier(random_state=42, eval_metric='mlogloss', enable_categorical=True)
        }

        for name, model in models.items():
            print(f"\nPerforming hyperparameter tuning for {name}...")
            grid_search = GridSearchCV(
                estimator=model,
                param_grid=param_grids[name],
                cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                scoring='accuracy',
                n_jobs=-1
            )
            grid_search.fit(X_train_preprocessed, self.y_train)
            
            self.best_models[name] = grid_search.best_estimator_
            print(f"Best parameters for {name}: {grid_search.best_params_}")
            print(f"Best cross-validation score: {grid_search.best_score_:.4f}")

    def train_models(self):
        X_train_preprocessed = self.preprocessor.transform(self.X_train)
        X_test_preprocessed = self.preprocessor.transform(self.X_test)

        for name, model in self.best_models.items():
            model.fit(X_train_preprocessed, self.y_train)
            y_pred = model.predict(X_test_preprocessed)
            y_pred_proba = model.predict_proba(X_test_preprocessed)

            self._evaluate_model(name, y_pred, y_pred_proba)

    def _evaluate_model(self, name, y_pred, y_pred_proba):
        accuracy = accuracy_score(self.y_test, y_pred)

        auc_score = roc_auc_score(self.y_test, y_pred_proba, average='macro', multi_class='ovr')
        
        # Store results in the results dictionary
        self.results[name] = {
            'Accuracy': accuracy,
            'ROC AUC Score': auc_score,
            'Classification Report': classification_report(self.y_test, y_pred, 
                                                          target_names=self.label_encoder.classes_, 
                                                          output_dict=True)
        }

        print(f"\n{name} Results:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"ROC AUC Score: {auc_score:.4f}")
        print("Classification Report:")
        print(classification_report(self.y_test, y_pred, target_names=self.label_encoder.classes_))

    def save_results_to_csv(self, filename='results/model_results.csv'):
        # Prepare data for CSV
        data = []
        for model_name, model_results in self.results.items():
            row = {
                'Model': model_name,
                'Accuracy': model_results['Accuracy'],
                'ROC AUC Score': model_results['ROC AUC Score']
            }
            # Add classification report metrics
            for class_name, metrics in model_results['Classification Report'].items():
                if isinstance(metrics, dict): 
                    for metric, value in metrics.items():
                        row[f'{class_name}_{metric}'] = value
            data.append(row)
        
        # Create DataFrame and save to CSV
        results_df = pd.DataFrame(data)
        results_df.to_csv(filename, index=False)
        print(f"Results saved to {filename}")

    def analyze_best_model_feature_importance(self, top_n=20):
        best_model_name = 'SVM'
        best_model = self.best_models[best_model_name]
        
        print(f"Best performing model: {best_model_name}")
        print(f"ROC AUC Score: {self.results[best_model_name]['ROC AUC Score']:.4f}")
        
        # Get feature names
        feature_names = (self.preprocessor.named_transformers_['num'].get_feature_names_out().tolist() +
                         self.preprocessor.named_transformers_['cat'].get_feature_names_out().tolist())
        
        # Preprocess X_test
        X_test_preprocessed = self.preprocessor.transform(self.X_test)
        
        # Calculate feature importance
        if best_model_name in ['RandomForest', 'XGBoost']:
            importances = best_model.feature_importances_
            importance_type = "Feature Importance"
        else:  
            perm_importance = permutation_importance(best_model, X_test_preprocessed, self.y_test, n_repeats=10, random_state=42)
            importances = perm_importance.importances_mean
            importance_type = "Permutation Importance"
        
        # Sort feature importances in descending order
        indices = np.argsort(importances)[::-1]
        
        # Select top 20 features
        top_indices = indices[:top_n]
        top_importances = importances[top_indices]
        top_features = [feature_names[i] for i in top_indices]
        
        # Print feature importances
        print(f"\nTop {top_n} {importance_type}:")
        for i, feature in enumerate(top_features):
            print(f"{feature}: {top_importances[i]:.4f}")
        
        # Save feature importances
        plt.figure(figsize=(15, 10))
        plt.title(f"Top {top_n} {importance_type} for {best_model_name}")
        plt.bar(range(top_n), top_importances)
        plt.xlabel("Features")
        plt.ylabel(importance_type)
        plt.xticks(range(top_n), top_features, rotation=90)
        plt.tight_layout()
        plt.savefig("results/feature_importance.png")
    

    def run(self):
        self.prepare_data()
        self.feature_selection()
        self.hyperparameter_tuning()
        self.train_models()
        self.save_results_to_csv()
        self.analyze_best_model_feature_importance()
