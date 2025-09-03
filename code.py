import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report

# Load the dataset
try:
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data'
    columns = [
        'checking_account_status', 'duration', 'credit_history', 'purpose', 'credit_amount', 
        'savings_account', 'present_employment', 'installment_rate', 'personal_status_sex', 
        'other_debtors', 'present_residence_since', 'property', 'age', 'other_installment_plans', 
        'housing', 'number_of_credits', 'job', 'people_liable', 'telephone', 'foreign_worker', 'risk'
    ]
    df = pd.read_csv(url, sep=' ', header=None, names=columns)
except Exception as e:
    print(f"Error loading data: {e}")
    df = pd.DataFrame() # Create an empty dataframe to avoid further errors

if not df.empty:
    # --- Data Preprocessing ---
    # Convert target variable to binary (0 for Good, 1 for Bad)
    df['risk'] = df['risk'].replace({1: 0, 2: 1})

    # Identify categorical and numerical features
    categorical_features = df.select_dtypes(include=['object']).columns
    numerical_features = df.select_dtypes(include=np.number).drop(columns=['risk']).columns

    # Define preprocessing pipelines for numerical and categorical features
    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    # Create a preprocessor object using ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    # --- Train-Test Split ---
    X = df.drop('risk', axis=1)
    y = df['risk']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # --- Model Training and Evaluation ---
    models = {
        'Logistic Regression': LogisticRegression(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42)
    }

    results = {}

    for model_name, model in models.items():
        # Create a pipeline that preprocesses the data and then trains the model
        pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                   ('classifier', model)])
        
        # Train the model
        pipeline.fit(X_train, y_train)
        
        # Make predictions
        y_pred = pipeline.predict(X_test)
        y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
        
        # Evaluate the model
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        results[model_name] = {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-score': f1,
            'ROC-AUC': roc_auc
        }
        
        print(f"--- {model_name} ---")
        print(classification_report(y_test, y_pred))
        print(f"ROC-AUC Score: {roc_auc:.4f}\n")

    # --- Hyperparameter Tuning (Example with Random Forest) ---
    print("--- Hyperparameter Tuning for Random Forest ---")
    
    # Define the parameter grid
    param_grid = {
        'classifier__n_estimators': [100, 200, 300],
        'classifier__max_depth': [None, 10, 20, 30],
        'classifier__min_samples_split': [2, 5, 10]
    }
    
    # Create the pipeline for Random Forest
    rf_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                  ('classifier', RandomForestClassifier(random_state=42))])

    # Create the GridSearchCV object
    grid_search = GridSearchCV(rf_pipeline, param_grid, cv=5, n_jobs=-1, scoring='roc_auc')
    
    # Fit the grid search to the data
    grid_search.fit(X_train, y_train)
    
    print(f"Best parameters found: {grid_search.best_params_}")
    
    # Evaluate the best model from grid search
    best_model = grid_search.best_estimator_
    y_pred_best = best_model.predict(X_test)
    y_pred_proba_best = best_model.predict_proba(X_test)[:, 1]
    
    print("\n--- Tuned Random Forest Evaluation ---")
    print(classification_report(y_test, y_pred_best))
    print(f"ROC-AUC Score: {roc_auc_score(y_test, y_pred_proba_best):.4f}")

    # --- Feature Importance from the best model ---
    try:
        # Get feature names after one-hot encoding
        ohe_feature_names = best_model.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(categorical_features)
        all_feature_names = np.concatenate([numerical_features, ohe_feature_names])
        
        # Get feature importances
        importances = best_model.named_steps['classifier'].feature_importances_
        feature_importance_df = pd.DataFrame({'feature': all_feature_names, 'importance': importances})
        feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)
        
        print("\n--- Top 10 Feature Importances ---")
        print(feature_importance_df.head(10))

        # Plot feature importances
        plt.figure(figsize=(12, 8))
        sns.barplot(x='importance', y='feature', data=feature_importance_df.head(10))
        plt.title('Top 10 Feature Importances from Tuned Random Forest')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        print("\nFeature importance plot saved as 'feature_importance.png'")

    except Exception as e:
        print(f"\nCould not generate feature importances: {e}")

else:
    print("Could not proceed with analysis as the dataframe is empty.")
