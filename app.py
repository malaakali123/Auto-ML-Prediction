import os
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify
import joblib

# ML Libraries
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
MODEL_FOLDER = 'models'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# --- MODULE 4, 5, 6: Data Cleaning & Processing ---
def clean_and_process_data(df, target_col):
    # 1. Drop columns with too many missing values (>50%)
    df = df.dropna(thresh=len(df)*0.5, axis=1)
    
    # 2. Separate Target
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # 3. Detect Problem Type
    if y.dtype == 'object' or len(y.unique()) < 20: 
        problem_type = 'Classification'
        le = LabelEncoder()
        y = le.fit_transform(y)
    else:
        problem_type = 'Regression'
    
    # 4. Feature Identification
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object', 'category']).columns

    # 5. Preprocessing Pipeline
    # Numeric: Impute Mean -> Scale
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    # Categorical: Impute Mode -> OneHot
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    return X, y, preprocessor, problem_type

# --- MODULE 7, 8, 9: AutoML Engine ---
def run_automl(X, y, preprocessor, problem_type):
    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    models = {}
    leaderboard = []

    if problem_type == 'Classification':
        models = {
            'Logistic Regression': LogisticRegression(max_iter=1000),
            'Random Forest': RandomForestClassifier(n_estimators=100),
            'XGBoost': xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
            'SVM': SVC(probability=True),
            'KNN': KNeighborsClassifier()
        }
    else: # Regression
        models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(n_estimators=100),
            'XGBoost': xgb.XGBRegressor(),
            'Ridge': Ridge(),
            'Lasso': Lasso()
        }

    best_model = None
    best_score = -float('inf')

    for name, model in models.items():
        # Create Pipeline
        clf = Pipeline(steps=[('preprocessor', preprocessor),
                              ('classifier', model)])
        
        # Train
        try:
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            
            metrics = {}
            score = 0
            
            if problem_type == 'Classification':
                score = accuracy_score(y_test, y_pred)
                metrics = {
                    'Accuracy': round(score, 4),
                    'F1': round(f1_score(y_test, y_pred, average='weighted'), 4),
                    'Precision': round(precision_score(y_test, y_pred, average='weighted'), 4)
                }
            else: # Regression
                score = r2_score(y_test, y_pred)
                metrics = {
                    'R2 Score': round(score, 4),
                    'RMSE': round(np.sqrt(mean_squared_error(y_test, y_pred)), 4),
                    'MAE': round(mean_absolute_error(y_test, y_pred), 4)
                }
            
            leaderboard.append({
                'model_name': name,
                'score': round(score, 4),
                'metrics': metrics
            })

            if score > best_score:
                best_score = score
                best_model = clf

        except Exception as e:
            print(f"Error training {name}: {e}")
            continue

    # Sort Leaderboard
    leaderboard = sorted(leaderboard, key=lambda x: x['score'], reverse=True)

    # Save Best Model
    if best_model:
        joblib.dump(best_model, os.path.join(MODEL_FOLDER, 'best_model.pkl'))

    # Feature Importance (if supported by best model)
    feature_importance = {}
    try:
        # Get feature names from preprocessor
        final_model = best_model.named_steps['classifier']
        
        # Check if model has feature_importances_
        if hasattr(final_model, 'feature_importances_'):
            # This is tricky with ColumnTransformer, simplifying for now
            # We will just return dummy or simple generic importance for visualization demo if extraction fails 
            # Correct extraction requires getting feature names from OneHotEncoder
            importances = final_model.feature_importances_
            # Just take top N for display to avoid index errors
            indices = np.argsort(importances)[::-1]
            for i in range(min(5, len(importances))):
                feature_importance[f"Feature_{i}"] = float(importances[indices[i]])
    except:
        pass

    return {
        'problem_type': problem_type,
        'best_model': leaderboard[0]['model_name'] if leaderboard else 'None',
        'best_score': best_score,
        'leaderboard': leaderboard,
        'feature_importance': feature_importance
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    target_col = request.form.get('target')
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        
        try:
            # Read CSV
            df = pd.read_csv(filepath)
            
            # Basic Validation
            if target_col not in df.columns:
                return jsonify({'error': f'Target column "{target_col}" not found in dataset'}), 400

            # Run AutoML Pipeline
            X, y, preprocessor, problem_type = clean_and_process_data(df, target_col)
            results = run_automl(X, y, preprocessor, problem_type)
            
            return jsonify(results)

        except Exception as e:
            return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
