import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

def train_and_save_model(data_path="flight_dataset.csv", model_path="model.pkl", meta_path="model_metadata.pkl"):
    print("🚀 Initializing Optimized Pipeline Training Engine...")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Core asset mismatch: '{data_path}' could not be located.")
        
    df = pd.read_csv(data_path)
    df.columns = df.columns.str.strip()
    
    # Remove granular flight column to reduce dimensionality and ensure fast training
    if 'flight' in df.columns:
        df = df.drop(columns=['flight'])
    
    target_col = "price"
    if target_col not in df.columns:
        raise ValueError(f"Target vector '{target_col}' missing from target dataframe.")
        
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    num_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    cat_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(transformers=[
        ('num', num_transformer, numerical_cols),
        ('cat', cat_transformer, categorical_cols)
    ])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    models = {
        "XGBoost Regressor": XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=50, max_depth=12, random_state=42, n_jobs=-1)
    }
    
    best_model_name = None
    best_score = -float('inf')
    best_pipeline = None
    best_metrics = {}
    
    for name, model in models.items():
        pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', model)])
        pipeline.fit(X_train, y_train)
        
        preds = pipeline.predict(X_test)
        r2 = r2_score(y_test, preds)
        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        
        if r2 > best_score:
            best_score = r2
            best_model_name = name
            best_pipeline = pipeline
            best_metrics = {"R2": r2, "MAE": mae, "RMSE": rmse}
            
    joblib.dump(best_pipeline, model_path)
    
    unique_cat_values = {col: df[col].dropna().unique().tolist() for col in categorical_cols}
    
    metadata = {
        "training_columns": X.columns.tolist(),
        "categorical_columns": categorical_cols,
        "numerical_columns": numerical_cols,
        "best_model_name": best_model_name,
        "best_score": best_score,
        "metrics": best_metrics,
        "unique_values": unique_cat_values
    }
    joblib.dump(metadata, meta_path)
    print("💾 Speed-optimized artifacts compiled and exported.")
    return metadata

if __name__ == "__main__":
    train_and_save_model()
