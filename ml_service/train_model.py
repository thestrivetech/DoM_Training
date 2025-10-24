#!/usr/bin/env python3
import sys
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
from pathlib import Path
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv
import os

def fetch_data_from_supabase():
    """Fetch property listings data from Supabase database"""
    # Load environment variables
    load_dotenv()

    # Get database URL from environment
    db_url = os.getenv('REAL_ESTATE_DIRECT_URL')

    if not db_url:
        raise ValueError("REAL_ESTATE_DIRECT_URL not found in environment variables")

    try:
        # Connect to the database
        conn = psycopg2.connect(db_url)

        # Simple query - fetch only basic features from inactive_listings
        # Calculate mean_days_on_market_by_zip as window function
        # Calculate propertyAge from yearBuilt
        query = """
            SELECT
                il."zipCode" as zip_code,
                AVG(il."daysOnMarket") OVER (PARTITION BY il."zipCode") as mean_days_on_market_by_zip,
                il.price,
                il."squareFootage" as square_footage,
                il."distToDowntown" as dist_to_downtown,
                il.bedrooms,
                il.bathrooms,
                il."propertyType" as property_type,
                EXTRACT(YEAR FROM CURRENT_DATE) - il."yearBuilt" as property_age,
                il."lotSize" as lot_size,
                il."isPeakSeason" as is_peak_season,
                il."daysIntoYear" as days_into_year,
                il."daysOnMarket" as days_on_market
            FROM inactive_listings il
            WHERE il."daysOnMarket" IS NOT NULL
            AND il.price IS NOT NULL
            AND il.bedrooms IS NOT NULL
            AND il."squareFootage" IS NOT NULL
            AND il."yearBuilt" IS NOT NULL
        """

        # Fetch data into a pandas DataFrame
        df = pd.read_sql_query(query, conn)

        # Close the connection
        conn.close()

        print(f"Successfully fetched {len(df)} sold property records from inactive_listings table", file=sys.stderr)

        return df

    except Exception as e:
        raise Exception(f"Failed to fetch data from Supabase: {str(e)}")

class RealEstateMLModel:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = []
        self.target_name = 'days_on_market'
        self.use_log_target = True  # Re-enabled: log transformation helps with right-skewed distribution

    def preprocess_data(self, df, is_training=True):
        """Simplified preprocessing for basic features only"""
        # Make a copy to avoid modifying original
        df = df.copy()

        # NOTE: Outlier capping is now done in train() before split to avoid log transformation issues

        # Identify numeric and categorical columns
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_columns = df.select_dtypes(include=['object']).columns.tolist()

        # Remove target if it's in the lists
        if self.target_name in numeric_columns:
            numeric_columns.remove(self.target_name)

        # Handle missing values
        for col in numeric_columns:
            df[col].fillna(df[col].median(), inplace=True)

        for col in categorical_columns:
            df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown', inplace=True)

        # Encode categorical variables (only property_type in our basic feature set)
        for col in categorical_columns:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                df[col] = self.label_encoders[col].fit_transform(df[col].astype(str))
            else:
                try:
                    df[col] = self.label_encoders[col].transform(df[col].astype(str))
                except ValueError:
                    # Handle unseen categories in test set
                    df[col] = self.label_encoders[col].transform([self.label_encoders[col].classes_[0]] * len(df))

        # Convert to numeric only (keep zip_code for now, we'll drop it later if needed)
        df = df.select_dtypes(include=[np.number])

        return df

    def train(self, data):
        """Train the model on the provided data"""
        # Convert to DataFrame if it's a list of dicts
        if isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            df = data

        # Check if target variable exists
        if self.target_name not in df.columns:
            raise ValueError(f"Target variable '{self.target_name}' not found in data")

        # Remove rows where target is null
        df = df[df[self.target_name].notna()].copy()

        if len(df) < 10:
            raise ValueError("Not enough data to train the model (need at least 10 samples)")

        # Separate features and target
        y = df[self.target_name].copy()
        X = df.drop(self.target_name, axis=1)

        # Drop ID columns and other non-predictive columns
        cols_to_drop = [col for col in X.columns if col.lower() in ['id', 'index', 'address', 'status']]
        X = X.drop(cols_to_drop, axis=1, errors='ignore')

        # APPLY OUTLIER CAPPING BEFORE SPLIT (to avoid data leakage, we cap on full dataset)
        # This must happen BEFORE log transformation
        days_95th = y.quantile(0.95)
        print(f"Capping days_on_market at 95th percentile: {days_95th:.0f} days", file=sys.stderr)
        y = y.clip(upper=days_95th)

        # Also cap price in X
        if 'price' in X.columns:
            price_95th = X['price'].quantile(0.95)
            print(f"Capping price at 95th percentile: ${price_95th:,.0f}", file=sys.stderr)
            X['price'] = X['price'].clip(upper=price_95th)

        # Split data FIRST to avoid data leakage
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Apply log transformation to target if enabled
        if self.use_log_target:
            print("Applying log transformation to target variable...", file=sys.stderr)
            y_train_original = y_train.copy()  # Keep for reference
            y_test_original = y_test.copy()
            y_train = np.log1p(y_train)
            y_test = np.log1p(y_test)

        # Preprocess training data - no outlier capping here (already done above)
        df_train = X_train.copy()
        df_train[self.target_name] = y_train
        df_train_preprocessed = self.preprocess_data(df_train, is_training=True)

        # Now preprocess test data
        df_test = X_test.copy()
        df_test[self.target_name] = y_test
        df_test_preprocessed = self.preprocess_data(df_test, is_training=False)

        # Separate features and target
        X_train = df_train_preprocessed.drop(self.target_name, axis=1, errors='ignore')
        X_test = df_test_preprocessed.drop(self.target_name, axis=1, errors='ignore')

        # Drop zip_code - we only need it for aggregation, not as a feature
        X_train = X_train.drop('zip_code', axis=1, errors='ignore')
        X_test = X_test.drop('zip_code', axis=1, errors='ignore')

        # Use all available features (simplified model)
        self.feature_names = X_train.columns.tolist()
        print(f"\nUsing all {len(self.feature_names)} basic features", file=sys.stderr)
        print(f"Features: {', '.join(self.feature_names)}", file=sys.stderr)

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # HYPERPARAMETER TUNING - COMMENTED OUT FOR FASTER TESTING
        # Uncomment this section when you want to tune hyperparameters
        # print("\nPerforming hyperparameter tuning...", file=sys.stderr)

        # # Focused grid search for RandomForest (optimized for speed)
        # rf_param_grid = {
        #     'n_estimators': [250, 300],  # Test only 2 values
        #     'max_depth': [15, 20],        # Test only 2 values
        # }
        # rf_base = RandomForestRegressor(
        #     random_state=42,
        #     n_jobs=-1,
        #     min_samples_split=5,  # Fixed at good default
        #     min_samples_leaf=2    # Fixed at good default
        # )
        # rf_grid = GridSearchCV(
        #     rf_base,
        #     rf_param_grid,
        #     cv=2,  # Reduced from 3 to 2 folds
        #     scoring='neg_mean_absolute_error',
        #     n_jobs=-1,
        #     verbose=1
        # )
        # print("Tuning RandomForest...", file=sys.stderr)
        # rf_grid.fit(X_train_scaled, y_train)
        # rf_model = rf_grid.best_estimator_
        # print(f"Best RF params: {rf_grid.best_params_}", file=sys.stderr)

        # # Focused grid for XGBoost (optimized for speed)
        # xgb_param_grid = {
        #     'max_depth': [6, 8],           # Test only 2 values
        #     'learning_rate': [0.05, 0.1],  # Test only 2 values
        # }
        # xgb_base = XGBRegressor(
        #     random_state=42,
        #     n_jobs=-1,
        #     n_estimators=200,  # Fixed at good default
        #     subsample=0.8      # Fixed at good default
        # )
        # xgb_grid = GridSearchCV(
        #     xgb_base,
        #     xgb_param_grid,
        #     cv=2,  # Reduced from 3 to 2 folds
        #     scoring='neg_mean_absolute_error',
        #     n_jobs=-1,
        #     verbose=1
        # )
        # print("Tuning XGBoost...", file=sys.stderr)
        # xgb_grid.fit(X_train_scaled, y_train)
        # xgb_model = xgb_grid.best_estimator_
        # print(f"Best XGB params: {xgb_grid.best_params_}", file=sys.stderr)

        # # Focused grid for LightGBM (optimized for speed)
        # lgbm_param_grid = {
        #     'n_estimators': [200, 250, 300],  # Test 3 values
        #     'num_leaves': [31, 50]             # Test 2 values
        # }
        # lgbm_base = LGBMRegressor(
        #     random_state=42,
        #     n_jobs=-1,
        #     verbose=-1,
        #     max_depth=10,        # Fixed at good default
        #     learning_rate=0.05,  # Fixed at good default
        #     enable_categorical=False  # Suppress feature name warnings
        # )
        # lgbm_grid = GridSearchCV(
        #     lgbm_base,
        #     lgbm_param_grid,
        #     cv=2,  # Reduced from 3 to 2 folds
        #     scoring='neg_mean_absolute_error',
        #     n_jobs=-1,
        #     verbose=1
        # )
        # print("Tuning LightGBM...", file=sys.stderr)
        # lgbm_grid.fit(X_train_scaled, y_train)
        # lgbm_model = lgbm_grid.best_estimator_
        # print(f"Best LGBM params: {lgbm_grid.best_params_}", file=sys.stderr)

        # USE BEST PARAMETERS FOUND FROM PREVIOUS TUNING (REDUCED FOR SMALLER MODEL SIZE)
        print("\nUsing pre-tuned hyperparameters (fast mode, reduced estimators)...", file=sys.stderr)

        # RandomForest with best params: max_depth=15, n_estimators=100 (reduced from 300)
        rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )

        # XGBoost with best params: learning_rate=0.05, max_depth=6, n_estimators=100 (reduced from 200)
        xgb_model = XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42,
            n_jobs=-1
        )

        # LightGBM with best params: n_estimators=100 (reduced from 200), num_leaves=31
        lgbm_model = LGBMRegressor(
            n_estimators=100,
            num_leaves=31,
            max_depth=10,
            learning_rate=0.05,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
            enable_categorical=False
        )

        # GradientBoosting with good default params, n_estimators=100 (reduced from 200)
        gb_model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=7,
            learning_rate=0.1,
            min_samples_split=5,
            random_state=42
        )
        gb_model.fit(X_train_scaled, y_train)

        # Train individual models and evaluate
        individual_models = {
            'RandomForest': rf_model,
            'GradientBoosting': gb_model,
            'XGBoost': xgb_model,
            'LightGBM': lgbm_model
        }

        results = {}

        for name, model in individual_models.items():
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)

            # Inverse transform predictions if using log target
            if self.use_log_target:
                y_pred = np.expm1(y_pred)
                y_test_eval = y_test_original
            else:
                y_test_eval = y_test

            mae = mean_absolute_error(y_test_eval, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test_eval, y_pred))
            r2 = r2_score(y_test_eval, y_pred)

            results[name] = {
                'mae': mae,
                'rmse': rmse,
                'r2': r2
            }

        # ENSEMBLE STACKING
        # Create stacking ensemble with Ridge meta-learner
        print("\nCreating stacking ensemble model...", file=sys.stderr)

        # StackingRegressor: Base models train on full data, meta-learner learns to combine them
        ensemble = StackingRegressor(
            estimators=[
                ('rf', rf_model),
                ('gb', gb_model),
                ('xgb', xgb_model),
                ('lgbm', lgbm_model)
            ],
            final_estimator=Ridge(alpha=1.0),  # Ridge regression as meta-learner
            cv=5  # 5-fold CV to generate meta-features
        )

        # Train ensemble
        ensemble.fit(X_train_scaled, y_train)
        y_pred_ensemble = ensemble.predict(X_test_scaled)

        # Inverse transform ensemble predictions if using log target
        if self.use_log_target:
            y_pred_ensemble = np.expm1(y_pred_ensemble)
            y_test_eval = y_test_original
        else:
            y_test_eval = y_test

        mae_ensemble = mean_absolute_error(y_test_eval, y_pred_ensemble)
        rmse_ensemble = np.sqrt(mean_squared_error(y_test_eval, y_pred_ensemble))
        r2_ensemble = r2_score(y_test_eval, y_pred_ensemble)

        results['Ensemble'] = {
            'mae': mae_ensemble,
            'rmse': rmse_ensemble,
            'r2': r2_ensemble
        }

        # Use ensemble as the final model
        self.model = ensemble
        best_model_name = 'Ensemble'
        best_score = r2_ensemble

        print(f"Ensemble MAE: {mae_ensemble:.2f} days (targeting < individual models)", file=sys.stderr)

        # Get feature importance (from RandomForest component of ensemble)
        # VotingRegressor doesn't have feature_importances_, so we use RF's
        if hasattr(rf_model, 'feature_importances_'):
            importances = rf_model.feature_importances_
            feature_importance = sorted(
                zip(self.feature_names, importances),
                key=lambda x: x[1],
                reverse=True
            )
        else:
            feature_importance = []

        # Save model with compression
        model_dir = Path(__file__).parent / 'models'
        model_dir.mkdir(exist_ok=True)

        # Use compress=3 for good compression (0-9, higher = smaller but slower)
        joblib.dump(self.model, model_dir / 'model.pkl', compress=3)
        joblib.dump(self.scaler, model_dir / 'scaler.pkl', compress=3)
        joblib.dump(self.label_encoders, model_dir / 'label_encoders.pkl', compress=3)
        joblib.dump(self.feature_names, model_dir / 'feature_names.pkl', compress=3)

        # Save zip code statistics if they exist
        if hasattr(self, 'zip_stats'):
            joblib.dump(self.zip_stats, model_dir / 'zip_stats.pkl', compress=3)
            print(f"Saved zip code statistics for {len(self.zip_stats)} zip codes", file=sys.stderr)

        return {
            'status': 'success',
            'best_model': best_model_name,
            'metrics': results[best_model_name],
            'all_results': results,
            'feature_importance': [
                {'feature': feat, 'importance': float(imp)}
                for feat, imp in feature_importance[:15]
            ],
            'training_samples': len(df)
        }

def main():
    try:
        # Check if we should fetch from database or use a file
        use_database = True
        data = None

        # If a file path is provided, use it instead
        if len(sys.argv) >= 2 and sys.argv[1] != '--database':
            use_database = False
            data_file = sys.argv[1]
            with open(data_file, 'r') as f:
                data = json.load(f)
            print(f"Using data from file: {data_file}", file=sys.stderr)
        else:
            # Fetch data from Supabase database
            print("Fetching data from Supabase database...", file=sys.stderr)
            data = fetch_data_from_supabase()

        # Train model
        ml_model = RealEstateMLModel()
        result = ml_model.train(data)

        # Print formatted output
        print("\n" + "="*70)
        print("  REAL ESTATE MODEL TRAINING RESULTS")
        print("="*70)
        print(f"\nStatus: {result['status'].upper()}")
        print(f"Training Samples: {result['training_samples']:,}")
        print(f"\nBest Model: {result['best_model']}")

        print("\n" + "-"*70)
        print("  BEST MODEL PERFORMANCE METRICS")
        print("-"*70)
        metrics = result['metrics']
        print(f"  Mean Absolute Error (MAE):  {metrics['mae']:.2f} days")
        print(f"  Root Mean Squared Error:    {metrics['rmse']:.2f} days")
        print(f"  R² Score:                   {metrics['r2']:.4f} ({metrics['r2']*100:.2f}%)")

        print("\n" + "-"*70)
        print("  ALL MODEL COMPARISONS")
        print("-"*70)
        for model_name, model_metrics in result['all_results'].items():
            marker = " *" if model_name == result['best_model'] else ""
            print(f"\n  {model_name}{marker}")
            print(f"    MAE:  {model_metrics['mae']:.2f} days")
            print(f"    RMSE: {model_metrics['rmse']:.2f} days")
            print(f"    R2:   {model_metrics['r2']:.4f} ({model_metrics['r2']*100:.2f}%)")

        print("\n" + "-"*70)
        print("  TOP 15 FEATURE IMPORTANCE")
        print("-"*70)
        for i, feat in enumerate(result['feature_importance'], 1):
            bar_length = int(feat['importance'] * 50)
            bar = "#" * bar_length
            print(f"  {i:2d}. {feat['feature']:20s} {feat['importance']*100:6.2f}% {bar}")

        print("\n" + "="*70)
        print("  Model saved to: ml_service/models/")
        print("="*70 + "\n")

        # Also output JSON for programmatic use
        with open(Path(__file__).parent / 'models' / 'training_results.json', 'w') as f:
            json.dump(result, f, indent=2)

        sys.exit(0)

    except Exception as e:
        print(json.dumps({'error': str(e)}), file=sys.stderr)
        sys.exit(1)



if __name__ == '__main__':
    main()
