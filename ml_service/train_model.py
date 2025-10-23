#!/usr/bin/env python3
import sys
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from xgboost import XGBRegressor
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

        # Query to fetch sold property listings from inactive_listings
        # WITH active listing count at the time of listing
        query = """
            WITH active_counts AS (
                SELECT
                    il."zipCode" as zip_code,
                    il."listedDate" as list_date,
                    COUNT(DISTINCT hl.id) as active_listings_at_time
                FROM inactive_listings il
                LEFT JOIN historical_listings hl ON
                    hl."zipCode" = il."zipCode"
                    AND hl."listedDate" <= il."listedDate"
                    AND (hl."removedDate" IS NULL OR hl."removedDate" >= il."listedDate")
                WHERE il."daysOnMarket" IS NOT NULL
                GROUP BY il."zipCode", il."listedDate"
            )
            SELECT
                il.id,
                il."formattedAddress" as formatted_address,
                il.city,
                il.state,
                il."zipCode" as zip_code,
                il.price,
                il.bedrooms,
                il.bathrooms,
                il."squareFootage" as square_footage,
                il."lotSize" as lot_size,
                il."yearBuilt" as year_built,
                il."propertyType" as property_type,
                il."listingType" as listing_type,
                il.status,
                il."listedDate" as listing_date,
                il."removedDate" as removed_date,
                il."daysOnMarket" as days_on_market,
                il.latitude,
                il.longitude,
                il.county,
                il."propertyAge" as property_age,
                il."distToDowntown" as dist_to_downtown,
                il."daysIntoYear" as days_into_year,
                il."isPeakSeason" as is_peak_season,
                COALESCE(ac.active_listings_at_time, 0) as active_listings_at_time
            FROM inactive_listings il
            LEFT JOIN active_counts ac ON
                ac.zip_code = il."zipCode"
                AND ac.list_date = il."listedDate"
            WHERE il."daysOnMarket" IS NOT NULL
            AND il.price IS NOT NULL
            AND il.bedrooms IS NOT NULL
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

    def preprocess_data(self, df, is_training=True):
        """Preprocess the real estate data"""
        # Make a copy to avoid modifying original
        df = df.copy()

        # Handle outliers BEFORE creating features
        if 'days_on_market' in df.columns and is_training:
            # Cap days_on_market at 99th percentile to remove extreme outliers
            days_99th = df['days_on_market'].quantile(0.99)
            print(f"Capping days_on_market at 99th percentile: {days_99th:.0f} days", file=sys.stderr)
            df['days_on_market'] = df['days_on_market'].clip(upper=days_99th)

        if 'price' in df.columns and is_training:
            # Cap price at 99th percentile
            price_99th = df['price'].quantile(0.99)
            print(f"Capping price at 99th percentile: ${price_99th:,.0f}", file=sys.stderr)
            df['price'] = df['price'].clip(upper=price_99th)

        # Remove low-value features EARLY before encoding
        low_value_cols = ['formatted_address', 'city', 'state', 'county', 'removed_date']
        df.drop([col for col in low_value_cols if col in df.columns], axis=1, inplace=True, errors='ignore')

        # IMPUTE MISSING listing_type
        if 'listing_type' in df.columns:
            # Infer New Construction from property age and price
            if 'property_age' in df.columns and 'price' in df.columns:
                # New construction typically: age < 2 years, price above median
                median_price = df['price'].median()
                mask_new = (df['property_age'] <= 2) & (df['price'] >= median_price) & df['listing_type'].isna()
                df.loc[mask_new, 'listing_type'] = 'New Construction'

            # Everything else is Standard
            df['listing_type'].fillna('Standard', inplace=True)
            print(f"Imputed listing_type: {df['listing_type'].value_counts().to_dict()}", file=sys.stderr)

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

        # ZIP CODE-BASED AGGREGATE FEATURES (Training only - we'll store these for prediction)
        if 'zip_code' in df.columns and is_training and self.target_name in df.columns:
            print("Creating zip code-based aggregate features...", file=sys.stderr)

            # Calculate zip code statistics
            zip_stats = df.groupby('zip_code').agg({
                self.target_name: ['median', 'mean', 'count'],
                'price': ['median', 'mean'],
                'square_footage': 'median'
            }).reset_index()

            # Flatten column names
            zip_stats.columns = ['zip_code', 'zip_median_days', 'zip_mean_days', 'zip_property_count',
                                 'zip_median_price', 'zip_mean_price', 'zip_median_sqft']

            # Calculate quick sale percentage (< 30 days)
            zip_quick_sales = df[df[self.target_name] < 30].groupby('zip_code').size().reset_index(name='quick_sales')
            zip_stats = zip_stats.merge(zip_quick_sales, on='zip_code', how='left')
            zip_stats['quick_sales'].fillna(0, inplace=True)
            zip_stats['zip_quick_sale_pct'] = (zip_stats['quick_sales'] / zip_stats['zip_property_count'] * 100)
            zip_stats.drop('quick_sales', axis=1, inplace=True)

            # Store zip stats for prediction time
            self.zip_stats = zip_stats

            # Merge back to dataframe
            df = df.merge(zip_stats, on='zip_code', how='left')

            # Create deviation features (how far from zip median)
            if 'price' in df.columns:
                df['price_vs_zip_median'] = (df['price'] - df['zip_median_price']) / (df['zip_median_price'] + 1)

            print(f"Created aggregate features for {len(zip_stats)} unique zip codes", file=sys.stderr)

        elif hasattr(self, 'zip_stats') and 'zip_code' in df.columns:
            # Use stored zip stats for prediction
            df = df.merge(self.zip_stats, on='zip_code', how='left')

            # Fill missing zip codes with global medians
            for col in self.zip_stats.columns:
                if col != 'zip_code' and col in df.columns:
                    df[col].fillna(df[col].median(), inplace=True)

            if 'price' in df.columns:
                df['price_vs_zip_median'] = (df['price'] - df['zip_median_price']) / (df['zip_median_price'] + 1)

        # Encode categorical variables (skip zip_code as we use it for aggregates)
        for col in categorical_columns:
            if col == 'zip_code':
                # Keep zip_code as is for merging, we'll drop it later if not needed
                continue
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                df[col] = self.label_encoders[col].fit_transform(df[col].astype(str))
            else:
                try:
                    df[col] = self.label_encoders[col].transform(df[col].astype(str))
                except ValueError:
                    # Handle unseen categories in test set
                    df[col] = self.label_encoders[col].transform([self.label_encoders[col].classes_[0]] * len(df))

        # Basic feature engineering
        if 'price' in df.columns and 'square_footage' in df.columns:
            df['price_per_sqft'] = df['price'] / (df['square_footage'] + 1)
        elif 'price' in df.columns and 'square_feet' in df.columns:
            df['price_per_sqft'] = df['price'] / (df['square_feet'] + 1)

        if 'bedrooms' in df.columns and 'bathrooms' in df.columns:
            df['bed_bath_ratio'] = df['bedrooms'] / (df['bathrooms'] + 1)

        # POLYNOMIAL FEATURES (non-linear relationships)
        if 'price_per_sqft' in df.columns:
            df['price_per_sqft_squared'] = df['price_per_sqft'] ** 2

        if 'property_age' in df.columns:
            df['property_age_squared'] = df['property_age'] ** 2

        if 'dist_to_downtown' in df.columns:
            df['dist_to_downtown_squared'] = df['dist_to_downtown'] ** 2

        # DECADE-BASED FEATURES from yearBuilt
        if 'year_built' in df.columns:
            df['decade'] = (df['year_built'] // 10) * 10
            # Create decade categories (more recent = higher value)
            df['is_pre_1950'] = (df['year_built'] < 1950).astype(int)
            df['is_1950_1979'] = ((df['year_built'] >= 1950) & (df['year_built'] < 1980)).astype(int)
            df['is_1980_1999'] = ((df['year_built'] >= 1980) & (df['year_built'] < 2000)).astype(int)
            df['is_2000_2009'] = ((df['year_built'] >= 2000) & (df['year_built'] < 2010)).astype(int)
            df['is_2010_plus'] = (df['year_built'] >= 2010).astype(int)

        # PROPERTY TYPE SPECIFIC FEATURES
        if 'property_type' in df.columns:
            # Flag high-variance property types
            df['is_townhouse'] = (df['property_type'] == 'Townhouse').astype(int)
            df['is_land'] = (df['property_type'] == 'Land').astype(int)
            df['is_single_family'] = (df['property_type'] == 'Single Family').astype(int)

        # ACTIVE LISTINGS COMPETITION FEATURES
        if 'active_listings_at_time' in df.columns:
            # Log transform to handle skewness
            df['active_listings_log'] = np.log1p(df['active_listings_at_time'])

            # Market heat indicator (high competition)
            if 'zip_code' in df.columns and is_training and self.target_name in df.columns:
                zip_active_stats = df.groupby('zip_code')['active_listings_at_time'].median().reset_index()
                zip_active_stats.columns = ['zip_code', 'zip_median_active']
                if hasattr(self, 'zip_stats'):
                    self.zip_stats = self.zip_stats.merge(zip_active_stats, on='zip_code', how='left')
                    df = df.merge(zip_active_stats, on='zip_code', how='left')
            elif hasattr(self, 'zip_stats') and 'zip_median_active' in self.zip_stats.columns:
                if 'zip_median_active' not in df.columns:
                    zip_active_stats = self.zip_stats[['zip_code', 'zip_median_active']]
                    df = df.merge(zip_active_stats, on='zip_code', how='left')

            # Competition vs zip median
            if 'zip_median_active' in df.columns:
                df['competition_vs_zip_avg'] = df['active_listings_at_time'] / (df['zip_median_active'] + 1)

        # INTERACTION FEATURES
        if 'price_per_sqft' in df.columns and 'zip_median_days' in df.columns:
            df['price_per_sqft_x_zip_days'] = df['price_per_sqft'] * df['zip_median_days']

        if 'property_age' in df.columns and 'is_peak_season' in df.columns:
            df['age_x_peak_season'] = df['property_age'] * df['is_peak_season']

        if 'square_footage' in df.columns and 'bedrooms' in df.columns:
            df['sqft_per_bedroom'] = df['square_footage'] / (df['bedrooms'] + 1)

        if 'bathrooms' in df.columns and 'price_per_sqft' in df.columns:
            df['bathrooms_x_price_per_sqft'] = df['bathrooms'] * df['price_per_sqft']

        # NEW INTERACTION FEATURES
        if 'price_per_sqft' in df.columns and 'active_listings_log' in df.columns:
            # Pricing in hot markets
            df['price_x_competition'] = df['price_per_sqft'] * df['active_listings_log']

        if 'property_age' in df.columns and 'competition_vs_zip_avg' in df.columns:
            # Old homes in competitive markets
            df['age_x_competition'] = df['property_age'] * df['competition_vs_zip_avg']

        if 'dist_to_downtown' in df.columns and 'is_peak_season' in df.columns:
            # Location + timing interaction
            df['location_x_season'] = df['dist_to_downtown'] * df['is_peak_season']

        if 'price_vs_zip_median' in df.columns and 'competition_vs_zip_avg' in df.columns:
            # Overpriced in competitive markets
            df['pricing_x_competition'] = df['price_vs_zip_median'] * df['competition_vs_zip_avg']

        # COMPETITIVENESS SCORE (relative value in local market)
        if 'price' in df.columns and 'zip_median_price' in df.columns and 'property_age' in df.columns:
            # Calculate average property age per zip
            if 'zip_code' in df.columns and is_training and self.target_name in df.columns:
                zip_age_stats = df.groupby('zip_code')['property_age'].median().reset_index()
                zip_age_stats.columns = ['zip_code', 'zip_median_age']
                if hasattr(self, 'zip_stats'):
                    self.zip_stats = self.zip_stats.merge(zip_age_stats, on='zip_code', how='left')
                    df = df.merge(zip_age_stats, on='zip_code', how='left')
            elif hasattr(self, 'zip_stats') and 'zip_median_age' in self.zip_stats.columns:
                if 'zip_median_age' not in df.columns:
                    zip_age_stats = self.zip_stats[['zip_code', 'zip_median_age']]
                    df = df.merge(zip_age_stats, on='zip_code', how='left')

            if 'zip_median_age' in df.columns:
                # Competitiveness = (relative price) * (relative age)
                # Lower score = better deal (cheaper and newer than neighborhood)
                price_ratio = df['price'] / (df['zip_median_price'] + 1)
                age_ratio = (df['property_age'] + 1) / (df['zip_median_age'] + 1)
                df['competitiveness_score'] = price_ratio * age_ratio

        if 'listing_date' in df.columns:
            df['listing_date'] = pd.to_datetime(df['listing_date'], errors='coerce')
            df['listing_month'] = df['listing_date'].dt.month
            df['listing_quarter'] = df['listing_date'].dt.quarter
            df['listing_day_of_week'] = df['listing_date'].dt.dayofweek
            df.drop('listing_date', axis=1, inplace=True)

        if 'sale_date' in df.columns:
            df.drop('sale_date', axis=1, inplace=True)

        # Drop non-numeric columns that couldn't be encoded
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

        # Split data FIRST to avoid data leakage
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Preprocess training data - calculate zip stats only on training set
        df_train = X_train.copy()
        df_train[self.target_name] = y_train
        df_train_preprocessed = self.preprocess_data(df_train, is_training=True)

        # Now preprocess test data using the zip stats from training
        df_test = X_test.copy()
        df_test[self.target_name] = y_test
        df_test_preprocessed = self.preprocess_data(df_test, is_training=False)

        # Separate features and target
        X_train = df_train_preprocessed.drop(self.target_name, axis=1, errors='ignore')
        X_test = df_test_preprocessed.drop(self.target_name, axis=1, errors='ignore')

        self.feature_names = X_train.columns.tolist()

        # Ensure both have the same columns
        for col in self.feature_names:
            if col not in X_test.columns:
                X_test[col] = 0
        X_test = X_test[self.feature_names]

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # HYPERPARAMETER TUNING
        print("\nPerforming hyperparameter tuning...", file=sys.stderr)

        # Focused grid search for RandomForest (best performing model)
        rf_param_grid = {
            'n_estimators': [200, 300],
            'max_depth': [15, 20],
            'min_samples_split': [5, 10],
            'min_samples_leaf': [1, 2]
        }

        rf_base = RandomForestRegressor(random_state=42, n_jobs=-1)
        rf_grid = GridSearchCV(
            rf_base,
            rf_param_grid,
            cv=3,
            scoring='neg_mean_absolute_error',
            n_jobs=-1,
            verbose=1
        )

        print("Tuning RandomForest...", file=sys.stderr)
        rf_grid.fit(X_train_scaled, y_train)
        rf_model = rf_grid.best_estimator_
        print(f"Best RF params: {rf_grid.best_params_}", file=sys.stderr)

        # Focused grid for XGBoost
        xgb_param_grid = {
            'n_estimators': [200, 300],
            'max_depth': [6, 8],
            'learning_rate': [0.05, 0.1],
            'subsample': [0.8, 0.9]
        }

        xgb_base = XGBRegressor(random_state=42, n_jobs=-1)
        xgb_grid = GridSearchCV(
            xgb_base,
            xgb_param_grid,
            cv=3,
            scoring='neg_mean_absolute_error',
            n_jobs=-1,
            verbose=1
        )

        print("Tuning XGBoost...", file=sys.stderr)
        xgb_grid.fit(X_train_scaled, y_train)
        xgb_model = xgb_grid.best_estimator_
        print(f"Best XGB params: {xgb_grid.best_params_}", file=sys.stderr)

        # Use default params for GradientBoosting (takes too long to tune)
        gb_model = GradientBoostingRegressor(
            n_estimators=200,
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
            'XGBoost': xgb_model
        }

        results = {}

        for name, model in individual_models.items():
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)

            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)

            results[name] = {
                'mae': mae,
                'rmse': rmse,
                'r2': r2
            }

        # ENSEMBLE STACKING
        # Create voting ensemble that combines all three models
        print("\nCreating ensemble model (voting regressor)...", file=sys.stderr)

        # Calculate optimal weights based on inverse MAE (better models get more weight)
        rf_mae = results['RandomForest']['mae']
        gb_mae = results['GradientBoosting']['mae']
        xgb_mae = results['XGBoost']['mae']

        # Inverse MAE as weights (lower MAE = higher weight)
        rf_weight = 1 / rf_mae
        gb_weight = 1 / gb_mae
        xgb_weight = 1 / xgb_mae

        # Normalize
        total = rf_weight + gb_weight + xgb_weight
        rf_weight = rf_weight / total * 10  # Scale to make numbers nicer
        gb_weight = gb_weight / total * 10
        xgb_weight = xgb_weight / total * 10

        print(f"Model weights: RF={rf_weight:.2f}, GB={gb_weight:.2f}, XGB={xgb_weight:.2f}", file=sys.stderr)

        ensemble = VotingRegressor(
            estimators=[
                ('rf', rf_model),
                ('gb', gb_model),
                ('xgb', xgb_model)
            ],
            weights=[rf_weight, gb_weight, xgb_weight]
        )

        # Train ensemble
        ensemble.fit(X_train_scaled, y_train)
        y_pred_ensemble = ensemble.predict(X_test_scaled)

        mae_ensemble = mean_absolute_error(y_test, y_pred_ensemble)
        rmse_ensemble = np.sqrt(mean_squared_error(y_test, y_pred_ensemble))
        r2_ensemble = r2_score(y_test, y_pred_ensemble)

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

        # Save model
        model_dir = Path(__file__).parent / 'models'
        model_dir.mkdir(exist_ok=True)

        joblib.dump(self.model, model_dir / 'model.pkl')
        joblib.dump(self.scaler, model_dir / 'scaler.pkl')
        joblib.dump(self.label_encoders, model_dir / 'label_encoders.pkl')
        joblib.dump(self.feature_names, model_dir / 'feature_names.pkl')

        # Save zip code statistics if they exist
        if hasattr(self, 'zip_stats'):
            joblib.dump(self.zip_stats, model_dir / 'zip_stats.pkl')
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
