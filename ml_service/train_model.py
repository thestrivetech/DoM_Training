#!/usr/bin/env python3
import sys
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
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
        query = """
            SELECT
                id,
                "formattedAddress" as formatted_address,
                city,
                state,
                "zipCode" as zip_code,
                price,
                bedrooms,
                bathrooms,
                "squareFootage" as square_footage,
                "lotSize" as lot_size,
                "yearBuilt" as year_built,
                "propertyType" as property_type,
                status,
                "listedDate" as listing_date,
                "removedDate" as removed_date,
                "daysOnMarket" as days_on_market,
                latitude,
                longitude,
                county,
                "propertyAge",
                "distToDowntown",
                "daysIntoYear",
                "isPeakSeason"
            FROM inactive_listings
            WHERE "daysOnMarket" IS NOT NULL
            AND price IS NOT NULL
            AND bedrooms IS NOT NULL
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

    def preprocess_data(self, df):
        """Preprocess the real estate data"""
        # Make a copy to avoid modifying original
        df = df.copy()

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

        # Encode categorical variables
        for col in categorical_columns:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                df[col] = self.label_encoders[col].fit_transform(df[col].astype(str))
            else:
                df[col] = self.label_encoders[col].transform(df[col].astype(str))

        # Feature engineering
        if 'price' in df.columns and 'square_footage' in df.columns:
            df['price_per_sqft'] = df['price'] / (df['square_footage'] + 1)
        elif 'price' in df.columns and 'square_feet' in df.columns:
            df['price_per_sqft'] = df['price'] / (df['square_feet'] + 1)

        if 'bedrooms' in df.columns and 'bathrooms' in df.columns:
            df['bed_bath_ratio'] = df['bedrooms'] / (df['bathrooms'] + 1)

        if 'year_built' in df.columns:
            current_year = pd.Timestamp.now().year
            df['property_age'] = current_year - df['year_built']

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

        # Preprocess
        X = self.preprocess_data(X)
        self.feature_names = X.columns.tolist()

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Try multiple models and select the best
        models = {
            'RandomForest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
            'GradientBoosting': GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42),
            'XGBoost': XGBRegressor(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1)
        }

        best_score = -np.inf
        best_model_name = None

        results = {}

        for name, model in models.items():
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

            if r2 > best_score:
                best_score = r2
                best_model_name = name
                self.model = model

        # Get feature importance
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
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

