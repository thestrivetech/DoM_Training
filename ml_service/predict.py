#!/usr/bin/env python3
import sys
import json
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

def load_model():
    """Load the trained model and preprocessing objects"""
    model_dir = Path(__file__).parent / 'models'

    if not model_dir.exists():
        raise FileNotFoundError("Model directory not found. Please train the model first.")

    model = joblib.load(model_dir / 'model.pkl')
    scaler = joblib.load(model_dir / 'scaler.pkl')
    label_encoders = joblib.load(model_dir / 'label_encoders.pkl')
    feature_names = joblib.load(model_dir / 'feature_names.pkl')

    return model, scaler, label_encoders, feature_names

def preprocess_property(property_data, label_encoders, feature_names):
    """Preprocess a single property for prediction"""
    df = pd.DataFrame([property_data])

    # Apply the same preprocessing as training
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()

    # Handle missing values
    for col in numeric_columns:
        if df[col].isna().any():
            df[col].fillna(0, inplace=True)

    # Encode categorical variables
    for col in categorical_columns:
        if col in label_encoders:
            try:
                df[col] = label_encoders[col].transform(df[col].astype(str))
            except ValueError:
                # Handle unseen categories
                df[col] = 0

    # Feature engineering
    if 'price' in df.columns and 'square_feet' in df.columns:
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

    # Ensure all expected features are present
    for feature in feature_names:
        if feature not in df.columns:
            df[feature] = 0

    # Select only the features used in training
    df = df[feature_names]

    return df

def generate_insights(property_data, predicted_days, feature_importance):
    """Generate actionable insights for brokers"""
    insights = []

    # Price-related insights
    if 'price' in property_data:
        price = property_data['price']
        if predicted_days > 60:
            insights.append(f"Property may be overpriced at ${price:,.0f}. Consider price reduction.")
        elif predicted_days < 15:
            insights.append(f"Property is competitively priced at ${price:,.0f}. High demand expected.")

    # Location insights
    if 'city' in property_data or 'state' in property_data:
        if predicted_days > 45:
            insights.append("Location may be less desirable. Emphasize unique property features in marketing.")

    # Property features
    if 'bedrooms' in property_data and property_data['bedrooms'] < 2:
        insights.append("Limited bedrooms may extend time on market. Target first-time buyers or investors.")

    # Timing insights
    if predicted_days > 90:
        insights.append("Extended market time expected. Consider staging, professional photography, or price adjustment.")
    elif predicted_days < 20:
        insights.append("Quick sale expected. Prepare for multiple offers and consider setting a deadline.")

    # Top features driving prediction
    if feature_importance:
        top_feature = feature_importance[0]
        insights.append(f"The most important factor is '{top_feature['feature']}'. Focus on highlighting this in listings.")

    return insights

def main():
    try:
        if len(sys.argv) < 2:
            print(json.dumps({'error': 'No property data provided'}))
            sys.exit(1)

        property_data = json.loads(sys.argv[1])

        # Load model
        model, scaler, label_encoders, feature_names = load_model()

        # Preprocess property
        X = preprocess_property(property_data, label_encoders, feature_names)

        # Scale features
        X_scaled = scaler.transform(X)

        # Make prediction
        prediction = model.predict(X_scaled)[0]

        # Get feature importance for this property
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            feature_importance = [
                {'feature': feat, 'importance': float(imp), 'value': property_data.get(feat, 'N/A')}
                for feat, imp in sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)[:10]
            ]
        else:
            feature_importance = []

        # Generate insights
        insights = generate_insights(property_data, prediction, feature_importance)

        # Confidence interval (rough estimate)
        confidence_margin = prediction * 0.15  # ±15% confidence interval

        result = {
            'predicted_days': float(prediction),
            'confidence_interval': {
                'lower': max(0, float(prediction - confidence_margin)),
                'upper': float(prediction + confidence_margin)
            },
            'feature_importance': feature_importance,
            'insights': insights,
            'recommendation': 'Quick sale expected' if prediction < 30 else 'Normal market time' if prediction < 60 else 'Extended time on market expected'
        }

        print(json.dumps(result))
        sys.exit(0)

    except FileNotFoundError as e:
        print(json.dumps({'error': str(e), 'message': 'Please train the model first'}), file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(json.dumps({'error': str(e)}), file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()
