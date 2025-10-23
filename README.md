# Real Estate Market Predictions

An AI-powered prediction system for real estate professionals to predict "Days on Market" for properties using machine learning.

## Features

- **Dashboard**: Overview of market statistics and property data
- **Model Training**: Train ML models (Random Forest, Gradient Boosting, XGBoost) on your real estate data
- **Predictions**: Get accurate predictions for how long a property will stay on market
- **Feature Importance**: Understand which property characteristics most impact market time
- **Broker Insights**: Actionable recommendations for pricing and marketing strategies

## Tech Stack

### Frontend
- React 18 with TypeScript
- Vite for fast development
- Recharts for visualizations
- Axios for API communication

### Backend
- Node.js with Express
- TypeScript
- Supabase for database
- Python ML service for predictions

### Machine Learning
- scikit-learn (Random Forest, Gradient Boosting)
- XGBoost
- Feature engineering pipeline
- Model performance metrics

## Prerequisites

- Node.js 18+ and npm
- Python 3.8+
- Supabase account with real estate data

## Installation

### 1. Clone the repository

```bash
cd RealEstatePredictions
```

### 2. Install Node.js dependencies

```bash
npm install
```

### 3. Set up Python environment

```bash
cd ml_service
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate

pip install -r requirements.txt
cd ..
```

### 4. Configure environment variables

Copy `.env.example` to `.env` and fill in your Supabase credentials:

```bash
cp .env.example .env
```

Edit `.env` with your actual credentials:

```env
VITE_REAL_ESTATE_SUPABASE_URL=https://your-project.supabase.co
VITE_REAL_ESTATE_SUPABASE_ANON_KEY=your_anon_key_here

REAL_ESTATE_SUPABASE_URL=https://your-project.supabase.co
REAL_ESTATE_SUPABASE_ANON_KEY=your_anon_key_here
```

## Running the Application

### Development Mode

You'll need three terminal windows:

**Terminal 1 - Frontend (React)**
```bash
npm run dev
```
The frontend will run on http://localhost:3000

**Terminal 2 - Backend (Node.js API)**
```bash
npm run server
```
The API server will run on http://localhost:3001

**Terminal 3 - Python ML Service**
Make sure Python virtual environment is activated:
```bash
cd ml_service
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

## Usage Guide

### 1. Dashboard
- View overall market statistics
- See total properties, average days on market, and average prices
- Understand property type distributions

### 2. Train Model
1. Navigate to "Train Model" tab
2. Select the number of properties to use for training (recommended: 1000-5000)
3. Click "Start Training"
4. Wait for training to complete (1-3 minutes)
5. Review model performance metrics and feature importance

### 3. Make Predictions
1. Navigate to "Make Prediction" tab (only available after training)
2. Enter property details:
   - Price
   - Bedrooms
   - Bathrooms
   - Square feet
   - Lot size
   - Year built
   - Property type
   - Location (city, state)
3. Click "Predict Days on Market"
4. Review:
   - Predicted days on market
   - Confidence interval
   - Broker insights and recommendations
   - Key factors influencing the prediction

### 4. Feature Importance
- View which property features most impact days on market
- See ranked list of all features
- Get actionable insights for broker activities

## Database Schema

Your Supabase database should have a table (e.g., `properties`, `listings`, or `real_estate_data`) with columns such as:

- `days_on_market` (target variable - REQUIRED)
- `price`
- `bedrooms`
- `bathrooms`
- `square_feet`
- `lot_size`
- `year_built`
- `property_type`
- `city`
- `state`
- `zip_code`
- `listing_date`
- `sale_date`

The model will automatically detect and use available columns.

## Machine Learning Details

### Models Trained
1. **Random Forest Regressor** - Ensemble of decision trees
2. **Gradient Boosting Regressor** - Sequential tree-based learning
3. **XGBoost** - Optimized gradient boosting

The system automatically selects the best performing model based on R² score.

### Feature Engineering
Automatically creates derived features:
- `price_per_sqft` - Price per square foot
- `bed_bath_ratio` - Bedrooms to bathrooms ratio
- `property_age` - Current year minus year built
- `listing_month` - Month of listing (seasonality)
- `listing_quarter` - Quarter of listing
- `listing_day_of_week` - Day of week listed

### Performance Metrics
- **R² Score**: Percentage of variance explained (higher is better)
- **MAE (Mean Absolute Error)**: Average prediction error in days
- **RMSE (Root Mean Squared Error)**: Standard deviation of prediction errors

## Project Structure

```
RealEstatePredictions/
├── src/                          # React frontend
│   ├── components/              # React components
│   │   ├── Dashboard.tsx        # Market overview
│   │   ├── ModelTraining.tsx    # Model training interface
│   │   ├── PredictionForm.tsx   # Prediction input form
│   │   └── FeatureImportance.tsx # Feature analysis
│   ├── services/                # API services
│   │   └── dataService.ts       # Data fetching logic
│   ├── lib/                     # Libraries
│   │   └── supabase.ts          # Supabase client
│   ├── App.tsx                  # Main app component
│   └── main.tsx                 # App entry point
├── server/                      # Node.js backend
│   └── index.ts                 # Express API server
├── ml_service/                  # Python ML service
│   ├── train_model.py           # Model training script
│   ├── predict.py               # Prediction script
│   ├── feature_importance.py   # Feature analysis script
│   ├── requirements.txt         # Python dependencies
│   └── models/                  # Saved models (generated)
├── package.json                 # Node dependencies
├── tsconfig.json               # TypeScript config
├── vite.config.ts              # Vite config
└── README.md                   # This file
```

## Troubleshooting

### "Model not found" error
- Make sure you've trained the model first in the "Train Model" tab
- Check that the `ml_service/models/` directory exists and contains `.pkl` files

### Python errors
- Ensure Python virtual environment is activated
- Verify all Python dependencies are installed: `pip install -r ml_service/requirements.txt`

### Database connection issues
- Verify your `.env` file has correct Supabase credentials
- Check that your database table name matches what the code expects
- Ensure the database has a `days_on_market` column

### API connection errors
- Make sure the backend server is running on port 3001
- Check that frontend proxy is configured correctly in `vite.config.ts`

## Contributing

This is a real estate prediction tool designed for broker insights. Feel free to extend it with:
- Additional ML models
- More sophisticated feature engineering
- Real-time market data integration
- Historical trend analysis

## License

MIT License - feel free to use and modify for your real estate business needs.

## Support

For issues or questions, please check:
1. Database schema matches expected format
2. All environment variables are set correctly
3. Python and Node.js dependencies are installed
4. All three services (frontend, backend, ML) are running

---

Built with ❤️ for Real Estate Professionals
