# Setup Guide

## Quick Start

Follow these steps to get your Real Estate Predictions app running:

### 1. Install Node.js Dependencies

```bash
npm install
```

This will automatically generate the Prisma client for your external real estate database.

### 2. Verify Database Connection

Test your connection to the Supabase database:

```bash
npm run test:db
```

You should see:
- ✅ Connection successful
- 📊 Database statistics
- 🏠 Sample property listing

### 3. Set Up Python ML Service

```bash
cd ml_service
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt
cd ..
```

### 4. Run the Application

Open **3 terminal windows**:

**Terminal 1 - Frontend:**
```bash
npm run dev
```
Frontend runs at: http://localhost:3000

**Terminal 2 - Backend API:**
```bash
npm run server
```
Backend runs at: http://localhost:3001

**Terminal 3 - Keep Python venv activated:**
```bash
cd ml_service
venv\Scripts\activate  # or source venv/bin/activate on Mac/Linux
```

### 5. Train Your Model

1. Open http://localhost:3000 in your browser
2. Click "Train Model" tab
3. Select number of properties (recommended: 1000-5000)
4. Click "Start Training"
5. Wait 1-3 minutes for training to complete

### 6. Make Predictions

1. Click "Make Prediction" tab
2. Enter property details
3. Get instant predictions with broker insights!

## Troubleshooting

### Red Lines in TypeScript Files

If you see TypeScript errors after setup:

1. Make sure you ran `npm install` (this generates Prisma client)
2. If errors persist, run manually:
   ```bash
   npm run prisma:generate
   ```

### "Cannot find module '@prisma/client-external-re'"

Run:
```bash
npm run prisma:generate
```

### Database Connection Failed

1. Check your `.env` file has correct credentials:
   - `REAL_ESTATE_DATABASE_URL`
   - `REAL_ESTATE_SUPABASE_URL`
   - `REAL_ESTATE_SUPABASE_ANON_KEY`

2. Verify you can connect to Supabase

3. Check if your IP is whitelisted in Supabase

### Python Errors

1. Make sure Python virtual environment is activated
2. Reinstall dependencies:
   ```bash
   cd ml_service
   pip install -r requirements.txt
   ```

## Next Steps

After setup is complete:

1. ✅ Train your model with your real estate data
2. ✅ Make predictions for new properties
3. ✅ View feature importance to understand market drivers
4. ✅ Use insights to help brokers price properties effectively

---

**Note:** This application only READS from your Supabase database. It never modifies your data. All ML training happens locally on your machine.
