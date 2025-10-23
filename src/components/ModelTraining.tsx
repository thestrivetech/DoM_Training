import { useState } from 'react'
import axios from 'axios'
import './ModelTraining.css'

interface TrainingResult {
  status: string
  best_model: string
  metrics: {
    mae: number
    rmse: number
    r2: number
  }
  feature_importance: Array<{
    feature: string
    importance: number
  }>
  training_samples: number
}

interface ModelTrainingProps {
  onTrainingComplete: () => void
}

const ModelTraining = ({ onTrainingComplete }: ModelTrainingProps) => {
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<TrainingResult | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [dataLimit, setDataLimit] = useState(5000)

  const handleTrain = async () => {
    try {
      setLoading(true)
      setError(null)
      setResult(null)

      // Fetch data from Supabase
      const dataResponse = await axios.get(`http://localhost:3001/api/data?limit=${dataLimit}`)

      if (!dataResponse.data || !dataResponse.data.data || dataResponse.data.data.length === 0) {
        throw new Error('No data available for training')
      }

      // Train the model
      const trainResponse = await axios.post('http://localhost:3001/api/train', {
        data: dataResponse.data.data
      })

      setResult(trainResponse.data)
      onTrainingComplete()
    } catch (err: any) {
      setError(err.response?.data?.error || err.message || 'Failed to train model')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="model-training">
      <h2>Train AI Model</h2>
      <p className="description">
        Train the machine learning model to predict "Days on Market" based on property features.
        The model will automatically identify the most important features for predictions.
      </p>

      <div className="training-config">
        <label>
          Number of properties to use for training:
          <input
            type="number"
            value={dataLimit}
            onChange={(e) => setDataLimit(parseInt(e.target.value))}
            min={100}
            max={50000}
            step={100}
            disabled={loading}
          />
        </label>

        <button
          onClick={handleTrain}
          disabled={loading}
          className="train-btn"
        >
          {loading ? 'Training Model...' : 'Start Training'}
        </button>
      </div>

      {loading && (
        <div className="loading-indicator">
          <div className="spinner"></div>
          <p>Training in progress... This may take a few minutes.</p>
        </div>
      )}

      {error && (
        <div className="error-message">
          <h3>Training Failed</h3>
          <p>{error}</p>
        </div>
      )}

      {result && (
        <div className="training-results">
          <h3>Training Complete!</h3>

          <div className="result-card">
            <h4>Model Performance</h4>
            <div className="metrics">
              <div className="metric">
                <span className="label">Best Model:</span>
                <span className="value">{result.best_model}</span>
              </div>
              <div className="metric">
                <span className="label">R² Score:</span>
                <span className="value">{(result.metrics.r2 * 100).toFixed(2)}%</span>
              </div>
              <div className="metric">
                <span className="label">Mean Absolute Error:</span>
                <span className="value">{result.metrics.mae.toFixed(2)} days</span>
              </div>
              <div className="metric">
                <span className="label">Root Mean Squared Error:</span>
                <span className="value">{result.metrics.rmse.toFixed(2)} days</span>
              </div>
              <div className="metric">
                <span className="label">Training Samples:</span>
                <span className="value">{result.training_samples.toLocaleString()}</span>
              </div>
            </div>
          </div>

          <div className="result-card">
            <h4>Top 10 Most Important Features</h4>
            <div className="feature-list">
              {result.feature_importance.map((feature, index) => (
                <div key={feature.feature} className="feature-item">
                  <span className="rank">#{index + 1}</span>
                  <span className="feature-name">{feature.feature}</span>
                  <div className="importance-bar">
                    <div
                      className="importance-fill"
                      style={{ width: `${(feature.importance / result.feature_importance[0].importance) * 100}%` }}
                    ></div>
                  </div>
                  <span className="importance-value">{(feature.importance * 100).toFixed(2)}%</span>
                </div>
              ))}
            </div>
          </div>

          <div className="success-message">
            <p>Model is ready for predictions! Go to the "Make Prediction" tab to try it out.</p>
          </div>
        </div>
      )}
    </div>
  )
}

export default ModelTraining
