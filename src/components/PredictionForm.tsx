import { useState } from 'react'
import axios from 'axios'
import './PredictionForm.css'

interface PredictionResult {
  predicted_days: number
  confidence_interval: {
    lower: number
    upper: number
  }
  feature_importance: Array<{
    feature: string
    importance: number
    value: any
  }>
  insights: string[]
  recommendation: string
}

const PredictionForm = () => {
  const [formData, setFormData] = useState({
    price: '',
    bedrooms: '',
    bathrooms: '',
    square_feet: '',
    lot_size: '',
    year_built: '',
    property_type: '',
    city: '',
    state: '',
  })

  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<PredictionResult | null>(null)
  const [error, setError] = useState<string | null>(null)

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value
    })
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setLoading(true)
    setError(null)
    setResult(null)

    try {
      // Convert numeric fields
      const propertyData = {
        ...formData,
        price: formData.price ? parseFloat(formData.price) : undefined,
        bedrooms: formData.bedrooms ? parseInt(formData.bedrooms) : undefined,
        bathrooms: formData.bathrooms ? parseFloat(formData.bathrooms) : undefined,
        square_feet: formData.square_feet ? parseFloat(formData.square_feet) : undefined,
        lot_size: formData.lot_size ? parseFloat(formData.lot_size) : undefined,
        year_built: formData.year_built ? parseInt(formData.year_built) : undefined,
      }

      const response = await axios.post('http://localhost:3001/api/predict', {
        property: propertyData
      })

      setResult(response.data)
    } catch (err: any) {
      setError(err.response?.data?.error || err.message || 'Failed to make prediction')
    } finally {
      setLoading(false)
    }
  }

  const handleReset = () => {
    setFormData({
      price: '',
      bedrooms: '',
      bathrooms: '',
      square_feet: '',
      lot_size: '',
      year_built: '',
      property_type: '',
      city: '',
      state: '',
    })
    setResult(null)
    setError(null)
  }

  return (
    <div className="prediction-form">
      <h2>Property Prediction</h2>
      <p className="description">
        Enter property details to predict how many days it will stay on the market.
      </p>

      <form onSubmit={handleSubmit}>
        <div className="form-grid">
          <div className="form-group">
            <label htmlFor="price">Price ($)</label>
            <input
              type="number"
              id="price"
              name="price"
              value={formData.price}
              onChange={handleChange}
              placeholder="e.g., 450000"
              required
            />
          </div>

          <div className="form-group">
            <label htmlFor="bedrooms">Bedrooms</label>
            <input
              type="number"
              id="bedrooms"
              name="bedrooms"
              value={formData.bedrooms}
              onChange={handleChange}
              placeholder="e.g., 3"
              required
            />
          </div>

          <div className="form-group">
            <label htmlFor="bathrooms">Bathrooms</label>
            <input
              type="number"
              step="0.5"
              id="bathrooms"
              name="bathrooms"
              value={formData.bathrooms}
              onChange={handleChange}
              placeholder="e.g., 2.5"
              required
            />
          </div>

          <div className="form-group">
            <label htmlFor="square_feet">Square Feet</label>
            <input
              type="number"
              id="square_feet"
              name="square_feet"
              value={formData.square_feet}
              onChange={handleChange}
              placeholder="e.g., 2000"
              required
            />
          </div>

          <div className="form-group">
            <label htmlFor="lot_size">Lot Size (sq ft)</label>
            <input
              type="number"
              id="lot_size"
              name="lot_size"
              value={formData.lot_size}
              onChange={handleChange}
              placeholder="e.g., 8000"
            />
          </div>

          <div className="form-group">
            <label htmlFor="year_built">Year Built</label>
            <input
              type="number"
              id="year_built"
              name="year_built"
              value={formData.year_built}
              onChange={handleChange}
              placeholder="e.g., 2015"
              required
            />
          </div>

          <div className="form-group">
            <label htmlFor="property_type">Property Type</label>
            <select
              id="property_type"
              name="property_type"
              value={formData.property_type}
              onChange={handleChange}
              required
            >
              <option value="">Select type...</option>
              <option value="Single Family">Single Family</option>
              <option value="Condo">Condo</option>
              <option value="Townhouse">Townhouse</option>
              <option value="Multi-Family">Multi-Family</option>
              <option value="Land">Land</option>
            </select>
          </div>

          <div className="form-group">
            <label htmlFor="city">City</label>
            <input
              type="text"
              id="city"
              name="city"
              value={formData.city}
              onChange={handleChange}
              placeholder="e.g., Seattle"
            />
          </div>

          <div className="form-group">
            <label htmlFor="state">State</label>
            <input
              type="text"
              id="state"
              name="state"
              value={formData.state}
              onChange={handleChange}
              placeholder="e.g., WA"
              maxLength={2}
            />
          </div>
        </div>

        <div className="form-actions">
          <button type="submit" disabled={loading} className="predict-btn">
            {loading ? 'Predicting...' : 'Predict Days on Market'}
          </button>
          <button type="button" onClick={handleReset} className="reset-btn">
            Reset Form
          </button>
        </div>
      </form>

      {error && (
        <div className="error-message">
          <h3>Prediction Failed</h3>
          <p>{error}</p>
        </div>
      )}

      {result && (
        <div className="prediction-results">
          <div className="result-header">
            <h3>Prediction Results</h3>
            <span className={`recommendation ${result.predicted_days < 30 ? 'good' : result.predicted_days < 60 ? 'normal' : 'concerning'}`}>
              {result.recommendation}
            </span>
          </div>

          <div className="prediction-card">
            <div className="prediction-value">
              <span className="label">Predicted Days on Market</span>
              <span className="value">{Math.round(result.predicted_days)} days</span>
            </div>

            <div className="confidence-range">
              <span className="label">Confidence Interval (95%)</span>
              <div className="range">
                <span>{Math.round(result.confidence_interval.lower)} days</span>
                <span>-</span>
                <span>{Math.round(result.confidence_interval.upper)} days</span>
              </div>
            </div>
          </div>

          {result.insights && result.insights.length > 0 && (
            <div className="insights-card">
              <h4>Broker Insights & Recommendations</h4>
              <ul className="insights-list">
                {result.insights.map((insight, index) => (
                  <li key={index}>{insight}</li>
                ))}
              </ul>
            </div>
          )}

          {result.feature_importance && result.feature_importance.length > 0 && (
            <div className="features-card">
              <h4>Key Factors Influencing This Prediction</h4>
              <div className="features-list">
                {result.feature_importance.slice(0, 5).map((feature, index) => (
                  <div key={index} className="feature-item">
                    <span className="feature-name">{feature.feature}</span>
                    <span className="feature-value">{feature.value !== undefined ? String(feature.value) : 'N/A'}</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

export default PredictionForm
