import { useState, useEffect } from 'react'
import axios from 'axios'
import './FeatureImportance.css'

interface Feature {
  feature: string
  importance: number
  importance_percentage: number
}

interface FeatureImportanceData {
  feature_importance: Feature[]
  total_features: number
  top_features: Feature[]
}

const FeatureImportance = () => {
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [data, setData] = useState<FeatureImportanceData | null>(null)

  useEffect(() => {
    fetchFeatureImportance()
  }, [])

  const fetchFeatureImportance = async () => {
    try {
      setLoading(true)
      const response = await axios.get('http://localhost:3001/api/feature-importance')
      setData(response.data)
      setError(null)
    } catch (err: any) {
      setError(err.response?.data?.error || err.message || 'Failed to fetch feature importance')
    } finally {
      setLoading(false)
    }
  }

  if (loading) {
    return (
      <div className="feature-importance loading">
        <div className="spinner"></div>
        <p>Loading feature importance...</p>
      </div>
    )
  }

  if (error) {
    return (
      <div className="feature-importance error">
        <h2>Error</h2>
        <p>{error}</p>
        <button onClick={fetchFeatureImportance}>Retry</button>
      </div>
    )
  }

  if (!data) {
    return (
      <div className="feature-importance">
        <p>No data available</p>
      </div>
    )
  }

  return (
    <div className="feature-importance">
      <h2>Feature Importance Analysis</h2>
      <p className="description">
        Understanding which property features have the most significant impact on Days on Market.
        These insights help brokers focus on the most critical factors when pricing and marketing properties.
      </p>

      <div className="summary-card">
        <h3>Summary</h3>
        <p>Total features analyzed: <strong>{data.total_features}</strong></p>
        <p>The model considers all available property characteristics to make accurate predictions.</p>
      </div>

      <div className="top-features-card">
        <h3>Top 10 Most Important Features</h3>
        <div className="features-chart">
          {data.top_features.map((feature, index) => (
            <div key={feature.feature} className="chart-item">
              <div className="chart-header">
                <span className="rank">#{index + 1}</span>
                <span className="feature-name">{feature.feature}</span>
                <span className="percentage">{feature.importance_percentage.toFixed(2)}%</span>
              </div>
              <div className="bar-container">
                <div
                  className="bar-fill"
                  style={{
                    width: `${(feature.importance_percentage / data.top_features[0].importance_percentage) * 100}%`
                  }}
                ></div>
              </div>
            </div>
          ))}
        </div>
      </div>

      <div className="insights-card">
        <h3>Broker Action Items</h3>
        <div className="action-items">
          <div className="action-item">
            <h4>Focus on Top Features</h4>
            <p>When listing a property, emphasize the most important features in your marketing materials and property descriptions.</p>
          </div>
          <div className="action-item">
            <h4>Pricing Strategy</h4>
            <p>Features with high importance scores significantly impact market time. Adjust pricing based on how your property compares in these areas.</p>
          </div>
          <div className="action-item">
            <h4>Property Improvements</h4>
            <p>Before listing, consider improving or highlighting features that the model identifies as important to reduce time on market.</p>
          </div>
          <div className="action-item">
            <h4>Market Positioning</h4>
            <p>Use these insights to position properties competitively in the market and set realistic expectations with sellers.</p>
          </div>
        </div>
      </div>

      <div className="all-features-card">
        <h3>All Features (Ranked by Importance)</h3>
        <div className="features-table">
          <table>
            <thead>
              <tr>
                <th>Rank</th>
                <th>Feature</th>
                <th>Importance</th>
                <th>% of Total</th>
              </tr>
            </thead>
            <tbody>
              {data.feature_importance.map((feature, index) => (
                <tr key={feature.feature}>
                  <td>{index + 1}</td>
                  <td>{feature.feature}</td>
                  <td>{feature.importance.toFixed(4)}</td>
                  <td>{feature.importance_percentage.toFixed(2)}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}

export default FeatureImportance
