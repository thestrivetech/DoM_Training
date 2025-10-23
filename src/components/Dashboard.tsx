import { useState, useEffect } from 'react'
import axios from 'axios'
import './Dashboard.css'

interface DataSummary {
  totalRecords: number
  avgDaysOnMarket: number
  avgPrice: number
  propertyTypes: Record<string, number>
}

const Dashboard = () => {
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [dataSummary, setDataSummary] = useState<DataSummary | null>(null)
  const [tableName, setTableName] = useState('')

  useEffect(() => {
    fetchData()
  }, [])

  const fetchData = async () => {
    try {
      setLoading(true)
      const response = await axios.get('http://localhost:3001/api/data?limit=5000')

      if (response.data && response.data.data) {
        const data = response.data.data
        setTableName(response.data.table)

        // Calculate summary statistics
        const totalRecords = data.length
        const daysOnMarketData = data.filter((p: any) => p.days_on_market != null)
        const avgDaysOnMarket = daysOnMarketData.reduce((acc: number, p: any) => acc + (p.days_on_market || 0), 0) / (daysOnMarketData.length || 1)

        const priceData = data.filter((p: any) => p.price != null)
        const avgPrice = priceData.reduce((acc: number, p: any) => acc + (p.price || 0), 0) / (priceData.length || 1)

        const propertyTypes = data.reduce((acc: Record<string, number>, p: any) => {
          const type = p.property_type || 'Unknown'
          acc[type] = (acc[type] || 0) + 1
          return acc
        }, {})

        setDataSummary({
          totalRecords,
          avgDaysOnMarket,
          avgPrice,
          propertyTypes
        })
      }

      setError(null)
    } catch (err: any) {
      setError(err.response?.data?.error || err.message || 'Failed to fetch data')
    } finally {
      setLoading(false)
    }
  }

  if (loading) {
    return <div className="dashboard loading">Loading data...</div>
  }

  if (error) {
    return (
      <div className="dashboard error">
        <h2>Error Loading Data</h2>
        <p>{error}</p>
        <button onClick={fetchData}>Retry</button>
      </div>
    )
  }

  return (
    <div className="dashboard">
      <h2>Market Overview</h2>

      {tableName && (
        <div className="info-banner">
          <p>Data Source: <strong>{tableName}</strong> table</p>
        </div>
      )}

      {dataSummary && (
        <div className="stats-grid">
          <div className="stat-card">
            <h3>Total Properties</h3>
            <p className="stat-value">{dataSummary.totalRecords.toLocaleString()}</p>
          </div>

          <div className="stat-card">
            <h3>Avg Days on Market</h3>
            <p className="stat-value">{dataSummary.avgDaysOnMarket.toFixed(1)} days</p>
          </div>

          <div className="stat-card">
            <h3>Avg Property Price</h3>
            <p className="stat-value">${dataSummary.avgPrice.toLocaleString(undefined, { maximumFractionDigits: 0 })}</p>
          </div>

          <div className="stat-card full-width">
            <h3>Property Types Distribution</h3>
            <div className="property-types">
              {Object.entries(dataSummary.propertyTypes)
                .sort((a, b) => b[1] - a[1])
                .slice(0, 5)
                .map(([type, count]) => (
                  <div key={type} className="property-type-item">
                    <span className="type-name">{type}</span>
                    <span className="type-count">{count} ({((count / dataSummary.totalRecords) * 100).toFixed(1)}%)</span>
                  </div>
                ))}
            </div>
          </div>
        </div>
      )}

      <div className="actions">
        <button onClick={fetchData} className="refresh-btn">Refresh Data</button>
      </div>
    </div>
  )
}

export default Dashboard
