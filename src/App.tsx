import { useState } from 'react'
import './App.css'
import Dashboard from './components/Dashboard'
import ModelTraining from './components/ModelTraining'
import PredictionForm from './components/PredictionForm'
import FeatureImportance from './components/FeatureImportance'

function App() {
  const [activeTab, setActiveTab] = useState<'dashboard' | 'train' | 'predict' | 'features'>('dashboard')
  const [isModelTrained, setIsModelTrained] = useState(false)

  return (
    <div className="app">
      <header className="app-header">
        <h1>Real Estate Market Predictions</h1>
        <p>AI-Powered Days on Market Prediction System</p>
      </header>

      <nav className="tab-navigation">
        <button
          className={activeTab === 'dashboard' ? 'active' : ''}
          onClick={() => setActiveTab('dashboard')}
        >
          Dashboard
        </button>
        <button
          className={activeTab === 'train' ? 'active' : ''}
          onClick={() => setActiveTab('train')}
        >
          Train Model
        </button>
        <button
          className={activeTab === 'predict' ? 'active' : ''}
          onClick={() => setActiveTab('predict')}
          disabled={!isModelTrained}
        >
          Make Prediction
        </button>
        <button
          className={activeTab === 'features' ? 'active' : ''}
          onClick={() => setActiveTab('features')}
          disabled={!isModelTrained}
        >
          Feature Importance
        </button>
      </nav>

      <main className="app-main">
        {activeTab === 'dashboard' && <Dashboard />}
        {activeTab === 'train' && <ModelTraining onTrainingComplete={() => setIsModelTrained(true)} />}
        {activeTab === 'predict' && <PredictionForm />}
        {activeTab === 'features' && <FeatureImportance />}
      </main>

      <footer className="app-footer">
        <p>Built for Real Estate Professionals | Data-Driven Market Insights</p>
      </footer>
    </div>
  )
}

export default App
