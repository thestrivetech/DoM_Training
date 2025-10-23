import express from 'express'
import cors from 'cors'
import dotenv from 'dotenv'
import { createClient } from '@supabase/supabase-js'
import { spawn } from 'child_process'
import path from 'path'
import { fileURLToPath } from 'url'
import { writeFileSync, unlinkSync } from 'fs'
import { prismaRE } from '../src/lib/database/prisma.js'

const __filename = fileURLToPath(import.meta.url)
const __dirname = path.dirname(__filename)

dotenv.config()

const app = express()
const PORT = process.env.PORT || 3001

// Middleware
app.use(cors())
app.use(express.json({ limit: '50mb' }))
app.use(express.urlencoded({ limit: '50mb', extended: true }))

// Supabase client
const supabaseUrl = process.env.REAL_ESTATE_SUPABASE_URL || ''
const supabaseKey = process.env.REAL_ESTATE_SUPABASE_ANON_KEY || ''
const supabase = createClient(supabaseUrl, supabaseKey)

// Health check endpoint
app.get('/api/health', (req, res) => {
  res.json({ status: 'ok', message: 'Server is running' })
})

// Fetch data from database
app.get('/api/data', async (req, res) => {
  try {
    const limit = parseInt(req.query.limit as string) || 1000

    // Fetch from property_listings table using Supabase
    const { data, error } = await supabase
      .from('property_listings')
      .select('*')
      .limit(limit)

    if (error) {
      console.error('Error fetching data:', error)
      return res.status(500).json({ error: 'Failed to fetch data', details: error.message })
    }

    if (!data || data.length === 0) {
      return res.status(404).json({
        error: 'No data found',
        details: 'No property listings with days_on_market data available in database'
      })
    }

    res.json({
      data,
      count: data.length,
      table: 'property_listings'
    })
  } catch (error: any) {
    console.error('Error in /api/data:', error)
    res.status(500).json({ error: 'Internal server error', details: error.message })
  }
})

// Get available tables
app.get('/api/tables', async (req, res) => {
  try {
    res.json({ tables: ['property_listings', 'county_economic_metrics', 'national_economic_metrics'] })
  } catch (error: any) {
    console.error('Error in /api/tables:', error)
    res.status(500).json({ error: 'Internal server error', details: error.message })
  }
})

// Train model endpoint
app.post('/api/train', async (req, res) => {
  try {
    const { data } = req.body

    if (!data || data.length === 0) {
      return res.status(400).json({ error: 'No data provided for training' })
    }

    // Write data to temporary file instead of passing as command line arg
    const tempDataFile = path.join(__dirname, '../ml_service/temp_training_data.json')
    writeFileSync(tempDataFile, JSON.stringify(data))

    // Call Python ML service with file path
    const pythonProcess = spawn('py', [
      path.join(__dirname, '../ml_service/train_model.py'),
      tempDataFile
    ])

    let result = ''
    let errorOutput = ''

    pythonProcess.stdout.on('data', (chunk) => {
      result += chunk.toString()
    })

    pythonProcess.stderr.on('data', (chunk) => {
      errorOutput += chunk.toString()
    })

    pythonProcess.on('close', (code) => {
      // Clean up temp file
      try {
        unlinkSync(tempDataFile)
      } catch (e) {
        console.error('Failed to delete temp file:', e)
      }

      if (code !== 0) {
        console.error('Python process error:', errorOutput)
        return res.status(500).json({
          error: 'Model training failed',
          details: errorOutput
        })
      }

      try {
        const parsedResult = JSON.parse(result)
        res.json(parsedResult)
      } catch (e) {
        res.json({ message: 'Model trained successfully', output: result })
      }
    })
  } catch (error: any) {
    console.error('Error in /api/train:', error)
    res.status(500).json({ error: 'Internal server error', details: error.message })
  }
})

// Predict endpoint
app.post('/api/predict', async (req, res) => {
  try {
    const { property } = req.body

    if (!property) {
      return res.status(400).json({ error: 'No property data provided' })
    }

    // Call Python ML service for prediction
    const pythonProcess = spawn('py', [
      path.join(__dirname, '../ml_service/predict.py'),
      JSON.stringify(property)
    ])

    let result = ''
    let errorOutput = ''

    pythonProcess.stdout.on('data', (data) => {
      result += data.toString()
    })

    pythonProcess.stderr.on('data', (data) => {
      errorOutput += data.toString()
    })

    pythonProcess.on('close', (code) => {
      if (code !== 0) {
        console.error('Python process error:', errorOutput)
        return res.status(500).json({
          error: 'Prediction failed',
          details: errorOutput
        })
      }

      try {
        const parsedResult = JSON.parse(result)
        res.json(parsedResult)
      } catch (e) {
        res.json({ message: 'Prediction completed', output: result })
      }
    })
  } catch (error: any) {
    console.error('Error in /api/predict:', error)
    res.status(500).json({ error: 'Internal server error', details: error.message })
  }
})

// Get feature importance
app.get('/api/feature-importance', async (req, res) => {
  try {
    const pythonProcess = spawn('py', [
      path.join(__dirname, '../ml_service/feature_importance.py')
    ])

    let result = ''
    let errorOutput = ''

    pythonProcess.stdout.on('data', (data) => {
      result += data.toString()
    })

    pythonProcess.stderr.on('data', (data) => {
      errorOutput += data.toString()
    })

    pythonProcess.on('close', (code) => {
      if (code !== 0) {
        console.error('Python process error:', errorOutput)
        return res.status(500).json({
          error: 'Failed to get feature importance',
          details: errorOutput
        })
      }

      try {
        const parsedResult = JSON.parse(result)
        res.json(parsedResult)
      } catch (e) {
        res.json({ message: 'Feature importance retrieved', output: result })
      }
    })
  } catch (error: any) {
    console.error('Error in /api/feature-importance:', error)
    res.status(500).json({ error: 'Internal server error', details: error.message })
  }
})

app.listen(PORT, () => {
  console.log(`Server running on http://localhost:${PORT}`)
})
