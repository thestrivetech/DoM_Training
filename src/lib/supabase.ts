import { createClient } from '@supabase/supabase-js'

const supabaseUrl = import.meta.env.VITE_REAL_ESTATE_SUPABASE_URL || ''
const supabaseAnonKey = import.meta.env.VITE_REAL_ESTATE_SUPABASE_ANON_KEY || ''

export const supabase = createClient(supabaseUrl, supabaseAnonKey)

// Types for Real Estate Data
export interface RealEstateProperty {
  id: string
  address?: string
  city?: string
  state?: string
  zip_code?: string
  price?: number
  bedrooms?: number
  bathrooms?: number
  square_feet?: number
  lot_size?: number
  year_built?: number
  property_type?: string
  days_on_market?: number
  status?: string
  listing_date?: string
  sale_date?: string
  [key: string]: any
}

export interface PredictionResult {
  predicted_days: number
  confidence_interval: {
    lower: number
    upper: number
  }
  feature_importance: Array<{
    feature: string
    importance: number
    value?: any
  }>
  insights: string[]
}
