import { supabase, RealEstateProperty } from '../lib/supabase'

export class DataService {
  /**
   * Fetch real estate properties from Supabase
   * @param limit - Maximum number of records to fetch
   * @param filters - Optional filters to apply
   */
  static async fetchProperties(
    limit: number = 1000,
    filters?: Partial<RealEstateProperty>
  ): Promise<RealEstateProperty[]> {
    try {
      let query = supabase
        .from('properties') // Adjust table name as needed
        .select('*')
        .limit(limit)

      // Apply filters if provided
      if (filters) {
        Object.entries(filters).forEach(([key, value]) => {
          if (value !== undefined && value !== null) {
            query = query.eq(key, value)
          }
        })
      }

      const { data, error } = await query

      if (error) {
        console.error('Error fetching properties:', error)
        throw error
      }

      return data || []
    } catch (error) {
      console.error('Failed to fetch properties:', error)
      throw error
    }
  }

  /**
   * Get available table names from the database
   */
  static async getAvailableTables(): Promise<string[]> {
    try {
      const { data, error } = await supabase.rpc('get_table_names')

      if (error) {
        console.error('Error fetching table names:', error)
        // Fallback to common table names
        return ['properties', 'listings', 'real_estate_data']
      }

      return data || []
    } catch (error) {
      console.error('Failed to fetch table names:', error)
      return ['properties', 'listings', 'real_estate_data']
    }
  }

  /**
   * Get schema information for a table
   */
  static async getTableSchema(tableName: string): Promise<any> {
    try {
      const { data, error } = await supabase
        .from(tableName)
        .select('*')
        .limit(1)

      if (error) {
        console.error('Error fetching table schema:', error)
        throw error
      }

      return data && data.length > 0 ? Object.keys(data[0]) : []
    } catch (error) {
      console.error('Failed to fetch table schema:', error)
      throw error
    }
  }

  /**
   * Get summary statistics of the data
   */
  static async getDataSummary(): Promise<{
    totalRecords: number
    avgDaysOnMarket: number
    avgPrice: number
    propertyTypes: Record<string, number>
  }> {
    try {
      const properties = await this.fetchProperties(10000)

      const totalRecords = properties.length
      const avgDaysOnMarket = properties
        .filter(p => p.days_on_market)
        .reduce((acc, p) => acc + (p.days_on_market || 0), 0) / properties.filter(p => p.days_on_market).length

      const avgPrice = properties
        .filter(p => p.price)
        .reduce((acc, p) => acc + (p.price || 0), 0) / properties.filter(p => p.price).length

      const propertyTypes = properties.reduce((acc, p) => {
        const type = p.property_type || 'Unknown'
        acc[type] = (acc[type] || 0) + 1
        return acc
      }, {} as Record<string, number>)

      return {
        totalRecords,
        avgDaysOnMarket,
        avgPrice,
        propertyTypes,
      }
    } catch (error) {
      console.error('Failed to get data summary:', error)
      throw error
    }
  }
}
