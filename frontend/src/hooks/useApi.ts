import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { useStore, FireLocation, EnvironmentalData, RiskAssessment, Location } from '../store/useStore'
import React from 'react'

// API base URL
const API_BASE = '/api'

// Types for API responses
interface DetectFiresResponse {
  fires: FireLocation[]
  count: number
  source: string
  timestamp: string
}

interface EnvDataResponse {
  data: EnvironmentalData
  timestamp: string
}

interface AssessRiskRequest {
  location: Location
  envData?: EnvironmentalData
}

interface AssessRiskResponse {
  risk: RiskAssessment
  location: Location
  features?: Record<string, unknown>
  timestamp: string
}

// Post-fire spread types
export interface SpreadPoint {
  latitude: number
  longitude: number
  probability: number
  time_step: number
}

export interface FireSpreadConditions {
  ndvi: number | null
  temperature_celsius: number | null
  humidity_percent: number | null
  wind_direction_deg: number | null
  wind_speed_ms: number | null
  data_source: string
}

export interface FireSpreadResponse {
  ignition_point: { latitude: number; longitude: number }
  spread_radius_km: number
  spread_probability: number
  spread_points: SpreadPoint[]
  conditions: FireSpreadConditions
  wind_direction: number
  wind_speed: number
  time_steps_simulated: number
  model_info: {
    model_type: string
    model_path: string
    features: string
    spread_logic: string
  }
  timestamp: string
}

export interface PostFireSpreadRequest {
  latitude: number
  longitude: number
  wind_direction?: number
  wind_speed?: number
  time_steps?: number
  cell_size_deg?: number
}

// Fetch fire locations
export function useDetectFires(location: Location | null, timeFrame: number = 24) {
  const setFireLocations = useStore((state) => state.setFireLocations)
  
  return useQuery({
    queryKey: ['detect-fires', location?.id, timeFrame],
    queryFn: async (): Promise<DetectFiresResponse> => {
      const response = await fetch(`${API_BASE}/detect-fires`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ location, hours: timeFrame }),
      })
      if (!response.ok) throw new Error('Failed to fetch fire data')
      const data = await response.json()
      setFireLocations(data.fires)
      return data
    },
    enabled: !!location,
  })
}

// Fetch environmental data
export function useEnvData(location: Location | null) {
  const setEnvData = useStore((state) => state.setEnvData)
  
  return useQuery({
    queryKey: ['env-data', location?.id],
    queryFn: async (): Promise<EnvDataResponse> => {
      const response = await fetch(`${API_BASE}/env-data`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ location }),
      })
      if (!response.ok) throw new Error('Failed to fetch environmental data')
      const data = await response.json()
      setEnvData(data.data)
      return data
    },
    enabled: !!location,
  })
}

// Assess risk
export function useAssessRisk() {
  const queryClient = useQueryClient()
  const setRiskAssessment = useStore((state) => state.setRiskAssessment)
  const setSelectedLocation = useStore((state) => state.setSelectedLocation)
  
  // Use a ref to store the abort controller
  const abortControllerRef = React.useRef<AbortController | null>(null)
  
  const mutation = useMutation({
    mutationFn: async (request: AssessRiskRequest): Promise<AssessRiskResponse> => {
      // Cancel any existing request
      if (abortControllerRef.current) {
        abortControllerRef.current.abort()
      }
      
      // Create new abort controller
      abortControllerRef.current = new AbortController()
      
      const response = await fetch(`${API_BASE}/assess-risk`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(request),
        signal: abortControllerRef.current.signal,
      })
      if (!response.ok) throw new Error('Failed to assess risk')
      const data = await response.json()
      setRiskAssessment(data.risk)
      // Also update the selected location if it's a click assessment
      if (request.location) {
        setSelectedLocation(request.location)
      }
      return data
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['risk'] })
    },
  })
  
  // Add cancel method
  const cancel = () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort()
      abortControllerRef.current = null
    }
    mutation.reset()
  }
  
  return { ...mutation, cancel }
}

// Assess risk at a specific location (for map clicks)
export function useAssessRiskAtLocation() {
  const assessRisk = useAssessRisk()
  
  const assessAtLocation = (lat: number, lng: number) => {
    const location: Location = {
      id: `click-${lat}-${lng}`,
      name: `Location (${lat.toFixed(2)}, ${lng.toFixed(2)})`,
      lat,
      lng,
    }
    
    return assessRisk.mutateAsync({ location })
  }
  
  return {
    assessAtLocation,
    isLoading: assessRisk.isPending,
    error: assessRisk.error,
  }
}

// Fetch standalone weather data
export function useWeather(lat: number, lon: number) {
  return useQuery({
    queryKey: ['weather', lat, lon],
    queryFn: async () => {
      const response = await fetch(`${API_BASE}/weather?lat=${lat}&lon=${lon}`)
      if (!response.ok) throw new Error('Failed to fetch weather')
      return response.json()
    },
    enabled: lat > 0 && lon > 0,
  })
}

// ============================================================
// Post-Fire Spread Simulation Hook
// ============================================================

export function usePostFireSpread() {
  const queryClient = useQueryClient()
  
  const mutation = useMutation({
    mutationFn: async (request: PostFireSpreadRequest): Promise<FireSpreadResponse> => {
      const response = await fetch(`${API_BASE}/post-fire-spread`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(request),
      })
      if (!response.ok) {
        const error = await response.json().catch(() => ({ detail: 'Unknown error' }))
        throw new Error(error.detail || 'Failed to calculate fire spread')
      }
      return response.json()
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['post-fire-spread'] })
    },
  })
  
  return mutation
}

// Hook for simulating fire spread at a specific location
export function useSimulateFireSpread() {
  const spreadMutation = usePostFireSpread()
  
  const simulateAtLocation = (
    latitude: number,
    longitude: number,
    windDirection: number = 90,
    windSpeed: number = 15,
    timeSteps: number = 5
  ) => {
    return spreadMutation.mutateAsync({
      latitude,
      longitude,
      wind_direction: windDirection,
      wind_speed: windSpeed,
      time_steps: timeSteps,
      cell_size_deg: 0.0045,  // ~500m resolution
    })
  }
  
  return {
    simulateAtLocation,
    isLoading: spreadMutation.isPending,
    error: spreadMutation.error,
    spreadData: spreadMutation.data,
    reset: spreadMutation.reset,
  }
}
