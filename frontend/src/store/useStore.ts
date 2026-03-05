import { create } from 'zustand'

export type AppMode = 'fire' | 'risk' | 'postfire'
export type MapStyle = 'dark' | 'satellite' | 'light'
export type BorderStyle = 'countries'

export interface FireLocation {
  id: string
  lat: number
  lng: number
  intensity: number
  confidence: number
  timestamp: string
  brightness?: number
  frp?: number
  satellite?: string
  acq_datetime?: string
}

export interface EnvironmentalData {
  temperature: number
  humidity: number
  windSpeed: number
  windDirection: number
  vegetationIndex: number
  droughtIndex: number
  dewpoint?: number
  cloudCover?: number
  pressure?: number
  precipitation?: number
}

export interface RiskAssessment {
  level: 'low' | 'medium' | 'high' | 'critical'
  score: number
  probability: number
  alert_priority: string
  confidence: string
  factors: {
    weather: number
    vegetation: number
    topography: number
    historical: number
  }
}

export interface Location {
  id: string
  name: string
  lat: number
  lng: number
}

interface AppState {
  // Mode
  mode: AppMode
  setMode: (mode: AppMode) => void
  
  // Map style
  mapStyle: MapStyle
  setMapStyle: (style: MapStyle) => void
  
  // Border visibility
  borderStyle: BorderStyle
  setBorderStyle: (style: BorderStyle) => void
  
  // Selected location
  selectedLocation: Location | null
  setSelectedLocation: (location: Location | null) => void
  
  // Fire detection time frame (in hours)
  fireTimeFrame: number
  setFireTimeFrame: (hours: number) => void
  
  // Clicked location (for map click assessment)
  clickedLocation: Location | null
  setClickedLocation: (location: Location | null) => void
  
  // Fire data
  fireLocations: FireLocation[]
  setFireLocations: (locations: FireLocation[]) => void
  
  // Environmental data
  envData: EnvironmentalData | null
  setEnvData: (data: EnvironmentalData | null) => void
  
  // Risk assessment
  riskAssessment: RiskAssessment | null
  setRiskAssessment: (assessment: RiskAssessment | null) => void
  
  // Map center
  mapCenter: [number, number]
  setMapCenter: (center: [number, number]) => void
  
  // Weather tab expanded state
  isWeatherExpanded: boolean
  setWeatherExpanded: (expanded: boolean) => void
  
  // Loading states
  isLoading: boolean
  setIsLoading: (loading: boolean) => void
  
  // Active panel (for tab navigation)
  activePanel: 'location' | 'weather' | 'fires' | 'risk'
  setActivePanel: (panel: 'location' | 'weather' | 'fires' | 'risk') => void
}

// Default location: Nepal
const DEFAULT_LOCATION: Location = {
  id: 'np',
  name: 'Nepal',
  lat: 28.3949,
  lng: 84.1240,
}

export const useStore = create<AppState>((set) => ({
  // Mode
  mode: 'fire',
  setMode: (mode) => set({ mode }),
  
  // Map style
  mapStyle: 'dark',
  setMapStyle: (mapStyle) => set({ mapStyle }),
  
  // Border style
  borderStyle: 'countries',
  setBorderStyle: (borderStyle) => set({ borderStyle }),
  
  // Selected location
  selectedLocation: DEFAULT_LOCATION,
  setSelectedLocation: (location) => set({ selectedLocation: location }),
  
  // Fire detection time frame (default 24 hours)
  fireTimeFrame: 24,
  setFireTimeFrame: (fireTimeFrame) => set({ fireTimeFrame }),
  
  // Clicked location (for map click assessment)
  clickedLocation: null,
  setClickedLocation: (location) => set({ clickedLocation: location }),
  
  // Fire data
  fireLocations: [],
  setFireLocations: (locations) => set({ fireLocations: locations }),
  
  // Environmental data
  envData: null,
  setEnvData: (data) => set({ envData: data }),
  
  // Risk assessment
  riskAssessment: null,
  setRiskAssessment: (assessment) => set({ riskAssessment: assessment }),
  
  // Map center
  mapCenter: [DEFAULT_LOCATION.lat, DEFAULT_LOCATION.lng],
  setMapCenter: (center) => set({ mapCenter: center }),
  
  // Weather tab expanded state
  isWeatherExpanded: false,
  setWeatherExpanded: (expanded) => set({ isWeatherExpanded: expanded }),
  
  // Loading states
  isLoading: false,
  setIsLoading: (loading) => set({ isLoading: loading }),
  
  // Active panel
  activePanel: 'location',
  setActivePanel: (panel) => set({ activePanel: panel }),
}))
