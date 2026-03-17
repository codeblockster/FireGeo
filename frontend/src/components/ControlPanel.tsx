import { motion } from 'framer-motion'
import { useStore } from '../store/useStore'
import { GlassCard, GlassInput, GlassButton } from './ui/GlassCard'
import { useEnvData, useAssessRisk, useDetectFires, useSimulateFireSpread } from '../hooks/useApi'
import { useEffect, useState } from 'react'
import { WeatherTab } from './WeatherTab'

// Icon components

function AlertIcon({ className = '' }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z" />
      <line x1="12" y1="9" x2="12" y2="13" />
      <line x1="12" y1="17" x2="12.01" y2="17" />
    </svg>
  )
}

// Risk Gauge Component
function RiskGauge({ score, level }: { score: number; level: string }) {
  const getColor = () => {
    switch (level) {
      case 'critical': return '#FF1744'
      case 'high': return '#FF6D00'
      case 'medium': return '#FFD600'
      case 'low': return '#00E676'
      default: return '#9C27B0'
    }
  }

  const color = getColor()
  const circumference = 2 * Math.PI * 45
  const progress = (score / 100) * circumference

  return (
    <div className="relative w-40 h-40 mx-auto">
      <svg className="w-full h-full -rotate-90" viewBox="0 0 100 100">
        {/* Background circle */}
        <circle
          cx="50"
          cy="50"
          r="45"
          fill="none"
          stroke="rgba(255,255,255,0.1)"
          strokeWidth="8"
        />
        {/* Progress circle */}
        <motion.circle
          cx="50"
          cy="50"
          r="45"
          fill="none"
          stroke={color}
          strokeWidth="8"
          strokeLinecap="round"
          strokeDasharray={circumference}
          initial={{ strokeDashoffset: circumference }}
          animate={{ strokeDashoffset: circumference - progress }}
          transition={{ duration: 1, delay: 0.3 }}
          style={{
            filter: `drop-shadow(0 0 8px ${color})`,
          }}
        />
      </svg>

      {/* Center text */}
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <motion.span
          initial={{ opacity: 0, scale: 0.5 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.5 }}
          className="text-3xl font-mono font-bold"
          style={{ color }}
        >
          {score.toFixed(4)}
        </motion.span>
        <span className="text-[10px] uppercase tracking-widest text-text-secondary">
          Risk Score
        </span>
      </div>
    </div>
  )
}

// Risk Factor Bar Component - Solid Style with visible text
function RiskFactorBar({ label, value, color }: { label: string; value: number; color: string }) {
  return (
    <div className="flex items-center gap-2">
      <span className="text-xs text-black w-20 truncate font-medium">{label}</span>
      <div className="flex-1 h-2 bg-gray-800 rounded-full overflow-hidden border border-gray-600">
        <motion.div
          initial={{ width: 0 }}
          animate={{ width: `${value}%` }}
          transition={{ duration: 0.5, delay: 0.2 }}
          className="h-full rounded-full"
          style={{ backgroundColor: color }}
        />
      </div>
      <span className="text-xs font-mono w-8 text-right font-bold" style={{ color }}>{value}%</span>
    </div>
  )
}

// Control Panel
export function ControlPanel() {
  const { selectedLocation, setSelectedLocation, riskAssessment, mode, fireTimeFrame, setFireTimeFrame } = useStore()
  const [locationName, setLocationName] = useState(selectedLocation?.name || 'Kathmandu Valley')

  // PostFire state
  const [isIgnitionMode, setIsIgnitionMode] = useState(false)
  const [windDirection, setWindDirection] = useState(90)
  const [windSpeed, setWindSpeed] = useState(15)
  const [spreadResult, setSpreadResult] = useState<any>(null)
  const [spreadProgress, setSpreadProgress] = useState(0)
  const [loading, setLoading] = useState(false)
  const [ignitionCoords, setIgnitionCoords] = useState<{ lat: number, lng: number } | null>(null)
  const [simDuration, setSimDuration] = useState<number>(5) // Default 5 time steps
  const [windFetchStatus, setWindFetchStatus] = useState<'idle' | 'loading' | 'done' | 'error'>('idle')
  const [simError, setSimError] = useState<string | null>(null)

  // Use the post-fire spread simulation hook (always call - React hooks must be unconditional)
  const spreadHook = useSimulateFireSpread()
  const simulateAtLocation = spreadHook?.simulateAtLocation
  // spreadHook.isLoading is available if needed

  const fetchWeatherForLocation = async (lat: number, lng: number) => {
    setWindFetchStatus('loading')
    try {
      const response = await fetch(`/api/weather?lat=${lat}&lon=${lng}`);
      if (response.ok) {
        const result = await response.json();
        const data = result.data;
        if (data && data.wind_direction !== undefined) setWindDirection(Math.round(data.wind_direction));
        if (data && data.wind_speed !== undefined) setWindSpeed(Math.round(data.wind_speed * 3.6));
        setWindFetchStatus('done')
      } else {
        setWindFetchStatus('error')
      }
    } catch (e) {
      console.error("Failed to fetch weather for ignition point", e);
      setWindFetchStatus('error')
    }
  }

  // Handle ignition - click on map sets coordinates, button triggers simulation
  const handleIgniteClick = async () => {
    if (isIgnitionMode) {
      // Cancel mode
      setIsIgnitionMode(false)
      window.dispatchEvent(new CustomEvent('setIgnitionMode', { detail: false }))
    } else {
      // Enter ignition mode
      setIsIgnitionMode(true)
      window.dispatchEvent(new CustomEvent('setIgnitionMode', { detail: true }))
    }
  }

  // Function to set ignition point from map clicks
  const setIgnitionPoint = (lat: number, lng: number) => {
    setIgnitionCoords({ lat, lng })
    setIsIgnitionMode(false)
    window.dispatchEvent(new CustomEvent('setIgnitionMode', { detail: false }))
    // Notify map to show ignition marker immediately
    window.dispatchEvent(new CustomEvent('setIgnitionOnMap', { detail: { lat, lng } }))
    fetchWeatherForLocation(lat, lng)
  }

  // Run the fire spread simulation
  const handleRunSimulation = async () => {
    if (!ignitionCoords || !simulateAtLocation) {
      return
    }

    setLoading(true)
    setSpreadProgress(0)
    setSimError(null)

    try {
      const data = await simulateAtLocation(
        ignitionCoords.lat,
        ignitionCoords.lng,
        windDirection,
        windSpeed,
        simDuration
      )

      setSpreadResult(data)

      // Animate progress
      let progress = 0
      const animate = () => {
        progress += 2
        setSpreadProgress(Math.min(progress, 100))
        if (progress < 100) {
          requestAnimationFrame(animate)
        }
      }
      animate()

      // Show fire on map
      window.dispatchEvent(new CustomEvent('showFireSpread', { detail: data }))

    } catch (err: any) {
      console.error('Fire spread simulation failed:', err)
      setSimError(err?.message || 'Simulation failed. Check backend logs.')
    } finally {
      setLoading(false)
    }
  }

  // Expose functions for map clicks
  useEffect(() => {
    const w = window as any
    w.igniteSetPoint = setIgnitionPoint
    w.igniteSetEnabled = (enabled: boolean) => setIsIgnitionMode(enabled)
    w.igniteGetWind = () => ({ direction: windDirection, speed: windSpeed })
    w.igniteSetResult = (result: any) => setSpreadResult(result)
    w.igniteSetProgress = (p: number) => setSpreadProgress(p)
    w.igniteSetLoading = (l: boolean) => setLoading(l)

    // Listen for map click events to set ignition point
    const handleMapClick = (e: CustomEvent) => {
      const { lat, lng } = e.detail
      setIgnitionPoint(lat, lng)
    }
    w.addEventListener('mapClickForIgnition', handleMapClick)

    return () => {
      w.removeEventListener('mapClickForIgnition', handleMapClick)
    }
  }, [windDirection, windSpeed, ignitionCoords])

  const handleResetFire = () => {
    setIsIgnitionMode(false)
    setIgnitionCoords(null)
    setSpreadResult(null)
    setSpreadProgress(0)
    setWindFetchStatus('idle')
    setSimError(null)
    window.dispatchEvent(new Event('resetFire'))
  }

  const { data: envDataResponse } = useEnvData(selectedLocation)
  const { data: fireData, refetch: detectFires, isLoading: fireLoading } = useDetectFires(mode === 'fire' ? selectedLocation : null, fireTimeFrame)
  const assessRisk = useAssessRisk()

  // Update envData from response
  useEffect(() => {
    if (envDataResponse?.data) {
      setSelectedLocation(selectedLocation)
    }
  }, [envDataResponse, selectedLocation, setSelectedLocation])

  // Trigger risk assessment when location changes in risk mode
  useEffect(() => {
    if (mode === 'risk' && selectedLocation) {
      assessRisk.mutate({ location: selectedLocation })
    }
  }, [selectedLocation?.lat, selectedLocation?.lng, mode])

  // Render PostFire Panel
  if (mode === 'postfire') {

    return (
      <motion.div
        initial={{ opacity: 0, x: 20 }}
        animate={{ opacity: 1, x: 0 }}
        transition={{ duration: 0.4 }}
        className="h-full flex flex-col gap-4 overflow-y-auto p-4"
      >
        <GlassCard glowColor="#FF4400">
          <h3 className="text-sm font-semibold text-text-primary mb-4 text-center">
            Post-Fire Spread
          </h3>

          {/* Ignition Configuration */}
          <div className="space-y-2 mb-4">
            <button
              onClick={handleIgniteClick}
              disabled={loading}
              className={`w-full py-3 px-4 rounded-lg font-bold text-white transition-all ${isIgnitionMode
                ? 'bg-gray-600 hover:bg-gray-500'
                : ignitionCoords
                  ? 'bg-blue-600 hover:bg-blue-500 shadow-lg shadow-blue-500/20'
                  : 'bg-red-600 hover:bg-red-500 shadow-lg shadow-red-500/30'
                } ${loading ? 'opacity-50 cursor-not-allowed' : ''}`}
            >
              {isIgnitionMode ? '✕ Cancel' : ignitionCoords ? '📍 CHANGE IGNITION' : '🎯 SET IGNITION'}
            </button>

            {isIgnitionMode && (
              <p className="text-xs text-orange-400 text-center animate-pulse">
                Click on the map to set ignition point
              </p>
            )}

            {ignitionCoords && !isIgnitionMode && (
              <div className="text-[10px] text-green-400 text-center p-1 bg-green-900/20 rounded border border-green-500/20">
                Selected: {ignitionCoords.lat.toFixed(4)}, {ignitionCoords.lng.toFixed(4)}
              </div>
            )}
          </div>

          {/* Run & Reset Actions */}
          <div className="grid grid-cols-4 gap-2 mb-4">
            <button
              onClick={handleRunSimulation}
              disabled={!ignitionCoords || loading}
              className={`col-span-3 py-3 px-4 rounded-lg font-bold text-white transition-all ${!ignitionCoords || loading
                ? 'bg-gray-700 text-gray-400 cursor-not-allowed'
                : 'bg-orange-600 hover:bg-orange-500 shadow-lg shadow-orange-500/30'
                }`}
            >
              {loading ? (
                <span className="flex items-center justify-center gap-2">
                  <span className="animate-spin text-lg">⏳</span> Simulating...
                </span>
              ) : '▶ RUN SIMULATION'}
            </button>

            <button
              onClick={handleResetFire}
              disabled={loading || (!ignitionCoords && !spreadResult)}
              className={`col-span-1 rounded-lg flex items-center justify-center transition-all ${loading || (!ignitionCoords && !spreadResult)
                ? 'bg-gray-800 text-gray-600 cursor-not-allowed'
                : 'bg-gray-700 hover:bg-gray-600 text-gray-300'
                }`}
              title="Reset Simulation"
            >
              🔄
            </button>
          </div>

          {/* Auto-fetched Wind Info */}
          <div className="mb-4 p-3 bg-blue-900/20 border border-blue-500/30 rounded-lg">
            <div className="flex items-center justify-between mb-1">
              <span className="text-xs text-blue-300 font-semibold">Wind (Auto from Weather API)</span>
              {windFetchStatus === 'loading' && <span className="text-[10px] text-blue-400 animate-pulse">fetching...</span>}
              {windFetchStatus === 'error' && <span className="text-[10px] text-red-400">fetch failed</span>}
            </div>
            {windFetchStatus === 'idle' ? (
              <p className="text-xs text-text-secondary">Set ignition point to fetch wind data</p>
            ) : windFetchStatus === 'loading' ? (
              <p className="text-xs text-blue-400">Fetching weather at ignition point...</p>
            ) : (
              <div className="grid grid-cols-2 gap-2 text-xs mt-1">
                <div>
                  <span className="text-text-secondary">Direction</span>
                  <p className="font-bold text-white">{windDirection}°
                    <span className="text-blue-300 ml-1">
                      {['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW'][Math.round(windDirection / 22.5) % 16]}
                    </span>
                  </p>
                </div>
                <div>
                  <span className="text-text-secondary">Speed</span>
                  <p className="font-bold text-white">{windSpeed} km/h</p>
                </div>
              </div>
            )}
          </div>

          {/* Simulation Duration */}
          <div className="mb-4">
            <label className="text-xs text-text-secondary block mb-2">
              Simulation Time (Hours): {simDuration}
            </label>
            <input
              type="range"
              min="1"
              max="72"
              value={simDuration}
              onChange={(e) => setSimDuration(parseInt(e.target.value))}
              className="w-full"
            />
            <div className="flex justify-between text-[10px] text-text-secondary px-1">
              <span>1h</span>
              <span>24h</span>
              <span>48h</span>
              <span>72h</span>
            </div>
          </div>

          {/* Reset Button */}
          {spreadResult && (
            <button
              onClick={handleResetFire}
              className="w-full py-2 px-4 rounded-lg bg-white/10 text-text-secondary hover:bg-white/20 transition-all"
            >
              Reset
            </button>
          )}
          {simError && (
            <div className="mt-2 p-3 bg-red-900/30 border border-red-500/40 rounded-lg">
              <p className="text-xs text-red-400 text-center">{simError}</p>
            </div>
          )}
        </GlassCard>

        {/* Results */}
        {spreadResult && !loading && (
          <GlassCard glowColor="#FF4400">
            <h3 className="text-sm font-semibold text-text-primary mb-2 text-center">
              Fire Spread: {spreadProgress}%
            </h3>

            <div className="grid grid-cols-2 gap-2 text-xs">
              <div>
                <span className="text-text-secondary">Probability</span>
                <p className="font-bold text-red-400">{spreadResult.spread_probability}%</p>
              </div>
              <div>
                <span className="text-text-secondary">Wind</span>
                <p className="font-bold">{spreadResult.wind_direction} @ {spreadResult.wind_speed} km/h</p>
              </div>
            </div>
          </GlassCard>
        )}

        {loading && (
          <GlassCard>
            <div className="text-center py-4">
              <p className="text-text-secondary text-sm">Calculating fire spread...</p>
            </div>
          </GlassCard>
        )}
      </motion.div>
    )
  }



  const locations = [
    'World',
    'Nepal',
    'Kathmandu Valley',
    'Pokhara',
    'Chitwan',
    'Himalayan Region',
    'Australia',
    'California',
  ]

  const handleLocationChange = (name: string) => {
    setLocationName(name)
    const coords: Record<string, { lat: number; lng: number }> = {
      'World': { lat: 20.0, lng: 0.0 },
      'Nepal': { lat: 28.3949, lng: 84.1240 },
      'Kathmandu Valley': { lat: 27.7172, lng: 85.3240 },
      'Pokhara': { lat: 28.2096, lng: 83.9856 },
      'Chitwan': { lat: 27.5322, lng: 84.4358 },
      'Himalayan Region': { lat: 28.0, lng: 86.0 },
      'Australia': { lat: -25.2744, lng: 133.7751 },
      'California': { lat: 36.7783, lng: -119.4179 },
    }
    const coord = coords[name]
    if (coord) {
      setSelectedLocation({
        id: name.toLowerCase().replace(/[^a-z]/g, '-'),
        name,
        lat: coord.lat,
        lng: coord.lng,
      })
    }
  }

  const handleDetectFires = () => {
    if (selectedLocation && mode === 'fire') {
      detectFires()
    }
  }

  const handleAssessRisk = () => {
    if (assessRisk.isPending) {
      // Cancel the current request
      assessRisk.cancel()
    } else if (selectedLocation) {
      assessRisk.mutate({ location: selectedLocation })
    }
  }

  // Determine risk color
  const getRiskColor = () => {
    if (!riskAssessment) return '#9C27B0'
    switch (riskAssessment.level) {
      case 'critical': return '#FF1744'
      case 'high': return '#FF6B35'
      case 'medium': return '#FFA502'
      case 'low': return '#2ED573'
      default: return '#9C27B0'
    }
  }

  return (
    <motion.div
      initial={{ opacity: 0, x: 20 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ duration: 0.4, delay: 0.2 }}
      className="h-full flex flex-col gap-4 overflow-y-auto p-4"
    >
      {/* Location Selector */}
      <GlassCard>
        <h3 className="text-sm font-semibold text-text-primary mb-4">
          Location
        </h3>
        <GlassInput
          label="Select Region"
          value={locationName}
          onChange={handleLocationChange}
          type="select"
          options={locations}
        />
      </GlassCard>

      {/* Time Frame Selector - Only visible in Fire mode */}
      {mode === 'fire' && (
        <GlassCard>
          <h3 className="text-sm font-semibold text-text-primary mb-4">
            Time Frame
          </h3>
          <div className="grid grid-cols-4 gap-2">
            {[24, 48, 72, 168].map((hours) => (
              <button
                key={hours}
                onClick={() => setFireTimeFrame(hours)}
                className={`px-3 py-2 rounded-lg text-xs font-medium transition-all ${fireTimeFrame === hours
                  ? 'bg-orange-500/30 text-orange-400 border border-orange-500/50'
                  : 'bg-white/5 text-text-secondary border border-white/10 hover:bg-white/10'
                  }`}
              >
                {hours === 168 ? '7 Days' : hours === 72 ? '3 Days' : hours === 48 ? '48h' : '24h'}
              </button>
            ))}
          </div>
          <p className="text-xs text-text-secondary mt-2 text-center">
            Showing fires from last {fireTimeFrame === 168 ? '7 days' : fireTimeFrame === 72 ? '3 days' : `${fireTimeFrame} hours`}
          </p>
        </GlassCard>
      )}

      {/* Weather Tab - Consolidated expandable weather */}
      <WeatherTab />

      {/* Action Button - Different for each mode */}
      <GlassButton
        onClick={mode === 'fire' ? handleDetectFires : handleAssessRisk}
        glowColor={mode === 'fire' ? "#FF1744" : getRiskColor()}
        disabled={mode === 'fire' && !selectedLocation}
      >
        {mode === 'fire' ? (
          fireLoading ? (
            <span>Scanning...</span>
          ) : (
            <span>Detect Fires</span>
          )
        ) : (
          assessRisk.isPending ? (
            <span>Stop Analysis</span>
          ) : (
            <span>Assess Risk</span>
          )
        )}
      </GlassButton>

      {/* Mode-specific Content */}
      {mode === 'fire' ? (
        /* Fire Detection Results */
        <GlassCard glowColor="#FF1744">
          <h3 className="text-sm font-semibold text-text-primary mb-4 text-center">
            Fire Detection Results
          </h3>

          {!fireData || fireData.count === 0 ? (
            <div className="text-center py-4">
              <p className="text-text-secondary text-sm">
                {fireLoading ? 'Scanning for fires...' : 'No fires detected in this area'}
              </p>
            </div>
          ) : (
            <div className="space-y-2">
              <div className="flex justify-between items-center mb-3">
                <span className="text-sm text-text-secondary">
                  Detected {fireData.count} fire{fireData.count !== 1 ? 's' : ''}
                </span>
                <span className="text-xs text-text-secondary bg-white/10 px-2 py-1 rounded">
                  {fireData.source}
                </span>
              </div>

              {fireData.fires.slice(0, 5).map((fire, idx) => (
                <motion.div
                  key={fire.id}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: idx * 0.1 }}
                  className="p-2 rounded-lg bg-white/5 flex justify-between items-center"
                >
                  <div>
                    <span className="text-xs text-orange-400">Fire #{idx + 1}</span>
                    <p className="text-xs text-text-secondary">
                      {fire.lat.toFixed(4)}, {fire.lng.toFixed(4)}
                    </p>
                  </div>
                  <div className="text-right">
                    <span className="text-sm font-bold text-red-400">{fire.intensity}%</span>
                    <p className="text-xs text-text-secondary">{fire.confidence}% conf.</p>
                  </div>
                </motion.div>
              ))}

              {fireData.count > 5 && (
                <p className="text-xs text-text-secondary text-center mt-2">
                  +{fireData.count - 5} more fires
                </p>
              )}
            </div>
          )}
        </GlassCard>
      ) : (
        /* Risk Assessment Panel */
        <GlassCard glowColor={getRiskColor()}>
          <h3 className="text-sm font-semibold text-text-primary mb-4 text-center">
            Fire Risk Assessment
          </h3>

          {assessRisk.isPending ? (
            <div className="flex flex-col items-center py-6">
              <motion.div
                animate={{ rotate: 360 }}
                transition={{ duration: 2, repeat: Infinity, ease: 'linear' }}
                className="w-12 h-12 border-4 border-purple-500/30 border-t-purple-500 rounded-full mb-4"
              />
              <p className="text-text-secondary text-sm">Analyzing environmental factors...</p>
            </div>
          ) : riskAssessment ? (
            <>
              <RiskGauge score={riskAssessment.score} level={riskAssessment.level} />

              <div className="mt-4 text-center">
                <motion.span
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ delay: 0.6 }}
                  className="text-sm font-medium uppercase tracking-wider"
                  style={{
                    color: getRiskColor()
                  }}
                >
                  {riskAssessment.level} Risk
                </motion.span>
              </div>

              {/* Risk Factors Breakdown */}
              <div className="mt-4 pt-4 border-t border-white/10 space-y-2">
                <p className="text-xs text-text-secondary mb-2">Risk Factors Breakdown</p>
                <RiskFactorBar
                  label="Weather"
                  value={riskAssessment.factors.weather}
                  color="#FF5722"
                />
                <RiskFactorBar
                  label="Vegetation"
                  value={riskAssessment.factors.vegetation}
                  color="#00E676"
                />
                <RiskFactorBar
                  label="Topography"
                  value={riskAssessment.factors.topography}
                  color="#70A1FF"
                />
                <RiskFactorBar
                  label="Historical"
                  value={riskAssessment.factors.historical}
                  color="#FFA502"
                />
              </div>

              {/* AI Info */}
              <div className="mt-3 pt-3 border-t border-white/10">
                <p className="text-xs text-text-secondary">
                  AI Confidence: {riskAssessment.confidence}
                </p>
                <p className="text-xs text-text-secondary mt-1">
                  Alert Priority: {riskAssessment.alert_priority}
                </p>
              </div>
            </>
          ) : (
            <div className="flex flex-col items-center py-6">
              <AlertIcon className="w-12 h-12 text-purple-400 mb-3" />
              <p className="text-text-secondary text-sm text-center">
                Click "Assess Risk" or click on the map to analyze fire risk
              </p>
            </div>
          )}
        </GlassCard>
      )}


    </motion.div>
  )
}
