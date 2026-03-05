import { motion, AnimatePresence } from 'framer-motion'
import { useStore } from '../store/useStore'
import { GlassCard, Skeleton } from './ui/GlassCard'
import { useEnvData } from '../hooks/useApi'

// Weather icon components
function WeatherIcon({ className = '' }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M2 12h2M20 12h2M6.34 17.66l-1.41 1.41M19.07 4.93l-1.41 1.41" />
      <circle cx="12" cy="12" r="4" />
    </svg>
  )
}

function ThermometerIcon({ className = '' }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M14 14.76V3.5a2.5 2.5 0 0 0-5 0v11.26a4.5 4.5 0 1 0 5 0z" />
    </svg>
  )
}

function DropletIcon({ className = '' }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M12 2.69l5.66 5.66a8 8 0 1 1-11.31 0z" />
    </svg>
  )
}

function WindIcon({ className = '' }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M9.59 4.59A2 2 0 1 1 11 8H2m10.59 11.41A2 2 0 1 0 14 16H2m15.73-8.27A2.5 2.5 0 1 1 19.5 12H2" />
    </svg>
  )
}

function CloudIcon({ className = '' }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M18 10h-1.26A8 8 0 1 0 9 20h9a5 5 0 0 0 0-10z" />
    </svg>
  )
}

function LeafIcon({ className = '' }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M11 20A7 7 0 0 1 9.8 6.1C15.5 5 17 4.48 19 2c1 2 2 4.18 2 8 0 5.5-4.78 10-10 10Z" />
      <path d="M2 21c0-3 1.85-5.36 5.08-6C9.5 14.52 12 13 13 12" />
    </svg>
  )
}

function SunIcon({ className = '' }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <circle cx="12" cy="12" r="5" />
      <path d="M12 1v2M12 21v2M4.22 4.22l1.42 1.42M18.36 18.36l1.42 1.42M1 12h2M21 12h2M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42" />
    </svg>
  )
}

function ExpandIcon({ className = '' }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M6 9l6 6 6-6" />
    </svg>
  )
}

// Single weather metric component
interface WeatherMetricProps {
  label: string
  value: string | number
  unit: string
  icon: React.ReactNode
  color: string
  index: number
}

function WeatherMetric({ label, value, unit, icon, color, index }: WeatherMetricProps) {
  return (
    <motion.div
      initial={{ opacity: 0, x: -20 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ delay: index * 0.1 }}
      className="flex items-center gap-3 p-3 rounded-lg hover:bg-white/5 transition-colors"
    >
      <div
        className="w-10 h-10 rounded-full flex items-center justify-center"
        style={{
          background: `linear-gradient(135deg, ${color}30, ${color}10)`,
          boxShadow: `0 0 12px ${color}30`,
        }}
      >
        <div style={{ color }} className="w-5 h-5">
          {icon}
        </div>
      </div>
      <div className="flex-1">
        <span className="text-[10px] uppercase tracking-wider text-text-secondary block">
          {label}
        </span>
        <span className="text-lg font-mono font-bold" style={{ color }}>
          {typeof value === 'number' ? value.toFixed(1) : value}
          <span className="text-xs text-text-secondary ml-1">{unit}</span>
        </span>
      </div>
    </motion.div>
  )
}

// Weather Tab Component
export function WeatherTab() {
  const { selectedLocation, envData, isWeatherExpanded, setWeatherExpanded } = useStore()
  const { data, isLoading, error } = useEnvData(selectedLocation)

  // Get data from either store or API response
  const weatherData = envData || data?.data

  const metrics = weatherData ? [
    {
      label: 'Temperature',
      value: weatherData.temperature,
      unit: '°C',
      icon: <ThermometerIcon />,
      color: '#FF5722'
    },
    {
      label: 'Humidity',
      value: weatherData.humidity,
      unit: '%',
      icon: <DropletIcon />,
      color: '#00B0FF'
    },
    {
      label: 'Wind Speed',
      value: weatherData.windSpeed,
      unit: 'km/h',
      icon: <WindIcon />,
      color: '#70A1FF'
    },
    {
      label: 'Wind Direction',
      value: weatherData.windDirection,
      unit: '°',
      icon: <WindIcon />,
      color: '#A0A1FF'
    },
    {
      label: 'Cloud Cover',
      value: weatherData.cloudCover ?? 0,
      unit: '%',
      icon: <CloudIcon />,
      color: '#B0BEC5'
    },
    {
      label: 'Vegetation Index',
      value: weatherData.vegetationIndex?.toFixed(4) ?? '0.5000',
      unit: 'NDVI',
      icon: <LeafIcon />,
      color: '#00E676'
    },
    {
      label: 'Drought Index',
      value: weatherData.droughtIndex?.toFixed(4) ?? '0.5000',
      unit: '',
      icon: <SunIcon />,
      color: '#FFA726'
    },
    {
      label: 'Pressure',
      value: weatherData.pressure ?? 1013,
      unit: 'hPa',
      icon: <WeatherIcon />,
      color: '#78909C'
    },
  ] : []

  return (
    <GlassCard glowColor="#00B0FF" delay={0}>
      {/* Header - Always visible */}
      <motion.button
        onClick={() => setWeatherExpanded(!isWeatherExpanded)}
        className="w-full flex items-center justify-between p-2"
        whileHover={{ scale: 1.01 }}
        whileTap={{ scale: 0.99 }}
      >
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-full bg-gradient-to-br from-cyan-500/30 to-blue-500/30 flex items-center justify-center">
            <WeatherIcon className="w-5 h-5 text-cyan-400" />
          </div>
          <div className="text-left">
            <h3 className="text-sm font-semibold text-text-primary">
              Weather Conditions
            </h3>
            <p className="text-xs text-text-secondary">
              {selectedLocation?.name || 'No location selected'}
            </p>
          </div>
        </div>

        <motion.div
          animate={{ rotate: isWeatherExpanded ? 180 : 0 }}
          transition={{ duration: 0.2 }}
          className="w-8 h-8 rounded-full bg-white/10 flex items-center justify-center"
        >
          <ExpandIcon className="w-4 h-4 text-text-secondary" />
        </motion.div>
      </motion.button>

      {/* Summary - Always visible */}
      {weatherData && !isWeatherExpanded && (
        <div className="px-2 pb-2 flex gap-2 overflow-x-auto">
          <div className="flex-shrink-0 px-3 py-1 rounded-full bg-orange-500/20 text-orange-400 text-xs font-medium flex items-center gap-1">
            <ThermometerIcon className="w-3 h-3" /> {weatherData.temperature.toFixed(1)}C
          </div>
          <div className="flex-shrink-0 px-3 py-1 rounded-full bg-blue-500/20 text-blue-400 text-xs font-medium flex items-center gap-1">
            <DropletIcon className="w-3 h-3" /> {weatherData.humidity}%
          </div>
          <div className="flex-shrink-0 px-3 py-1 rounded-full bg-cyan-500/20 text-cyan-400 text-xs font-medium flex items-center gap-1">
            <WindIcon className="w-3 h-3" /> {weatherData.windSpeed} km/h
          </div>
          <div className="flex-shrink-0 px-3 py-1 rounded-full bg-green-500/20 text-green-400 text-xs font-medium flex items-center gap-1">
            <LeafIcon className="w-3 h-3" /> {weatherData.vegetationIndex?.toFixed(4) ?? '0.5000'} NDVI
          </div>
        </div>
      )}

      {/* Expanded content */}
      <AnimatePresence>
        {isWeatherExpanded && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.3 }}
            className="overflow-hidden"
          >
            <div className="pt-2 border-t border-white/10">
              {isLoading ? (
                <div className="flex flex-col gap-2 p-2">
                  <Skeleton className="h-12" />
                  <Skeleton className="h-12" />
                  <Skeleton className="h-12" />
                </div>
              ) : error ? (
                <div className="p-4 text-center">
                  <p className="text-red-400 text-sm">Failed to load weather data</p>
                  <p className="text-text-secondary text-xs mt-1">{String(error)}</p>
                </div>
              ) : weatherData ? (
                <div className="grid grid-cols-2 gap-2 p-2">
                  {metrics.map((metric, index) => (
                    <WeatherMetric
                      key={metric.label}
                      label={metric.label}
                      value={metric.value}
                      unit={metric.unit}
                      icon={metric.icon}
                      color={metric.color}
                      index={index}
                    />
                  ))}
                </div>
              ) : (
                <div className="p-4 text-center">
                  <p className="text-text-secondary text-sm">Select a location to view weather</p>
                </div>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </GlassCard>
  )
}
