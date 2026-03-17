import { motion, AnimatePresence } from 'framer-motion'
import { useStore } from '../store/useStore'
import { useState } from 'react'
import LocalFireDepartmentIcon from '@mui/icons-material/LocalFireDepartment'

// Full Page Overlay Component for Software Info
function InfoOverlay({ isOpen, onClose }: { isOpen: boolean; onClose: () => void }) {
  return (
    <AnimatePresence>
      {isOpen && (
        <>
          {/* Backdrop */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black/60 backdrop-blur-sm z-[1100]"
            onClick={onClose}
          />

          {/* Full Page Overlay */}
          <motion.div
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            transition={{ type: "spring", stiffness: 300, damping: 25 }}
            className="fixed top-0 left-0 right-0 bottom-0 z-[1100] overflow-y-auto"
          >
            <div className="min-h-screen bg-bg-dark p-8">
              {/* Header with close button */}
              <div className="flex items-center justify-between mb-8 max-w-4xl mx-auto">
                <div className="flex items-center gap-3">
                  <div className="w-12 h-12 rounded-xl flex items-center justify-center shadow-lg overflow-hidden">
                    <img src="/fire-logo.png" alt="Logo" className="w-full h-full object-contain" />
                  </div>
                  <div>
                    <h2 className="text-2xl font-bold text-red-600">FIRE ALERT</h2>
                    <p className="text-sm text-black/70">Wildfire Detection and Risk Assessment System</p>
                  </div>
                </div>
                <button
                  onClick={onClose}
                  className="w-10 h-10 rounded-full bg-white/20 hover:bg-white/30 flex items-center justify-center transition-colors"
                >
                  <svg className="w-6 h-6 text-black" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </button>
              </div>

              {/* Content */}
              <div className="max-w-4xl mx-auto space-y-6">
                <p className="text-lg text-black">
                  Wildfire Watch is an advanced wildfire detection and risk assessment system that combines satellite data with AI-powered predictions.
                </p>

                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  {/* Fire Detection */}
                  <div className="bg-white/50 rounded-2xl p-6 border border-black/10">
                    <div className="flex items-center gap-3 mb-4">
                      <SearchIcon className="w-8 h-8 text-orange-500" />
                      <h3 className="text-lg font-semibold text-orange-500">Fire Detection</h3>
                    </div>
                    <p className="text-black/80">
                      Uses NASA FIRMS satellite data to detect active fires in near real-time. Monitors VIIRS and MODIS sensors for accurate fire hotspots.
                    </p>
                  </div>

                  {/* Risk Assessment */}
                  <div className="bg-white/50 rounded-2xl p-6 border border-black/10">
                    <div className="flex items-center gap-3 mb-4">
                      <BrainIcon className="w-8 h-8 text-purple-500" />
                      <h3 className="text-lg font-semibold text-purple-500">AI Risk Assessment</h3>
                    </div>
                    <p className="text-black/80">
                      Uses CatBoost ML model trained on environmental factors (temperature, humidity, wind, vegetation) to predict fire risk probability.
                    </p>
                  </div>

                  {/* Weather Data */}
                  <div className="bg-white/50 rounded-2xl p-6 border border-black/10">
                    <div className="flex items-center gap-3 mb-4">
                      <CloudIcon className="w-8 h-8 text-blue-500" />
                      <h3 className="text-lg font-semibold text-blue-500">Weather Integration</h3>
                    </div>
                    <p className="text-black/80">
                      Fetches real-time weather data from Open-Meteo API including temperature, humidity, wind speed, and precipitation.
                    </p>
                  </div>
                </div>

                <div className="pt-6 border-t border-black/10">
                  <p className="text-center text-black/60">
                    Powered by NASA FIRMS | Open-Meteo | CatBoost AI
                  </p>
                </div>
              </div>
            </div>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  )
}

// SVG Icons
function FireIcon({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none">
      <path d="M12 2C12 2 8 6 8 10C8 14 10 16 12 16C14 16 16 14 16 10C16 6 12 2 12 2Z" fill="currentColor" />
      <path d="M12 16C12 16 6 20 6 22C6 23.1 6.9 24 8 24H16C17.1 24 18 23.1 18 22C18 20 12 16 12 16Z" fill="currentColor" />
    </svg>
  )
}

function SearchIcon({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <circle cx="11" cy="11" r="8" />
      <path d="M21 21l-4.35-4.35" />
    </svg>
  )
}

function BrainIcon({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M12 2a9 9 0 0 1 9 9c0 3.5-2 6.5-5 8l-1 1-1-1c-3-1.5-5-4.5-5-8a9 9 0 0 1 9-9z" />
      <path d="M12 8v4M12 16h.01" />
    </svg>
  )
}

function CloudIcon({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M18 10h-1.26A8 8 0 1 0 9 20h9a5 5 0 0 0 0-10z" />
    </svg>
  )
}

// Modern Toggle Component with smooth animation
// Order: Fire → Risk → Post Fire
export function ModeToggle() {
  const { mode, setMode } = useStore()
  const isRiskMode = mode === 'risk'
  const isPostFireMode = mode === 'postfire'
  const isFireMode = !isRiskMode && !isPostFireMode

  // Get current mode label
  const getModeLabel = () => {
    if (isFireMode) return 'FIRE'
    if (isRiskMode) return 'RISK'
    return 'SPREAD'
  }

  return (
    <motion.div
      className="flex items-center gap-3"
      initial={{ opacity: 0, scale: 0.9 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.3 }}
    >
      {/* Fire Mode - Active = Orange */}
      <motion.div
        className={`flex items-center gap-1.5 cursor-pointer px-2 py-1 rounded-lg transition-all ${isFireMode ? 'bg-orange-500/20' : 'opacity-60 hover:opacity-100'}`}
        onClick={() => setMode('fire')}
        whileHover={{ scale: 1.05 }}
        whileTap={{ scale: 0.95 }}
      >
        <FireIcon className={`w-4 h-4 ${isFireMode ? 'text-orange-400' : 'text-text-secondary'}`} />
        <span className={`text-xs font-bold ${isFireMode ? 'text-orange-400' : 'text-text-secondary'}`}>
          Fire
        </span>
      </motion.div>

      {/* Toggle Switch - Shows current mode */}
      <motion.button
        className="relative h-7 px-3 rounded-full cursor-pointer overflow-hidden flex items-center justify-center"
        onClick={() => {
          if (mode === 'fire') setMode('risk')
          else if (mode === 'risk') setMode('postfire')
          else setMode('fire')
        }}
        style={{
          background: isRiskMode
            ? 'linear-gradient(135deg, #7c3aed, #a855f7)'
            : isPostFireMode
              ? 'linear-gradient(135deg, #dc2626, #ef4444)'
              : 'linear-gradient(135deg, #ea580c, #f97316)',
        }}
        whileHover={{ scale: 1.05 }}
        whileTap={{ scale: 0.95 }}
      >
        <span className="text-xs font-bold text-white z-10">
          {getModeLabel()}
        </span>
        {/* Arrow hint */}
        <span className="absolute right-1 text-[10px] text-white/70">▶</span>
      </motion.button>

      {/* Risk Mode - Active = Purple */}
      <motion.div
        className={`flex items-center gap-1.5 cursor-pointer px-2 py-1 rounded-lg transition-all ${isRiskMode ? 'bg-purple-500/20' : 'opacity-60 hover:opacity-100'}`}
        onClick={() => setMode('risk')}
        whileHover={{ scale: 1.05 }}
        whileTap={{ scale: 0.95 }}
      >
        <WarningIcon className={`w-4 h-4 ${isRiskMode ? 'text-purple-400' : 'text-text-secondary'}`} />
        <span className={`text-xs font-bold ${isRiskMode ? 'text-purple-400' : 'text-text-secondary'}`}>
          Risk
        </span>
      </motion.div>



      {/* Post Fire Mode - Active = Red */}
      <motion.div
        className={`flex items-center gap-1.5 cursor-pointer px-2 py-1 rounded-lg transition-all ${isPostFireMode ? 'bg-red-500/20' : 'opacity-60 hover:opacity-100'}`}
        onClick={() => setMode('postfire')}
        whileHover={{ scale: 1.05 }}
        whileTap={{ scale: 0.95 }}
      >
        <LocalFireDepartmentIcon className={`w-4 h-4 ${isPostFireMode ? 'text-red-400' : 'text-text-secondary'}`} />
        <span className={`text-xs font-bold ${isPostFireMode ? 'text-red-400' : 'text-text-secondary'}`}>
          Spread
        </span>
      </motion.div>
    </motion.div>
  )
}

function WarningIcon({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M12 9v4M12 17h.01" />
      <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z" />
    </svg>
  )
}

export function Navbar() {
  const [showInfo, setShowInfo] = useState(false)

  return (
    <>
      <motion.nav
        initial={{ y: -20, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ duration: 0.4 }}
        className="h-20 glass border-b border-glass-border flex items-center justify-between px-6"
      >
        {/* Logo / App Name - FIRE ALERT Style */}
        <motion.div
          className="flex items-center gap-3 cursor-pointer"
          onClick={() => setShowInfo(!showInfo)}
          whileHover={{ scale: 1.02 }}
          whileTap={{ scale: 0.98 }}
        >
          {/* Fire Logo Image */}
          <motion.div
            className="relative w-12 h-12 rounded-lg flex items-center justify-center overflow-hidden"
          >
            <img
              src="/fire-logo.png"
              alt="FIRE ALERT Logo"
              className="w-full h-full object-contain"
            />
          </motion.div>

          <div className="flex flex-col">
            <span className="text-xl font-bold text-red-600 tracking-tight">
              FIRE ALERT
            </span>
            <span className="text-[9px] uppercase tracking-widest text-black/60">
              Wildfire Detection System
            </span>
          </div>
        </motion.div>

        {/* Mode Toggle */}
        <ModeToggle />
      </motion.nav>

      {/* Info Overlay */}
      <InfoOverlay isOpen={showInfo} onClose={() => setShowInfo(false)} />
    </>
  )
}
