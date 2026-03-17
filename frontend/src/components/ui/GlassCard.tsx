import React from 'react'
import { motion } from 'framer-motion'
import { ReactNode, useRef } from 'react'

interface GlassCardProps {
  children: ReactNode
  className?: string
  glowColor?: string
  delay?: number
}

export function GlassCard({ children, className = '', glowColor, delay = 0 }: GlassCardProps) {
  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.4, delay: delay * 0.1, ease: 'easeOut' }}
      className={`
        glass squircle p-4 ice-crystal
        ${glowColor ? `border-b-2` : ''}
        ${className}
      `}
      style={{
        borderBottomColor: glowColor ? `${glowColor}40` : undefined,
        boxShadow: glowColor ? `0 4px 20px -4px ${glowColor}30` : undefined,
      }}
    >
      {children}
    </motion.div>
  )
}

interface GlassInputProps {
  label: string
  value: string | number
  onChange: (value: string) => void
  type?: 'text' | 'number' | 'select'
  options?: string[]
  placeholder?: string
}

export function GlassInput({ label, value, onChange, type = 'text', options = [], placeholder }: GlassInputProps) {
  return (
    <div className="flex flex-col gap-2">
      <label className="text-xs uppercase tracking-wider text-text-secondary font-medium">
        {label}
      </label>
      {type === 'select' ? (
        <select
          value={value}
          onChange={(e) => onChange(e.target.value)}
          className="
            glass glass-focus 
            px-4 py-3 
            text-text-primary 
            font-mono text-sm
            squircle
            cursor-pointer
            appearance-none
            bg-[url('data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTIiIGhlaWdodD0iOCIgdmlld0JveD0iMCAwIDEyIDgiIGZpbGw9Im5vbmUiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+PHBhdGggZD0iTTEgMS41TDYgNi41TDExIDEuNSIgc3Ryb2tlPSIjRjVGNUY1IiBzdHJva2Utd2lkdGg9IjEuNSIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIiBzdHJva2UtbGluZWpvaW49InJvdW5kIi8+PC9zdmc+')]
            bg-[length:12px] bg-[right_12px_center] bg-no-repeat
          "
        >
          {options.map((option) => (
            <option key={option} value={option} className="bg-bg-dark">
              {option}
            </option>
          ))}
        </select>
      ) : (
        <input
          type={type}
          value={value}
          onChange={(e) => onChange(e.target.value)}
          placeholder={placeholder}
          className="
            glass glass-focus 
            px-4 py-3 
            text-text-primary font-mono text-sm
            squircle
            placeholder:text-text-secondary
          "
        />
      )}
    </div>
  )
}

interface GlassButtonProps {
  children: ReactNode
  onClick?: () => void
  variant?: 'primary' | 'secondary' | 'danger'
  glowColor?: string
  disabled?: boolean
  className?: string
}

export function GlassButton({
  children,
  onClick,
  variant = 'primary',
  glowColor = '#FF5722',
  disabled = false,
  className = ''
}: GlassButtonProps) {
  const getVariantStyles = () => {
    switch (variant) {
      case 'danger':
        return 'text-risk-critical'
      case 'secondary':
        return 'text-text-secondary hover:text-text-primary'
      default:
        return 'text-fire'
    }
  }

  return (
    <motion.button
      whileHover={{ scale: 1.02 }}
      whileTap={{ scale: 0.98 }}
      onClick={onClick}
      disabled={disabled}
      className={`
        liquid-shimmer
        glass squircle
        px-6 py-3
        ${getVariantStyles()}
        font-medium text-sm uppercase tracking-wider
        disabled:opacity-50 disabled:cursor-not-allowed
        ${className}
      `}
      style={{
        '--glow-color': glowColor,
      } as React.CSSProperties}
    >
      {children}
    </motion.button>
  )
}

interface ModeToggleProps {
  modes: { id: string; label: string; color: string }[]
  activeMode: string
  onModeChange: (mode: string) => void
}

export function ModeToggle({ modes, activeMode, onModeChange }: ModeToggleProps) {
  return (
    <div className="flex gap-2 p-1 glass squircle">
      {modes.map((mode) => {
        const isActive = activeMode === mode.id
        return (
          <motion.button
            key={mode.id}
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
            onClick={() => onModeChange(mode.id)}
            className={`
              relative px-4 py-2 rounded-xl text-sm font-medium transition-colors
              ${isActive ? 'text-white' : 'text-text-secondary hover:text-text-primary'}
            `}
            style={{
              background: isActive ? `linear-gradient(135deg, ${mode.color}30, ${mode.color}10)` : 'transparent',
              boxShadow: isActive ? `0 0 20px ${mode.color}40, inset 0 1px 0 ${mode.color}60` : 'none',
            }}
          >
            {isActive && (
              <motion.div
                layoutId="activeGlow"
                className="absolute inset-0 rounded-xl"
                style={{
                  background: `radial-gradient(circle at center, ${mode.color}30 0%, transparent 70%)`,
                }}
                transition={{ type: 'spring', bounce: 0.2, duration: 0.6 }}
              />
            )}
            <span className="relative z-10">{mode.label}</span>
          </motion.button>
        )
      })}
    </div>
  )
}

interface SkeletonProps {
  className?: string
}

export function Skeleton({ className = '' }: SkeletonProps) {
  return (
    <div className={`frost-shimmer squircle ${className}`} />
  )
}

interface MetricCardProps {
  label: string
  value: string | number
  unit?: string
  glowColor?: string
  delay?: number
}

export function MetricCard({ label, value, unit, glowColor = '#FF5722', delay = 0 }: MetricCardProps) {
  return (
    <GlassCard delay={delay} glowColor={glowColor}>
      <div className="flex flex-col items-center gap-1">
        <motion.span
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: delay * 0.1 + 0.3 }}
          className="text-3xl font-mono font-bold"
          style={{ color: glowColor }}
        >
          {value}
        </motion.span>
        {unit && (
          <span className="text-xs text-text-secondary uppercase tracking-wider">
            {unit}
          </span>
        )}
        <span className="text-[10px] uppercase tracking-widest text-text-secondary mt-1">
          {label}
        </span>
      </div>
    </GlassCard>
  )
}

// Frost Toggle Component - Custom SVG design for Fire Detect / Risk Assessment
export function FrostToggle({ isActive, onToggle }: { isActive: boolean; onToggle: () => void }) {
  const handleRef = useRef<SVGGElement>(null)

  // Calculate handle position
  const targetX = isActive ? 180 : 60

  return (
    <div
      className="relative cursor-pointer select-none"
      onClick={onToggle}
    >
      <svg
        viewBox="0 0 300 120"
        width="280"
        height="112"
        className="overflow-visible"
      >
        <defs>
          {/* Frost Light Track Gradient */}
          <linearGradient id="track-light-frost" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stopColor="#e2e8f0" />
            <stop offset="100%" stopColor="#f8fafc" />
          </linearGradient>

          {/* Fire Dark Track Gradient */}
          <linearGradient id="track-dark-fire" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stopColor="#3f0808" />
            <stop offset="50%" stopColor="#7f1d1d" />
            <stop offset="100%" stopColor="#b91c1c" />
          </linearGradient>

          {/* Frost Handle Gradient */}
          <linearGradient id="frost-handle" x1="0%" y1="0%" x2="0%" y2="100%">
            <stop offset="0%" stopColor="#ffffff" stopOpacity="0.95" />
            <stop offset="100%" stopColor="#ffffff" stopOpacity="0.3" />
          </linearGradient>

          {/* Handle Highlight */}
          <linearGradient id="frost-highlight" x1="0%" y1="0%" x2="0%" y2="100%">
            <stop offset="0%" stopColor="#ffffff" stopOpacity="0.9" />
            <stop offset="100%" stopColor="#ffffff" stopOpacity="0.0" />
          </linearGradient>

          {/* Fire Right BG */}
          <linearGradient id="fire-right-bg" x1="0%" y1="0%" x2="0%" y2="100%">
            <stop offset="0%" stopColor="#ff9800" />
            <stop offset="40%" stopColor="#f44336" />
            <stop offset="100%" stopColor="#990000" />
          </linearGradient>

          {/* Fire Right Core */}
          <linearGradient id="fire-right-core" x1="0%" y1="0%" x2="0%" y2="100%">
            <stop offset="0%" stopColor="#ffff00" />
            <stop offset="40%" stopColor="#ff9800" />
            <stop offset="100%" stopColor="#ff3d00" />
          </linearGradient>

          {/* Fire Left BG */}
          <linearGradient id="fire-left-bg" x1="0%" y1="0%" x2="0%" y2="100%">
            <stop offset="0%" stopColor="#ff9800" />
            <stop offset="60%" stopColor="#e53935" />
            <stop offset="100%" stopColor="#c62828" />
          </linearGradient>

          {/* Fire Left Rim */}
          <linearGradient id="fire-left-rim" x1="0%" y1="0%" x2="0%" y2="100%">
            <stop offset="0%" stopColor="#ffff00" />
            <stop offset="100%" stopColor="#ffc107" />
          </linearGradient>

          {/* Risk Shield */}
          <linearGradient id="risk-shield" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stopColor="#ffffff" />
            <stop offset="100%" stopColor="#fca5a5" />
          </linearGradient>

          {/* Shadows */}
          <filter id="shadow-large" x="-50%" y="-50%" width="200%" height="200%">
            <feDropShadow dx="0" dy="8" stdDeviation="10" floodColor="#000000" floodOpacity="0.15" />
          </filter>
          <filter id="shadow-tight" x="-30%" y="-30%" width="160%" height="160%">
            <feDropShadow dx="0" dy="3" stdDeviation="4" floodColor="#000000" floodOpacity="0.25" />
          </filter>
        </defs>

        {/* Labels - Fire Detect */}
        <text
          x="60" y="108"
          textAnchor="middle"
          className="text-[11px] font-semibold"
          fill={!isActive ? '#374151' : '#9ca3af'}
          style={{ fontFamily: 'Inter, system-ui, sans-serif', transition: 'fill 0.3s' }}
        >
          Fire Detect
        </text>

        {/* Labels - Risk Assessment */}
        <text
          x="240" y="108"
          textAnchor="middle"
          className="text-[11px] font-semibold"
          fill={isActive ? '#374151' : '#9ca3af'}
          style={{ fontFamily: 'Inter, system-ui, sans-serif', transition: 'fill 0.3s' }}
        >
          Risk Assessment
        </text>

        {/* Interactive Area */}
        <g className="cursor-pointer">
          {/* Track Background - Light Frost */}
          <rect
            x="20" y="20"
            width="260" height="70"
            rx="35"
            fill="url(#track-light-frost)"
          />

          {/* Track Background - Dark Fire (fades in when active) */}
          <rect
            x="20" y="20"
            width="260" height="70"
            rx="35"
            fill="url(#track-dark-fire)"
            style={{
              opacity: isActive ? 1 : 0,
              transition: 'opacity 0.5s ease-in-out'
            }}
          />

          {/* Inner Shadow Overlay */}
          <rect
            x="20" y="20"
            width="260" height="70"
            rx="35"
            fill="url(#track-depth)"
            style={{ pointerEvents: 'none' }}
          />

          {/* Sliding Frost Handle */}
          <motion.g
            ref={handleRef}
            animate={{ x: targetX }}
            transition={{
              type: 'spring',
              stiffness: 300,
              damping: 25,
              mass: 1
            }}
            style={{ cursor: 'pointer' }}
          >
            {/* Shadow */}
            <circle
              cx="60" cy="55"
              r="32"
              fill="transparent"
              filter="url(#shadow-large)"
            />
            <circle
              cx="60" cy="55"
              r="30"
              fill="transparent"
              filter="url(#shadow-tight)"
            />

            {/* Glass Sphere Base */}
            <circle
              cx="60" cy="55"
              r="30"
              fill="url(#frost-handle)"
              stroke="#ffffff"
              strokeWidth="1.5"
              strokeOpacity="0.8"
            />

            {/* Liquid Top Highlight */}
            <ellipse
              cx="60" cy="43"
              rx="18" ry="10"
              fill="url(#frost-highlight)"
            />

            {/* Fire Icon (when NOT active - left side) */}
            <g style={{ opacity: !isActive ? 1 : 0, transition: 'opacity 0.3s' }}>
              <g transform="translate(42, 35) scale(0.7)">
                {/* Right Flame Layer 1: Darker Red Outer */}
                <path d="M 1,-28 C 12,-10 25,5 20,24 C 16,34 4,36 -5,34 C 12,28 12,8 0,-4 C -6,-10 -4,-20 1,-28 Z" fill="url(#fire-right-bg)" />
                {/* Right Flame Layer 2: Inner Core */}
                <path d="M 0,-26 C 6,-10 18,3 15,20 C 13,28 2,32 -3,32 C 10,26 10,8 0,-2 C -4,-8 -2,-18 0,-26 Z" fill="url(#fire-right-core)" />
                {/* Left Flame Layer 1: Red Body */}
                <path d="M -4,-18 C -14,-8 -18,-2 -10,4 C -18,6 -26,16 -20,28 C -15,34 -5,34 0,34 C -14,26 -12,12 -2,2 C 4,-4 2,-12 -4,-18 Z" fill="url(#fire-left-bg)" />
                {/* Left Flame Layer 2: Rim Highlight */}
                <path d="M -4,-18 C -12,-10 -18,-2 -10,4 C -16,-1 -16,-7 -4,-18 Z" fill="url(#fire-left-rim)" />
                <path d="M -10,4 C -18,7 -26,16 -20,28 C -26,16 -18,8 -10,4 Z" fill="url(#fire-left-rim)" />
              </g>
            </g>

            {/* Shield Icon (when ACTIVE - right side) */}
            <g style={{ opacity: isActive ? 1 : 0, transition: 'opacity 0.3s' }}>
              <g transform="translate(60, 55) scale(1.3)">
                <path
                  d="M 0,-12 L -10,-8 L -10,2 C -10,8 -4,12 0,15 C 4,12 10,8 10,2 L 10,-8 Z"
                  fill="url(#risk-shield)"
                />
                <polyline
                  points="-5,4 -1,-1 2,2 5,-3"
                  stroke="#b91c1c"
                  strokeWidth="2"
                  fill="none"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
                <polyline
                  points="2,-3 5,-3 5,1"
                  stroke="#b91c1c"
                  strokeWidth="2"
                  fill="none"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
              </g>
            </g>
          </motion.g>
        </g>
      </svg>
    </div>
  )
}

// Expandable Section Component for Environmental Data
interface ExpandableSectionProps {
  title: string
  icon: ReactNode
  color: string
  isExpanded: boolean
  onToggle: () => void
  children: ReactNode
  delay?: number
}

export function ExpandableSection({
  title,
  icon,
  color,
  isExpanded,
  onToggle,
  children,
  delay = 0
}: ExpandableSectionProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3, delay: delay * 0.1 }}
      className="glass squircle overflow-hidden"
    >
      {/* Header - Clickable */}
      <button
        onClick={onToggle}
        className="w-full flex items-center gap-3 p-4 hover:bg-white/5 transition-colors"
      >
        <div
          className="w-8 h-8 rounded-full flex items-center justify-center"
          style={{
            background: `linear-gradient(135deg, ${color}30, ${color}10)`,
            boxShadow: `0 0 12px ${color}40, inset 0 1px 0 ${color}60`,
          }}
        >
          <div style={{ color }} className="w-4 h-4">
            {icon}
          </div>
        </div>
        <span className="flex-1 text-left text-sm font-medium text-text-primary">
          {title}
        </span>
        <motion.div
          animate={{ rotate: isExpanded ? 180 : 0 }}
          transition={{ duration: 0.2 }}
          className="text-text-secondary"
        >
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M6 9l6 6 6-6" />
          </svg>
        </motion.div>
      </button>

      {/* Expandable Content */}
      <motion.div
        initial={false}
        animate={{
          height: isExpanded ? 'auto' : 0,
          opacity: isExpanded ? 1 : 0
        }}
        transition={{ duration: 0.3, ease: 'easeInOut' }}
        className="overflow-hidden"
      >
        <div className="px-4 pb-4">
          {children}
        </div>
      </motion.div>
    </motion.div>
  )
}
