/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        // Frost Blue Light Theme
        'bg-dark': '#e0f2fe',
        'bg-dark-secondary': '#bae6fd',
        'bg-accent': '#7dd3fc',
        
        // Glass surfaces - White/frost
        'glass-surface': 'rgba(255, 255, 255, 0.7)',
        'glass-border': 'rgba(14, 165, 233, 0.3)',
        'glass-border-focus': 'rgba(14, 165, 233, 0.6)',
        
        // Frost/Ice Blue Accents
        'frost-white': '#0c4a6e',
        'frost-blue': '#0284c7',
        'frost-cyan': '#0891b2',
        
        // Fire accent
        'fire': '#ea580c',
        'fire-amber': '#d97706',
        'fire-red': '#dc2626',
        'fire-yellow': '#ca8a04',
        
        // Risk levels
        'risk-critical': '#ef4444',
        'risk-high': '#f97316',
        'risk-medium': '#eab308',
        'risk-low': '#22c55e',
        
        // Environmental
        'vegetation': '#16a34a',
        'wind': '#3b82f6',
        'water': '#2563eb',
        
        // Text - Pure Black for light bg
        'text-primary': '#000000',
        'text-secondary': '#000000',
        'text-mono': '#000000',
        
        // Accent colors
        'accent-cyan': '#0891b2',
        'accent-blue': '#0284c7',
        'accent-purple': '#9333ea',
        'accent-pink': '#db2777',
        'accent-green': '#16a34a',
        'accent-orange': '#ea580c',
      },
      fontFamily: {
        'sans': ['Inter', 'system-ui', '-apple-system', 'BlinkMacSystemFont', 'sans-serif'],
        'mono': ['SF Mono', 'JetBrains Mono', 'Menlo', 'monospace'],
      },
      borderRadius: {
        'squircle': '20px',
      },
      boxShadow: {
        'glass': '0 4px 16px rgba(0, 0, 0, 0.1), 0 0 20px rgba(14, 165, 233, 0.15)',
        'glass-frost': '0 4px 16px rgba(0, 0, 0, 0.1), 0 0 30px rgba(14, 165, 233, 0.2)',
        'fire-glow': '0 0 20px rgba(234, 88, 12, 0.4)',
        'risk-glow': '0 0 20px rgba(239, 68, 68, 0.4)',
        'frost-glow': '0 0 15px rgba(2, 132, 199, 0.3)',
      },
    },
  },
  plugins: [],
}
