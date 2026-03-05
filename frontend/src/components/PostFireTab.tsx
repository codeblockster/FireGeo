import React, { useState, useEffect, useRef } from 'react';
import { GlassCard, GlassButton } from './ui/GlassCard';
import { useSimulateFireSpread, FireSpreadResponse } from '../hooks/useApi';

// Compass Component
const Compass: React.FC<{ value: number; onChange: (angle: number) => void }> = ({ value, onChange }) => {
  const compassRef = useRef<HTMLDivElement>(null);
  const [isDragging, setIsDragging] = useState(false);

  const getAngleFromCenter = (e: React.MouseEvent | MouseEvent, center: DOMRect) => {
    const x = e.clientX - center.left - center.width / 2;
    const y = e.clientY - center.top - center.height / 2;
    // Atan2 gives angle from positive X axis (East), we want from North (0)
    let angle = Math.atan2(x, -y) * (180 / Math.PI);
    if (angle < 0) angle += 360;
    return Math.round(angle);
  };

  const handleMouseDown = (e: React.MouseEvent) => {
    setIsDragging(true);
    if (compassRef.current) {
      const angle = getAngleFromCenter(e, compassRef.current.getBoundingClientRect());
      onChange(angle);
    }
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (isDragging && compassRef.current) {
      const angle = getAngleFromCenter(e, compassRef.current.getBoundingClientRect());
      onChange(angle);
    }
  };

  const handleMouseUp = () => {
    setIsDragging(false);
  };

  // Handle global mouse events
  useEffect(() => {
    const handleGlobalMouseUp = () => setIsDragging(false);
    window.addEventListener('mouseup', handleGlobalMouseUp);
    return () => window.removeEventListener('mouseup', handleGlobalMouseUp);
  }, []);

  // Direction label based on angle
  const getDirectionLabel = (angle: number) => {
    const dirs = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'];
    const idx = Math.round(angle / 45) % 8;
    return dirs[idx];
  };

  return (
    <div className="flex flex-col items-center">
      <div
        ref={compassRef}
        className="relative w-32 h-32 rounded-full border-2 border-white/30 cursor-pointer select-none"
        style={{ background: 'radial-gradient(circle, rgba(30,30,50,0.9) 0%, rgba(20,20,40,0.95) 100%)' }}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
      >
        {/* Cardinal directions */}
        {['N', 'E', 'S', 'W'].map((dir, i) => (
          <span
            key={dir}
            className="absolute text-xs font-bold text-white/70"
            style={{
              top: i === 0 ? '4px' : i === 2 ? 'auto' : '50%',
              bottom: i === 2 ? '4px' : 'auto',
              left: i === 1 ? 'auto' : i === 3 ? '4px' : '50%',
              right: i === 1 ? '4px' : 'auto',
              transform: 'translate(-50%, -50%)'
            }}
          >
            {dir}
          </span>
        ))}

        {/* Tick marks */}
        {[...Array(8)].map((_, i) => (
          <div
            key={i}
            className="absolute w-px h-2 bg-white/30"
            style={{
              top: '8px',
              left: '50%',
              transform: `translateX(-50%) rotate(${i * 45}deg)`,
              transformOrigin: 'center 56px'
            }}
          />
        ))}

        {/* Arrow indicator */}
        <div
          className="absolute left-1/2 top-1/2 w-0 h-0"
          style={{
            borderLeft: '8px solid transparent',
            borderRight: '8px solid transparent',
            borderBottom: '24px solid #FF6B35',
            transform: `translateX(-50%) translateY(-100%) rotate(${value}deg)`,
            transformOrigin: 'center 24px',
            transition: isDragging ? 'none' : 'transform 0.2s ease'
          }}
        />

        {/* Center dot */}
        <div className="absolute left-1/2 top-1/2 w-3 h-3 rounded-full bg-white/50 -translate-x-1/2 -translate-y-1/2" />
      </div>

      {/* Current direction display */}
      <div className="mt-2 text-center">
        <span className="text-lg font-bold text-orange-400">{getDirectionLabel(value)}</span>
        <span className="text-sm text-text-secondary ml-2">{value}°</span>
      </div>
    </div>
  );
};


let globalWindDirection = 90;
let globalWindSpeed = 15;
let selectedIgnitionLat: number | null = null;
let selectedIgnitionLng: number | null = null;

export const setGlobalWindSettings = (dir: number, speed: number) => {
  globalWindDirection = dir;
  globalWindSpeed = speed;
};

export const getGlobalWindSettings = () => ({
  direction: globalWindDirection,
  speed: globalWindSpeed
});

export const setSelectedIgnitionPoint = (lat: number, lng: number) => {
  selectedIgnitionLat = lat;
  selectedIgnitionLng = lng;
  // Emit event to update UI
  window.dispatchEvent(new Event('ignitionPointChanged'));
};

export const getSelectedIgnitionPoint = () => {
  if (selectedIgnitionLat !== null && selectedIgnitionLng !== null) {
    return { lat: selectedIgnitionLat, lng: selectedIgnitionLng };
  }
  return null;
};

const PostFirePanel: React.FC = () => {
  const [windDirection, setWindDirection] = useState(90);
  const [windSpeed, setWindSpeed] = useState(15);
  const [ignitionPoint, setIgnitionPoint] = useState<{ lat: number; lng: number } | null>(null);
  const [ignitionMode, setIgnitionMode] = useState(false);
  const [spreadResult, setSpreadResult] = useState<FireSpreadResponse | null>(null);
  const [spreadProgress, setSpreadProgress] = useState(0);
  const animationRef = useRef<number | null>(null);

  // Use the post-fire spread simulation hook with fallback
  const spreadHook = useSimulateFireSpread();
  const simulateAtLocation = spreadHook?.simulateAtLocation || (() => Promise.reject(new Error('Hook not ready')));
  const isLoading = spreadHook?.isLoading || false;

  // Update global wind settings
  useEffect(() => {
    setGlobalWindSettings(windDirection, windSpeed);
  }, [windDirection, windSpeed]);

  // Listen for ignition point changes
  useEffect(() => {
    const handleIgnitionPoint = () => {
      const point = getSelectedIgnitionPoint();
      if (point) {
        setIgnitionPoint(point);
      }
    };
    window.addEventListener('ignitionPointChanged', handleIgnitionPoint);
    return () => window.removeEventListener('ignitionPointChanged', handleIgnitionPoint);
  }, []);

  // Toggle ignition mode
  const handleIgniteClick = () => {
    setIgnitionMode(true);
    window.dispatchEvent(new CustomEvent('setIgnitionMode', { detail: true }));
  };

  // Listen for map clicks to set ignition point
  useEffect(() => {
    const handleMapClick = (e: CustomEvent) => {
      const { lat, lng } = e.detail;
      setSelectedIgnitionPoint(lat, lng);
      setIgnitionPoint({ lat, lng });
      setIgnitionMode(false);
      window.dispatchEvent(new CustomEvent('setIgnitionMode', { detail: false }));
    };
    window.addEventListener('mapClickForIgnition', handleMapClick as any);
    return () => window.removeEventListener('mapClickForIgnition', handleMapClick as any);
  }, []);

  const handleRunSimulation = async () => {
    if (!ignitionPoint) return;

    setSpreadResult(null);
    setSpreadProgress(0);

    try {
      const data = await simulateAtLocation(
        ignitionPoint.lat,
        ignitionPoint.lng,
        windDirection,
        windSpeed,
        5  // time steps
      );

      setSpreadResult(data);
      startAnimation();

      // Show fire on map
      window.dispatchEvent(new CustomEvent('showFireSpread', { detail: data }));

    } catch (err) {
      console.error('Fire spread simulation failed:', err);
      // Don't crash - just log the error
    }
  };

  const startAnimation = () => {
    let progress = 0;

    const animate = () => {
      progress += 2;
      setSpreadProgress(Math.min(progress, 100));

      if (progress < 100) {
        animationRef.current = requestAnimationFrame(animate);
      }
    };

    animationRef.current = requestAnimationFrame(animate);
  };

  const handleReset = () => {
    setIgnitionPoint(null);
    setSpreadResult(null);
    setSpreadProgress(0);
    setIgnitionMode(false);
    selectedIgnitionLat = null;
    selectedIgnitionLng = null;
    window.dispatchEvent(new CustomEvent('setIgnitionMode', { detail: false }));
    window.dispatchEvent(new Event('resetFire'));
  };

  return (
    <div className="h-full flex flex-col gap-4 p-4">
      {/* Header */}
      <GlassCard>
        <h3 className="text-sm font-semibold text-text-primary mb-2 text-center">
          Post-Fire Spread Simulator
        </h3>
        <p className="text-xs text-text-secondary text-center">
          Click on map to select ignition point, then run simulation
        </p>
      </GlassCard>

      {/* Ignition Point Display / Ignite Button */}
      {ignitionPoint ? (
        <GlassCard glowColor="#FF4400">
          <div className="text-xs text-text-secondary mb-1">Ignition Point</div>
          <div className="font-bold">
            {ignitionPoint.lat.toFixed(4)}°N, {ignitionPoint.lng.toFixed(4)}°E
          </div>
        </GlassCard>
      ) : ignitionMode ? (
        <GlassCard glowColor="#FF6600">
          <p className="text-xs text-orange-400 text-center animate-pulse">
            Click on the map to set ignition point
          </p>
        </GlassCard>
      ) : (
        <GlassButton
          onClick={handleIgniteClick}
          glowColor="#FF4400"
        >
          Ignite
        </GlassButton>
      )}

      {/* Run Simulation Button */}
      <GlassButton
        onClick={handleRunSimulation}
        disabled={!ignitionPoint || isLoading}
        glowColor="#FF4400"
      >
        {isLoading ? 'Calculating...' : '▶ Run Simulation'}
      </GlassButton>

      {/* Wind Direction - Compass */}
      <GlassCard>
        <h4 className="text-xs font-semibold text-text-primary mb-2 text-center">Wind Direction</h4>
        <Compass value={windDirection} onChange={setWindDirection} />
      </GlassCard>

      {/* Wind Speed */}
      <GlassCard>
        <h4 className="text-xs font-semibold text-text-primary mb-2">
          Wind Speed: {windSpeed} km/h
        </h4>
        <input
          type="range"
          min="0"
          max="50"
          value={windSpeed}
          onChange={(e) => setWindSpeed(parseInt(e.target.value))}
          className="w-full"
        />
      </GlassCard>

      {/* Results */}
      {spreadResult && spreadResult.spread_points && spreadResult.spread_points.length > 0 && (
        <GlassCard glowColor="#FF4400">
          <h4 className="text-sm font-semibold text-text-primary mb-2 text-center">
            Fire Spread: {spreadProgress}%
          </h4>

          <div className="grid grid-cols-2 gap-2 text-xs">
            <div>
              <span className="text-text-secondary">Radius</span>
              <p className="font-bold">{spreadResult.spread_radius_km} km</p>
            </div>
            <div>
              <span className="text-text-secondary">Probability</span>
              <p className="font-bold text-red-400">{spreadResult.spread_probability}%</p>
            </div>
            <div>
              <span className="text-text-secondary">Wind</span>
              <p className="font-bold">{spreadResult.wind_direction}° @ {spreadResult.wind_speed} km/h</p>
            </div>
            <div>
              <span className="text-text-secondary">Data</span>
              <p className="font-bold">{spreadResult.conditions.data_source}</p>
            </div>
          </div>
        </GlassCard>
      )}

      {/* Reset Button */}
      {(ignitionPoint || spreadResult) && (
        <GlassButton onClick={handleReset}>
          Reset
        </GlassButton>
      )}
    </div>
  );
};

export default PostFirePanel;
