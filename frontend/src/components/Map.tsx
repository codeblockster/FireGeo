import { MapContainer, TileLayer, CircleMarker, Popup, useMap, LayerGroup, Polygon } from 'react-leaflet'
import { useEffect, useState, useRef, useCallback } from 'react'
import { useStore } from '../store/useStore'
import { motion } from 'framer-motion'
import L from 'leaflet'
import 'leaflet/dist/leaflet.css'

// Fix Leaflet icon issue
// eslint-disable-next-line @typescript-eslint/no-explicit-any
delete (L.Icon.Default.prototype as any)._getIconUrl
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-icon-2x.png',
  iconUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-icon.png',
  shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-shadow.png',
})

// Component to update map center when store changes
function MapUpdater() {
  const { mapCenter } = useStore()
  const map = useMap()

  useEffect(() => {
    map.setView(mapCenter, map.getZoom())
  }, [mapCenter, map])

  return null
}

// Map click handler component
function MapClickHandler({ onMapClick }: { onMapClick: (lat: number, lng: number) => void }) {
  const map = useMap()
  const { mode } = useStore()
  const [isIgnitionMode, setIsIgnitionMode] = useState(false)

  // Listen for ignition mode from PostFireTab
  useEffect(() => {
    const handleIgnition = (e: CustomEvent) => {
      setIsIgnitionMode(e.detail);
    };
    window.addEventListener('setIgnitionMode', handleIgnition as any);
    return () => window.removeEventListener('setIgnitionMode', handleIgnition as any);
  }, []);

  useEffect(() => {
    const handleClick = (e: L.LeafletMouseEvent) => {
      // In risk mode, click triggers risk assessment
      // In postfire mode with ignition enabled, click triggers fire spread
      if (mode === 'risk' || (mode === 'postfire' && isIgnitionMode)) {
        onMapClick(e.latlng.lat, e.latlng.lng)
      }
    }

    map.on('click', handleClick)

    return () => {
      map.off('click', handleClick)
    }
  }, [map, mode, isIgnitionMode, onMapClick])

  return null
}

// Custom layer control for map styles - Redesigned collapsible
function MapStyleControl() {
  const map = useMap()
  const controlRef = useRef<L.Control | null>(null)
  const { mapStyle, setMapStyle } = useStore()
  const isCollapsedRef = useRef(false)

  useEffect(() => {
    const createControl = () => {
      const Control = L.Control.extend({
        onAdd: function () {
          const container = L.DomUtil.create('div', 'leaflet-bar leaflet-control')
          container.style.background = '#1a1a2e'
          container.style.border = '2px solid #9b59b6'
          container.style.borderRadius = '10px'
          container.style.boxShadow = '0 4px 20px rgba(155, 89, 182, 0.4)'
          container.style.padding = '0'
          container.style.overflow = 'hidden'

          // Header with toggle
          const header = L.DomUtil.create('div', '', container)
          header.style.background = 'linear-gradient(135deg, #2d2d44, #1a1a2e)'
          header.style.padding = '10px 12px'
          header.style.cursor = 'pointer'
          header.style.display = 'flex'
          header.style.justifyContent = 'space-between'
          header.style.alignItems = 'center'
          header.style.borderBottom = isCollapsedRef.current ? 'none' : '1px solid rgba(155, 89, 182, 0.3)'

          // Title
          const title = L.DomUtil.create('span', '', header)
          title.innerHTML = '🗺️ Map Style'
          title.style.color = '#d4a5e6'
          title.style.fontSize = '12px'
          title.style.fontWeight = 'bold'
          title.style.fontFamily = 'Inter, sans-serif'

          // Toggle icon
          const toggleIcon = L.DomUtil.create('span', '', header)
          toggleIcon.innerHTML = isCollapsedRef.current ? '&#9654;' : '&#9660;'
          toggleIcon.style.color = '#9b59b6'
          toggleIcon.style.fontSize = '10px'

          // Content area
          const content = L.DomUtil.create('div', '', container)
          content.id = 'map-style-content'
          content.style.padding = '8px'
          content.style.display = isCollapsedRef.current ? 'none' : 'block'

          const styleButtons = [
            { id: 'dark', label: 'Dark', color: '#0f0f23' },
            { id: 'satellite', label: 'Satellite', color: '#1a1a2e' },
            { id: 'light', label: 'Light', color: '#f0f0f5' },
          ]

          styleButtons.forEach(btn => {
            const btnEl = L.DomUtil.create('button', '', content)
            btnEl.innerHTML = btn.label
            btnEl.style.display = 'block'
            btnEl.style.width = '100%'
            btnEl.style.padding = '8px 10px'
            btnEl.style.marginBottom = '4px'
            btnEl.style.border = mapStyle === btn.id ? '2px solid #9b59b6' : '1px solid rgba(155, 89, 182, 0.3)'
            btnEl.style.borderRadius = '6px'
            btnEl.style.background = mapStyle === btn.id ? 'rgba(155, 89, 182, 0.4)' : 'rgba(255,255,255, 0.1)'
            btnEl.style.color = mapStyle === btn.id ? '#fff' : '#c8c8d0'
            btnEl.style.fontSize = '11px'
            btnEl.style.cursor = 'pointer'
            btnEl.style.textAlign = 'left'
            btnEl.style.fontWeight = mapStyle === btn.id ? 'bold' : 'normal'
            btnEl.style.fontFamily = 'Inter, sans-serif'
            btnEl.style.transition = 'all 0.2s'

            // Hover effect
            btnEl.onmouseover = () => {
              btnEl.style.background = 'rgba(155, 89, 182, 0.3)'
              btnEl.style.borderColor = '#9b59b6'
            }
            btnEl.onmouseout = () => {
              btnEl.style.background = mapStyle === btn.id ? 'rgba(155, 89, 182, 0.4)' : 'rgba(255,255,255, 0.1)'
              btnEl.style.borderColor = mapStyle === btn.id ? '#9b59b6' : 'rgba(155, 89, 182, 0.3)'
            }

            L.DomEvent.on(btnEl, 'click', (e: Event) => {
              L.DomEvent.stopPropagation(e)
              setMapStyle(btn.id as 'dark' | 'satellite' | 'light')
            })
          })

          // Header click handler for collapse toggle
          L.DomEvent.on(header, 'click', (e: Event) => {
            L.DomEvent.stopPropagation(e)
            isCollapsedRef.current = !isCollapsedRef.current
            content.style.display = isCollapsedRef.current ? 'none' : 'block'
            header.style.borderBottom = isCollapsedRef.current ? 'none' : '1px solid rgba(155, 89, 182, 0.3)'
            toggleIcon.innerHTML = isCollapsedRef.current ? '&#9654;' : '&#9660;'
          })

          // Prevent map click when clicking inside control
          L.DomEvent.on(container, 'click', (e: Event) => {
            L.DomEvent.stopPropagation(e)
          })

          return container
        },
        options: {
          position: 'topright'
        }
      })

      return new Control()
    }

    controlRef.current = createControl()
    map.addControl(controlRef.current)

    return () => {
      if (controlRef.current) {
        map.removeControl(controlRef.current)
      }
    }
  }, [map, mapStyle, setMapStyle])

  return null
}

// Fire Marker Component with pulsing animation
function FireMarker({ fire }: { fire: { id: string; lat: number; lng: number; intensity: number; confidence: number } }) {
  const getColor = () => {
    if (fire.intensity > 80) return '#ff4757'
    if (fire.intensity > 60) return '#ff6b35'
    return '#ffa502'
  }

  const color = getColor()

  return (
    <CircleMarker
      center={[fire.lat, fire.lng]}
      radius={14}
      pathOptions={{
        fillColor: color,
        fillOpacity: 0.85,
        color: color,
        weight: 2,
      }}
    >
      <Popup>
        <div className="text-center p-2" style={{ background: '#1a1a2e', borderRadius: '8px' }}>
          <h3 className="font-bold text-sm mb-1" style={{ color: '#ff6b35' }}>Fire Detected</h3>
          <p className="text-xs" style={{ color: '#9b59b6' }}>Intensity: {fire.intensity}%</p>
          <p className="text-xs" style={{ color: '#9b59b6' }}>Confidence: {fire.confidence}%</p>
        </div>
      </Popup>
    </CircleMarker>
  )
}

// Heatmap layer component
function HeatmapLayer({ fires }: { fires: { id: string; lat: number; lng: number; intensity: number }[] }) {
  return (
    <>
      {fires.map((fire, index) => (
        <CircleMarker
          key={`heat-${fire.id || index}`}
          center={[fire.lat, fire.lng]}
          radius={35 + (fire.intensity / 8)}
          pathOptions={{
            fillColor: '#ff4757',
            fillOpacity: 0.12,
            color: 'transparent',
          }}
        />
      ))}
    </>
  )
}

// Risk Location Pin - Now clickable
function RiskPin({ lat, lng, isSelected = false }: { lat: number; lng: number; isSelected?: boolean }) {
  const getColor = () => {
    if (isSelected) return '#00E676'
    return '#9b59b6'
  }

  const color = getColor()

  return (
    <CircleMarker
      center={[lat, lng]}
      radius={isSelected ? 14 : 10}
      pathOptions={{
        fillColor: color,
        fillOpacity: isSelected ? 0.95 : 0.9,
        color: isSelected ? '#69F0AE' : '#e1bee7',
        weight: isSelected ? 3 : 2,
      }}
    >
      <Popup>
        <div className="text-center p-2" style={{ background: '#1a1a2e', borderRadius: '8px' }}>
          <h3 className="font-bold text-sm" style={{ color: '#ff6b35' }}>
            {isSelected ? 'Selected Location' : 'Risk Assessment Point'}
          </h3>
          <p className="text-xs mt-1" style={{ color: '#9b59b6' }}>
            Lat: {lat.toFixed(4)}, Lng: {lng.toFixed(4)}
          </p>
          <p className="text-xs mt-1" style={{ color: '#9b59b6' }}>
            Click to assess risk
          </p>
        </div>
      </Popup>
    </CircleMarker>
  )
}

// Border layer component - Uses ESRI World Boundary tiles
export function BorderLayer({ showBorders }: { showBorders: boolean }) {
  if (!showBorders) return null

  return (
    <LayerGroup>
      {/* ESRI World Boundaries - Countries */}
      <TileLayer
        url="https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}"
        attribution="&copy; Esri"
        opacity={0.7}
        maxZoom={19}
      />
    </LayerGroup>
  )
}

// Click indicator animation
function ClickIndicator({ position }: { position: [number, number] | null }) {
  const [show, setShow] = useState(false)

  useEffect(() => {
    if (position) {
      setShow(true)
      const timer = setTimeout(() => setShow(false), 1500)
      return () => clearTimeout(timer)
    }
  }, [position])

  if (!show || !position) return null

  return (
    <CircleMarker
      center={position}
      radius={25}
      pathOptions={{
        fillColor: '#00E676',
        fillOpacity: 0.3,
        color: '#00E676',
        weight: 2,
        dashArray: '5, 5',
      }}
    />
  )
}

export function MapView() {
  const { fireLocations, selectedLocation, mode, mapCenter, mapStyle, setSelectedLocation, setMapCenter, riskAssessment } = useStore()
  const [clickPosition, setClickPosition] = useState<[number, number] | null>(null)
  const [spreadPoints, setSpreadPoints] = useState<{ lat: number; lng: number; probability: number; time_step: number }[]>([])
  const [ignitionPoint, setIgnitionPoint] = useState<{ lat: number; lng: number } | null>(null)
  const [currentStep, setCurrentStep] = useState<number>(0)
  const [maxStep, setMaxStep] = useState<number>(0)

  // Listen for fire spread events from PostFireTab
  useEffect(() => {
    const handleShowFireSpread = (e: CustomEvent) => {
      const data = e.detail;
      console.log('Fire spread data received:', data);

      if (data.spread_points && Array.isArray(data.spread_points)) {
        const pts = data.spread_points.map((p: { latitude: number; longitude: number; probability: number; time_step: number }) => ({
          lat: p.latitude,
          lng: p.longitude,
          probability: p.probability,
          time_step: p.time_step || 1
        }));
        setSpreadPoints(pts);
        const mxStep = pts.length > 0 ? Math.max(...pts.map((p: { time_step: number }) => p.time_step)) : 1;
        setMaxStep(mxStep);
        setCurrentStep(1); // start animation from step 1
      }

      if (data.ignition_point) {
        setIgnitionPoint({
          lat: data.ignition_point.latitude,
          lng: data.ignition_point.longitude
        });
      }
    };

    window.addEventListener('showFireSpread', handleShowFireSpread as any);
    return () => window.removeEventListener('showFireSpread', handleShowFireSpread as any);
  }, []);

  // Listen for reset events
  useEffect(() => {
    const handleReset = () => {
      setSpreadPoints([]);
      setIgnitionPoint(null);
      setCurrentStep(0);
      setMaxStep(0);
    };
    window.addEventListener('resetFire', handleReset);
    return () => window.removeEventListener('resetFire', handleReset);
  }, []);

  // Animate fire spread step by step (reveal one time-step every 600ms)
  useEffect(() => {
    if (currentStep === 0 || currentStep >= maxStep) return;
    const timer = setTimeout(() => {
      setCurrentStep(prev => Math.min(prev + 1, maxStep));
    }, 600);
    return () => clearTimeout(timer);
  }, [currentStep, maxStep]);

  // Listen for immediate ignition point placement (shows marker right away)
  useEffect(() => {
    const handleIgnitionSet = (e: CustomEvent) => {
      const { lat, lng } = e.detail;
      setIgnitionPoint({ lat, lng });
      setSpreadPoints([]); // clear any previous spread when repositioning ignition
      setCurrentStep(0);
      setMaxStep(0);
    };
    window.addEventListener('setIgnitionOnMap', handleIgnitionSet as any);
    return () => window.removeEventListener('setIgnitionOnMap', handleIgnitionSet as any);
  }, []);

  // Map tile URLs
  const getTileUrl = (style: string) => {
    switch (style) {
      case 'satellite':
        return 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}'
      case 'light':
        return 'https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png'
      default:
        return 'https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png'
    }
  }

  const getAttribution = (style: string) => {
    switch (style) {
      case 'satellite':
        return '&copy; Esri'
      case 'light':
        return '&copy; CARTO'
      default:
        return '&copy; CARTO'
    }
  }

  const getBackgroundColor = () => {
    switch (mapStyle) {
      case 'satellite': return '#0d1117'
      case 'light': return '#f8f9fa'
      default: return '#0f0f23'
    }
  }

  // Handle map click for risk assessment or fire ignition
  const handleMapClick = useCallback((lat: number, lng: number) => {
    if (mode === 'risk') {
      // Update clicked position for animation
      setClickPosition([lat, lng])

      // Update selected location
      setSelectedLocation({
        id: `click-${lat.toFixed(4)}-${lng.toFixed(4)}`,
        name: `Location (${lat.toFixed(2)}, ${lng.toFixed(2)})`,
        lat,
        lng,
      })

      // Center map on clicked position
      setMapCenter([lat, lng])
    } else if (mode === 'postfire') {
      // In postfire mode, set the ignition point directly
      setClickPosition([lat, lng])
      setMapCenter([lat, lng])

      // Emit event to ControlPanel/PostFireTab and also call window function
      window.dispatchEvent(new CustomEvent('mapClickForIgnition', { detail: { lat, lng } }));

      // Also call window function directly for ControlPanel
      const w = window as any
      if (w.igniteSetPoint) {
        w.igniteSetPoint(lat, lng)
      }
    }
  }, [mode, setSelectedLocation, setMapCenter])

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.5, delay: 0.1, ease: 'easeOut' }}
      className="h-full w-full overflow-hidden squircle relative"
      style={{
        boxShadow: 'inset 0 0 60px rgba(0, 0, 0, 0.6)',
      }}
    >
      {/* Mode indicator */}
      <div className="absolute top-4 left-4 z-[1000]">
        <motion.div
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          className={`px-4 py-2 rounded-full text-sm font-semibold ${mode === 'fire'
            ? 'bg-orange-500/20 text-orange-400 border border-orange-500/30'
            : mode === 'postfire'
              ? 'bg-red-500/20 text-red-400 border border-red-500/30'
              : 'bg-purple-500/20 text-purple-400 border border-purple-500/30'
            }`}
        >
          {mode === 'fire' ? 'Fire Detection Mode' : mode === 'postfire' ? 'Post Fire Spread' : 'Risk Assessment Mode'}
        </motion.div>

        {mode === 'risk' && (
          <motion.p
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.3 }}
            className="text-xs mt-2 bg-black/50 px-3 py-1 rounded-full" style={{ color: '#9b59b6' }}
          >
            Select a location to assess risk
          </motion.p>
        )}

        {mode === 'postfire' && (
          <motion.p
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.3 }}
            className="text-xs mt-2 bg-black/50 px-3 py-1 rounded-full" style={{ color: '#FF6600' }}
          >
            Use the panel → SET IGNITION → Click map to place fire start
          </motion.p>
        )}
      </div>

      <MapContainer
        center={mapCenter}
        zoom={8}
        style={{
          height: '100%',
          width: '100%',
          background: getBackgroundColor(),
        }}
        zoomControl={true}
      >
        {/* Base tile layer */}
        <TileLayer
          attribution={getAttribution(mapStyle)}
          url={getTileUrl(mapStyle)}
        />

        <MapUpdater />
        <MapStyleControl />
        <MapClickHandler onMapClick={handleMapClick} />

        {/* Border layer - ESRI World Boundaries */}
        <BorderLayer showBorders={true} />

        {/* Click indicator animation */}
        <ClickIndicator position={clickPosition} />

        {/* Render based on mode */}
        {mode === 'fire' ? (
          <>
            {/* Heatmap layer */}
            <HeatmapLayer fires={fireLocations} />

            {/* Fire markers */}
            {fireLocations.map((fire) => (
              <FireMarker key={fire.id} fire={fire} />
            ))}
          </>
        ) : mode === 'postfire' ? (
          <>
            {/* Ignition point marker */}
            {ignitionPoint && (
              <CircleMarker
                center={[ignitionPoint.lat, ignitionPoint.lng]}
                radius={16}
                pathOptions={{
                  fillColor: '#FF4400',
                  fillOpacity: 0.9,
                  color: '#FF6600',
                  weight: 3,
                }}
              >
                <Popup>
                  <div className="text-center p-2" style={{ background: '#1a1a2e', borderRadius: '8px' }}>
                    <h3 className="font-bold text-sm" style={{ color: '#FF4400' }}>Ignition Point</h3>
                    <p className="text-xs mt-1" style={{ color: '#9b59b6' }}>
                      Lat: {ignitionPoint.lat.toFixed(4)}, Lng: {ignitionPoint.lng.toFixed(4)}
                    </p>
                  </div>
                </Popup>
              </CircleMarker>
            )}

            {/* ── Fire Spread Grid Cells: one rectangle per cell, colored by risk tier ── */}
            {(() => {
              const visible = spreadPoints.filter(p => p.time_step <= currentStep);
              if (visible.length === 0) return null;

              // Expand cells by 10% beyond exact grid size so adjacent cells overlap slightly,
              // sealing sub-pixel rendering gaps without any visible seams.
              const half = 0.0055;

              const getRiskStyle = (prob: number) => {
                if (prob >= 70) return { fillColor: '#CC0000', fillOpacity: 0.75, weight: 0, label: '🔴 Critical Zone', desc: 'Probability ≥ 70%' };
                if (prob >= 40) return { fillColor: '#FF5500', fillOpacity: 0.60, weight: 0, label: '🟠 High Risk Zone', desc: 'Probability ≥ 40%' };
                return { fillColor: '#FFAA00', fillOpacity: 0.40, weight: 0, label: '🟡 Moderate Zone', desc: 'Outer spread boundary' };
              };

              return (
                <>
                  {visible.map((p, i) => {
                    const style = getRiskStyle(p.probability);
                    return (
                      <Polygon
                        key={`cell-${i}`}
                        positions={[
                          [p.lat - half, p.lng - half],
                          [p.lat - half, p.lng + half],
                          [p.lat + half, p.lng + half],
                          [p.lat + half, p.lng - half],
                        ]}
                        pathOptions={{
                          fillColor: style.fillColor,
                          fillOpacity: style.fillOpacity,
                          weight: style.weight,
                          stroke: false,
                        }}
                      >
                        <Popup>
                          <div style={{ background: '#1a1a2e', borderRadius: 8, padding: 8, textAlign: 'center' }}>
                            <b style={{ color: style.fillColor }}>{style.label}</b>
                            <p style={{ color: '#ccc', fontSize: 11, margin: '4px 0 2px' }}>{style.desc}</p>
                            <p style={{ color: '#aaa', fontSize: 10 }}>Prob: {p.probability}% · Step {p.time_step}</p>
                            <p style={{ color: '#aaa', fontSize: 10 }}>{p.lat.toFixed(4)}, {p.lng.toFixed(4)}</p>
                          </div>
                        </Popup>
                      </Polygon>
                    );
                  })}
                </>
              );
            })()}
          </>
        ) : (
          selectedLocation && (
            <RiskPin
              lat={selectedLocation.lat}
              lng={selectedLocation.lng}
              isSelected={!!riskAssessment}
            />
          )
        )}
      </MapContainer>

      {/* Custom metallic border overlay */}
      <div
        className="absolute inset-0 pointer-events-none"
        style={{
          border: '3px solid transparent',
          borderRadius: '24px',
          background: 'linear-gradient(#0f0f23, #0f0f23) padding-box, linear-gradient(135deg, rgba(155, 89, 182, 0.5), rgba(255, 107, 53, 0.3)) border-box',
          boxShadow: 'inset 0 0 30px rgba(0, 0, 0, 0.5), 0 0 20px rgba(155, 89, 182, 0.2)',
        }}
      />
    </motion.div>
  )
}
