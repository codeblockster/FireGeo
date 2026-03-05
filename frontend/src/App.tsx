import { useEffect } from 'react'
import 'leaflet/dist/leaflet.css'
import { Navbar } from './components/Navbar'
import { MapView } from './components/Map'
import { ControlPanel } from './components/ControlPanel'
import { useStore } from './store/useStore'

function App() {
  const { selectedLocation } = useStore()


  // Set map center based on selected location
  useEffect(() => {
    if (selectedLocation) {
      useStore.getState().setMapCenter([selectedLocation.lat, selectedLocation.lng])
    }
  }, [selectedLocation])

  // 2-Panel Layout with Toggle in Right Panel
  return (
    <div className="h-screen w-screen bg-bg-dark flex flex-col overflow-hidden">
      <Navbar />
      <main className="flex-1 flex gap-2 p-2 overflow-hidden">
        {/* Center - Map (70%) */}
        <div className="flex-[7] min-w-0">
          <MapView />
        </div>

        {/* Right Panel - Toggle between Fire/Risk/PostFire (30%) */}
        <div className="flex-[3] min-w-[320px] max-w-[400px]">
          <div className="h-full glass squircle overflow-y-auto">
            <ControlPanel />
          </div>
        </div>
      </main>
    </div>
  )
}

export default App
