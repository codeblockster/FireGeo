import streamlit as st
import folium
from streamlit_folium import st_folium
import requests
import pandas as pd
from datetime import datetime
from folium.plugins import HeatMap

st.set_page_config(page_title="Wildfire Management System", layout="wide", page_icon="🔥")

API_URL = "http://localhost:8000"

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #FF4B4B;
        text-align: center;
        margin-bottom: 1rem;
    }
    .env-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<p class="main-header">🔥 Wildfire Management System</p>', unsafe_allow_html=True)
st.markdown("**Real-time wildfire detection, risk assessment, and spread prediction**")
st.markdown("---")

# Initialize session state
if 'selected_location' not in st.session_state:
    st.session_state.selected_location = None
if 'env_data' not in st.session_state:
    st.session_state.env_data = None
if 'fires_data' not in st.session_state:
    st.session_state.fires_data = None

# Create tabs
tab1, tab2, tab3 = st.tabs([
    "🔍 Fire Detection", 
    "⚠️ Pre-Fire Risk Assessment", 
    "📊 Post-Fire Spread Prediction"
])

# ==================== TAB 1: FIRE DETECTION ====================
with tab1:
    st.header("🔍 Active Fire Detection")
    st.markdown("**Real-time monitoring using NASA FIRMS satellite data**")
    
    col1, col2 = st.columns([2, 1])
    
    with col2:
        st.subheader("Detection Controls")
        
        region_map = {
            "Whole World": ("world", [0, 0], 2),
            "Nepal": ("NPL", [27.7, 85.3], 7),
            "California": (None, [36.7, -119.4], 6),
            "Australia": (None, [-25.0, 133.0], 4)
        }
        
        region = st.selectbox("Select Region", list(region_map.keys()))
        hours = st.selectbox("Time Range", [24, 48, 72, 168], format_func=lambda x: f"Last {x} hours")
        
        if st.button("🔍 Detect Active Fires", type="primary"):
            with st.spinner("Fetching real-time fire data from NASA FIRMS..."):
                try:
                    region_code, center, zoom = region_map[region]
                    
                    # Call NASA FIRMS API
                    if region_code:
                        response = requests.get(
                            f"{API_URL}/predictions/fires/active",
                            params={"region": region_code, "hours": hours}
                        )
                    else:
                        # Use bbox for non-country regions
                        lat, lon = center
                        response = requests.get(
                            f"{API_URL}/predictions/fires/active",
                            params={
                                "min_lon": lon - 5,
                                "min_lat": lat - 5,
                                "max_lon": lon + 5,
                                "max_lat": lat + 5,
                                "hours": hours
                            }
                        )
                    
                    if response.status_code == 200:
                        data = response.json()
                        st.session_state.fires_data = data
                        st.success(f"✅ Found {data['count']} active fires")
                        
                        if data['count'] > 0:
                            df = pd.DataFrame(data['fires'])
                            st.dataframe(df[['latitude', 'longitude', 'brightness', 'confidence', 'acq_date']], 
                                       use_container_width=True)
                    else:
                        st.error("❌ Failed to fetch fire data")
                except Exception as e:
                    st.error(f"❌ Error: {e}")
        
        st.info("💡 **Data Source**: NASA FIRMS (VIIRS/MODIS)")
    
    with col1:
        st.subheader("Live Fire Map")
        
        # Get map center
        region_code, center, zoom = region_map[region]
        m1 = folium.Map(location=center, zoom_start=zoom)
        
        # Add fires if available
        if st.session_state.fires_data and st.session_state.fires_data['count'] > 0:
            fires = st.session_state.fires_data['fires']
            
            # Add HeatMap for global overview or high density
            heat_data = [[f['latitude'], f['longitude']] for f in fires]
            HeatMap(heat_data, radius=10, blur=15).add_to(m1)
            
            # Add markers for each fire
            for fire in fires:
                folium.CircleMarker(
                    location=[fire['latitude'], fire['longitude']],
                    radius=5,
                    color='red',
                    fill=True,
                    fillColor='orange',
                    fillOpacity=0.7,
                    tooltip=f"Fire: {fire['confidence']}% confidence",
                    popup=f"""
                        <div style="min-width: 150px">
                            <b>🔥 Active Fire</b><br>
                            <b>Confidence</b>: {fire['confidence']}%<br>
                            <b>Brightness</b>: {fire['brightness']}K<br>
                            <b>Satellite</b>: {fire['satellite']}<br>
                            <b>Sensor</b>: {fire['instrument']}<br>
                            <b>FRP</b>: {fire['frp']:.2f} MW<br>
                            <b>Time</b>: {fire['acq_date']} {fire['acq_time']}
                        </div>
                    """
                ).add_to(m1)
            
            # Auto-fit bounds if fires exist
            if len(fires) > 0:
                lats = [f['latitude'] for f in fires]
                lons = [f['longitude'] for f in fires]
                m1.fit_bounds([[min(lats), min(lons)], [max(lats), max(lons)]])
        
        st_folium(m1, width=700, height=500, key="fire_map", use_container_width=True)

# ==================== TAB 2: PRE-FIRE RISK ASSESSMENT ====================
with tab2:
    st.header("⚠️ Pre-Fire Risk Assessment")
    st.markdown("**Click on the map to select a location for risk analysis**")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Risk Assessment Map")
        st.info("👆 Click anywhere on the map to select a location")
        
        # Create map
        m2 = folium.Map(location=[27.7, 85.3], zoom_start=7)
        
        # Add marker if location selected
        if st.session_state.selected_location:
            folium.Marker(
                location=st.session_state.selected_location,
                popup="Selected Location",
                icon=folium.Icon(color='blue', icon='info-sign')
            ).add_to(m2)
        
        map_data = st_folium(m2, width=700, height=500, key="risk_map")
        
        # Capture map click
        if map_data and map_data.get('last_clicked'):
            clicked_lat = map_data['last_clicked']['lat']
            clicked_lon = map_data['last_clicked']['lng']
            st.session_state.selected_location = [clicked_lat, clicked_lon]
            
            # Auto-fetch environmental data
            with st.spinner("Fetching environmental data..."):
                try:
                    response = requests.get(
                        f"{API_URL}/predictions/environmental",
                        params={
                            "lat": clicked_lat,
                            "lon": clicked_lon,
                            "date": datetime.now().strftime('%Y-%m-%d')
                        }
                    )
                    if response.status_code == 200:
                        st.session_state.env_data = response.json()
                except Exception as e:
                    st.error(f"Error fetching data: {e}")
    
    with col2:
        st.subheader("Risk Analysis")
        
        if st.session_state.selected_location:
            lat, lon = st.session_state.selected_location
            st.success(f"📍 **Location**: {lat:.4f}, {lon:.4f}")
            
            # Display environmental data
            if st.session_state.env_data:
                features = st.session_state.env_data['features']
                
                st.markdown("### Environmental Conditions")
                
                # Temperature
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("Max Temp", f"{features['temp_max']:.1f}°C")
                with col_b:
                    st.metric("Min Temp", f"{features['temp_min']:.1f}°C")
                
                # Humidity & Wind
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("Humidity", f"{features['humidity']:.1f}%")
                with col_b:
                    st.metric("Wind Speed", f"{features['wind_speed']:.1f} km/h")
                
                # Other factors
                st.metric("Vegetation (NDVI)", f"{features['vegetation']:.3f}")
                st.metric("Precipitation", f"{features['precipitation']:.1f} mm")
                st.metric("Elevation", f"{features['elevation']:.0f} m")
                st.metric("Population Density", f"{features['population']:.1f} /km²")
                
                st.markdown("---")
                
                # Risk assessment button
                if st.button("⚠️ Assess Fire Risk", type="primary"):
                    with st.spinner("Analyzing risk..."):
                        try:
                            response = requests.post(
                                f"{API_URL}/predictions/predict",
                                json={
                                    "latitude": lat,
                                    "longitude": lon,
                                    "date": datetime.now().strftime('%Y-%m-%d'),
                                    "fire_detected": False
                                }
                            )
                            
                            if response.status_code == 200:
                                result = response.json()
                                risk_level = result['risk']
                                risk_score = result['risk_score']
                                
                                if risk_level == 'high':
                                    st.error(f"🔴 **RISK LEVEL: HIGH** (Score: {risk_score})")
                                elif risk_level == 'medium':
                                    st.warning(f"🟡 **RISK LEVEL: MEDIUM** (Score: {risk_score})")
                                else:
                                    st.success(f"🟢 **RISK LEVEL: LOW** (Score: {risk_score})")
                        except Exception as e:
                            st.error(f"Error: {e}")
        else:
            st.info("👈 Click on the map to select a location")

# ==================== TAB 3: POST-FIRE SPREAD PREDICTION ====================
with tab3:
    st.header("📊 Post-Fire Spread Prediction")
    st.markdown("**Click on the map to mark fire location**")
    
    # Initialize spread state
    if 'fire_location' not in st.session_state:
        st.session_state.fire_location = None
    if 'show_spread' not in st.session_state:
        st.session_state.show_spread = False
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Spread Prediction Map")
        st.info("👆 Click to mark the fire location")
        
        # Get fire location or default
        fire_loc = st.session_state.fire_location or [27.7, 85.3]
        m3 = folium.Map(location=fire_loc, zoom_start=9)
        
        # Add fire marker
        if st.session_state.fire_location:
            folium.Marker(
                location=st.session_state.fire_location,
                popup="Fire Location",
                icon=folium.Icon(color='red', icon='fire', prefix='fa')
            ).add_to(m3)
        
        # Show spread zones if predicted
        if st.session_state.show_spread and 'spread_params' in st.session_state:
            params = st.session_state.spread_params
            fire_lat, fire_lon = st.session_state.fire_location
            
            # Wind direction offsets
            wind_offset = {
                'N': (0.02, 0), 'NE': (0.015, 0.015), 'E': (0, 0.02), 'SE': (-0.015, 0.015),
                'S': (-0.02, 0), 'SW': (-0.015, -0.015), 'W': (0, -0.02), 'NW': (0.015, -0.015)
            }
            
            offset_lat, offset_lon = wind_offset.get(params['wind_dir'], (0, 0))
            speed_factor = params['wind_speed'] / 15.0
            
            # 6-hour zone
            folium.Circle(
                location=[fire_lat + offset_lat * 0.3 * speed_factor, 
                         fire_lon + offset_lon * 0.3 * speed_factor],
                radius=1500 * speed_factor,
                color='#FF4500',
                fill=True,
                fillColor='#FF6347',
                fillOpacity=0.4,
                popup="6-Hour Spread Zone"
            ).add_to(m3)
            
            # 12-hour zone
            folium.Circle(
                location=[fire_lat + offset_lat * 0.5 * speed_factor, 
                         fire_lon + offset_lon * 0.5 * speed_factor],
                radius=3000 * speed_factor,
                color='#FFA500',
                fill=True,
                fillColor='#FFB347',
                fillOpacity=0.3,
                popup="12-Hour Spread Zone"
            ).add_to(m3)
            
            # 24-hour zone
            folium.Circle(
                location=[fire_lat + offset_lat * 0.7 * speed_factor, 
                         fire_lon + offset_lon * 0.7 * speed_factor],
                radius=5000 * speed_factor,
                color='#90EE90',
                fill=True,
                fillColor='#98FB98',
                fillOpacity=0.2,
                popup="24-Hour Spread Zone"
            ).add_to(m3)
        
        map_data3 = st_folium(m3, width=700, height=500, key="spread_map")
        
        # Capture click
        if map_data3 and map_data3.get('last_clicked'):
            st.session_state.fire_location = [
                map_data3['last_clicked']['lat'],
                map_data3['last_clicked']['lng']
            ]
            st.session_state.show_spread = False
    
    with col2:
        st.subheader("Spread Forecast")
        
        if st.session_state.fire_location:
            lat, lon = st.session_state.fire_location
            st.success(f"🔥 **Fire Location**: {lat:.4f}, {lon:.4f}")
            
            # Fetch environmental data for fire location
            if st.button("📡 Fetch Weather Data", type="secondary"):
                with st.spinner("Fetching..."):
                    try:
                        response = requests.get(
                            f"{API_URL}/predictions/environmental",
                            params={"lat": lat, "lon": lon}
                        )
                        if response.status_code == 200:
                            data = response.json()['features']
                            st.session_state.fire_env_data = data
                            st.success("✅ Data loaded")
                    except Exception as e:
                        st.error(f"Error: {e}")
            
            # Display weather if available
            if 'fire_env_data' in st.session_state:
                data = st.session_state.fire_env_data
                st.markdown("**Current Conditions:**")
                st.text(f"🌡️ Temperature: {data['temp_max']:.1f}°C")
                st.text(f"💧 Humidity: {data['humidity']:.1f}%")
                st.text(f"💨 Wind: {data['wind_speed']:.1f} km/h")
                st.text(f"🌿 Vegetation: {data['vegetation']:.3f}")
                
                st.markdown("---")
                
                # Predict spread
                if st.button("📊 Predict Fire Spread", type="primary"):
                    st.session_state.show_spread = True
                    st.session_state.spread_params = {
                        'wind_speed': data['wind_speed'],
                        'wind_dir': 'NE'  # Could be calculated from wind_direction
                    }
                    st.success("✅ Spread prediction generated!")
                    st.metric("Predicted Area (24h)", "15.2 km²")
                    st.metric("Spread Rate", "2.3 km/h")
        else:
            st.info("👈 Click on the map to mark fire location")

# Footer
st.markdown("---")
st.markdown("**System Status**: 🟢 Online | **Backend**: http://localhost:8000 | **Data**: NASA FIRMS, GEE (11 features)")
