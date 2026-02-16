"""
Wildfire Management System - Premium Material 3 Expressive UI
Professional Edition with Real-Time API Integration
Designed for Pixel-class experience with scientific precision
"""

import sys
from pathlib import Path

# Add project root to sys.path for direct imports
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import streamlit as st
import folium
from streamlit_folium import st_folium
import pandas as pd
from datetime import datetime
from folium.plugins import HeatMap
import plotly.graph_objects as go
import plotly.express as px

# Direct imports from backend modules
from backend.src.data_collection.gee_extractor import get_gee_extractor
from backend.firedetect import FireDetector
from backend.prefire import PreFireAnalyzer

st.set_page_config(
    page_title="Wildfire Intelligence - Professional Edition",
    layout="wide",
    page_icon=" ",
    initial_sidebar_state="expanded"
)

# BACKEND INITIALIZATION WITH ERROR HANDLING

@st.cache_resource
def get_backend_components():
    """Initialize all backend components with proper error handling."""
    try:
        components = {
            "gee": get_gee_extractor(),
            "fire_detector": FireDetector(),
            "pre_fire": PreFireAnalyzer(),
            
           
        }
        return components
    except Exception as e:
        st.error(f" Backend initialization failed: {str(e)}")
        st.info("Please ensure all backend modules are properly configured and dependencies are installed.")
        return None

backend = get_backend_components()

# Exit if backend initialization failed
if backend is None:
    st.stop()

# MATERIAL 3 EXPRESSIVE DESIGN SYSTEM

# Dynamic color palette (Material 3 Expressive)
COLORS = {
    # Primary palette
    'primary': '#FF6B35',        # Vibrant coral-orange (fire theme)
    'on_primary': '#FFFFFF',
    'primary_container': '#FFE5DC',
    'on_primary_container': '#3D0D00',
    
    # Secondary palette
    'secondary': '#7B5800',      # Warm amber
    'on_secondary': '#FFFFFF',
    'secondary_container': '#FFE08A',
    'on_secondary_container': '#261900',
    
    # Tertiary palette
    'tertiary': '#006C4C',       # Forest green
    'on_tertiary': '#FFFFFF',
    'tertiary_container': '#7FFCD0',
    'on_tertiary_container': '#002116',
    
    # Surface colors
    'surface': '#FFFBFF',
    'surface_dim': '#E3DFE0',
    'surface_bright': '#FFFBFF',
    'surface_container': '#F4F0F1',
    'surface_container_low': '#FEF7F8',
    'surface_container_high': '#EEE9EA',
    'on_surface': '#1D1B1C',
    'on_surface_variant': '#504349',
    
    # Accent colors for data
    'critical': '#D32F2F',       # Critical alerts
    'high': '#F57C00',           # High risk
    'medium': '#FFA000',         # Medium risk
    'low': '#388E3C',            # Low risk
    'info': '#1976D2',           # Information
    
    # Chart colors (scientific palette)
    'chart_1': '#FF6B35',
    'chart_2': '#004E89',
    'chart_3': '#FFA000',
    'chart_4': '#388E3C',
    'chart_5': '#7B5800',
    
    # Semantic colors
    'success': '#2E7D32',
    'warning': '#F57C00',
    'error': '#C62828',
    'outline': '#837377',
    'outline_variant': '#D7C2C9',
}

def hex_to_rgba(hex_color: str, alpha: float = 1.0) -> str:
    """Convert hex color to rgba string."""
    hex_color = hex_color.lstrip('#')
    if len(hex_color) == 6:
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        return f"rgba({r}, {g}, {b}, {alpha})"
    return hex_color

# Material 3 Expressive CSS
st.markdown(f"""
<style>
    /* ===== GLOBAL RESET & FOUNDATIONS ===== */
    @import url('https://fonts.googleapis.com/css2?family=Google+Sans:wght@400;500;700&family=Google+Sans+Display:wght@400;500;700&display=swap');
    
    * {{
        font-family: 'Google Sans', system-ui, -apple-system, sans-serif !important;
    }}
    
    .stApp {{
        background: linear-gradient(135deg, {COLORS['surface']} 0%, {COLORS['surface_container_low']} 100%);
    }}

    /* HIDE STREAMLIT HEADER AND FOOTER */
    header {{visibility: hidden;}}
    .stApp > header {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    #MainMenu {{visibility: hidden;}}
    
    /* Remove default Streamlit padding */
    .block-container {{
        padding-top: 0rem !important;
        padding-bottom: 2rem !important;
        max-width: 1400px !important;
    }}
    
    /* ===== TYPOGRAPHY SCALE ===== */
    h1, .display-large {{
        font-family: 'Google Sans Display', sans-serif !important;
        font-size: 3.5rem !important;
        font-weight: 700 !important;
        line-height: 1.1 !important;
        letter-spacing: -0.02em !important;
        background: linear-gradient(135deg, {COLORS['primary']} 0%, {COLORS['secondary']} 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem !important;
    }}
    
    h2, .headline-medium {{
        font-family: 'Google Sans Display', sans-serif !important;
        font-size: 2rem !important;
        font-weight: 600 !important;
        line-height: 1.2 !important;
        color: {COLORS['on_surface']} !important;
        margin-bottom: 1rem !important;
    }}
    
    h3, .title-large {{
        font-size: 1.5rem !important;
        font-weight: 600 !important;
        color: {COLORS['on_surface']} !important;
        margin-bottom: 0.75rem !important;
    }}
    
    p, .body-large {{
        font-size: 1rem !important;
        line-height: 1.6 !important;
        color: {COLORS['on_surface_variant']} !important;
        margin-bottom: 1rem !important;
    }}
    
    .label-large {{
        font-size: 0.875rem !important;
        font-weight: 500 !important;
        letter-spacing: 0.01em !important;
        color: {COLORS['on_surface_variant']} !important;
    }}
    
    /* ===== EXPRESSIVE CARDS ===== */
    .material-card {{
        background: {COLORS['surface_container']} !important;
        border-radius: 28px !important;
        padding: 1.5rem !important;
        box-shadow: 
            0 1px 2px 0 rgba(0, 0, 0, 0.05),
            0 1px 3px 1px rgba(0, 0, 0, 0.08) !important;
        transition: all 0.3s cubic-bezier(0.4, 0.0, 0.2, 1) !important;
        border: 1px solid {COLORS['outline_variant']} !important;
        margin-bottom: 1.5rem !important;
    }}
    
    .material-card:hover {{
        box-shadow: 
            0 2px 4px 0 rgba(0, 0, 0, 0.08),
            0 4px 8px 2px rgba(0, 0, 0, 0.12) !important;
        transform: translateY(-2px) !important;
    }}
    
    .elevated-card {{
        background: {COLORS['surface']} !important;
        border-radius: 24px !important;
        padding: 2rem !important;
        box-shadow: 
            0 4px 6px -1px rgba(0, 0, 0, 0.1),
            0 2px 4px -2px rgba(0, 0, 0, 0.1) !important;
        border: none !important;
        margin-bottom: 2rem !important;
    }}
    
    /* ===== EXPRESSIVE METRICS ===== */
    .metric-card {{
        background: linear-gradient(135deg, {COLORS['primary_container']} 0%, {COLORS['surface_container_high']} 100%) !important;
        border-radius: 20px !important;
        padding: 1.25rem 1.5rem !important;
        border-left: 4px solid {COLORS['primary']} !important;
        box-shadow: 0 2px 8px rgba(255, 107, 53, 0.15) !important;
        transition: all 0.3s cubic-bezier(0.34, 1.56, 0.64, 1) !important;
    }}
    
    .metric-card:hover {{
        transform: scale(1.02) !important;
        box-shadow: 0 4px 12px rgba(255, 107, 53, 0.25) !important;
    }}
    
    .metric-value {{
        font-family: 'Google Sans Display', sans-serif !important;
        font-size: 2.5rem !important;
        font-weight: 700 !important;
        color: {COLORS['primary']} !important;
        line-height: 1 !important;
        margin-bottom: 0.25rem !important;
    }}
    
    .metric-label {{
        font-size: 0.875rem !important;
        font-weight: 500 !important;
        color: {COLORS['on_surface_variant']} !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
    }}
    
    .metric-unit {{
        font-size: 1rem !important;
        font-weight: 400 !important;
        color: {COLORS['on_surface_variant']} !important;
        margin-left: 0.25rem !important;
    }}
    
    /* ===== BUTTONS (Material 3 Expressive) ===== */
    .stButton > button {{
        background: linear-gradient(135deg, {COLORS['primary']} 0%, {COLORS['secondary']} 100%) !important;
        color: {COLORS['on_primary']} !important;
        border: none !important;
        border-radius: 100px !important;
        padding: 0.75rem 2rem !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        letter-spacing: 0.02em !important;
        box-shadow: 0 2px 8px rgba(255, 107, 53, 0.3) !important;
        transition: all 0.3s cubic-bezier(0.34, 1.56, 0.64, 1) !important;
        text-transform: none !important;
    }}
    
    .stButton > button:hover {{
        transform: scale(1.05) translateY(-2px) !important;
        box-shadow: 0 4px 16px rgba(255, 107, 53, 0.4) !important;
    }}
    
    .stButton > button:active {{
        transform: scale(0.98) !important;
    }}
    
    /* Secondary button */
    .stButton.secondary > button {{
        background: {COLORS['surface_container_high']} !important;
        color: {COLORS['primary']} !important;
        border: 2px solid {COLORS['outline_variant']} !important;
        box-shadow: none !important;
    }}
    
    /* ===== TABS (Pixel-style) ===== */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 1rem !important;
        background: {COLORS['surface_container']} !important;
        border-radius: 16px !important;
        padding: 0.5rem !important;
        margin-bottom: 2rem !important;
    }}
    
    .stTabs [data-baseweb="tab"] {{
        border-radius: 12px !important;
        padding: 0.75rem 1.5rem !important;
        font-weight: 500 !important;
        font-size: 0.95rem !important;
        color: {COLORS['on_surface_variant']} !important;
        background: transparent !important;
        border: none !important;
        transition: all 0.3s cubic-bezier(0.4, 0.0, 0.2, 1) !important;
    }}
    
    .stTabs [data-baseweb="tab"]:hover {{
        background: {COLORS['surface_container_high']} !important;
        color: {COLORS['on_surface']} !important;
    }}
    
    .stTabs [aria-selected="true"] {{
        background: {COLORS['primary']} !important;
        color: {COLORS['on_primary']} !important;
        font-weight: 600 !important;
        box-shadow: 0 2px 8px rgba(255, 107, 53, 0.3) !important;
    }}
    
    /* ===== STATUS CHIPS ===== */
    .status-chip {{
        display: inline-flex !important;
        align-items: center !important;
        gap: 0.5rem !important;
        padding: 0.5rem 1rem !important;
        border-radius: 16px !important;
        font-weight: 500 !important;
        font-size: 0.875rem !important;
        letter-spacing: 0.01em !important;
        transition: all 0.2s cubic-bezier(0.4, 0.0, 0.2, 1) !important;
    }}
    
    .chip-critical {{
        background: {COLORS['error']}15 !important;
        color: {COLORS['error']} !important;
        border: 1.5px solid {COLORS['error']}40 !important;
    }}
    
    .chip-high {{
        background: {COLORS['high']}15 !important;
        color: {COLORS['high']} !important;
        border: 1.5px solid {COLORS['high']}40 !important;
    }}
    
    .chip-medium {{
        background: {COLORS['medium']}15 !important;
        color: {COLORS['medium']} !important;
        border: 1.5px solid {COLORS['medium']}40 !important;
    }}
    
    .chip-low {{
        background: {COLORS['low']}15 !important;
        color: {COLORS['low']} !important;
        border: 1.5px solid {COLORS['low']}40 !important;
    }}
    
    /* ===== INFO BOXES ===== */
    .stAlert {{
        border-radius: 16px !important;
        border: none !important;
        padding: 1rem 1.5rem !important;
        font-size: 0.95rem !important;
    }}
    
    .stSuccess {{
        background: {COLORS['success']}10 !important;
        color: {COLORS['success']} !important;
    }}
    
    .stWarning {{
        background: {COLORS['warning']}10 !important;
        color: {COLORS['warning']} !important;
    }}
    
    .stError {{
        background: {COLORS['error']}10 !important;
        color: {COLORS['error']} !important;
    }}
    
    .stInfo {{
        background: {COLORS['info']}10 !important;
        color: {COLORS['info']} !important;
    }}
    
    /* ===== DATA TABLE ===== */
    .dataframe {{
        border: none !important;
        border-radius: 16px !important;
        overflow: hidden !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08) !important;
    }}
    
    .dataframe thead tr {{
        background: {COLORS['primary_container']} !important;
        color: {COLORS['on_primary_container']} !important;
        font-weight: 600 !important;
    }}
    
    .dataframe tbody tr:hover {{
        background: {COLORS['surface_container_high']} !important;
    }}
    
    /* ===== EXPANDER ===== */
    .streamlit-expanderHeader {{
        background: {COLORS['surface_container']} !important;
        border-radius: 12px !important;
        font-weight: 500 !important;
        padding: 1rem 1.5rem !important;
        border: 1px solid {COLORS['outline_variant']} !important;
    }}
    
    .streamlit-expanderHeader:hover {{
        background: {COLORS['surface_container_high']} !important;
    }}
    
    /* ===== SIDEBAR ===== */
    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, {COLORS['surface_container']} 0%, {COLORS['surface_container_low']} 100%) !important;
        border-right: 1px solid {COLORS['outline_variant']} !important;
    }}
    
    /* ===== PLOTLY CHARTS ===== */
    .js-plotly-plot {{
        border-radius: 16px !important;
        overflow: hidden !important;
    }}
    
    /* ===== LOADING SPINNER ===== */
    .stSpinner > div {{
        border-top-color: {COLORS['primary']} !important;
    }}
    
    /* ===== CUSTOM COMPONENTS ===== */
    .hero-section {{
        background: linear-gradient(135deg, {COLORS['primary_container']} 0%, {COLORS['secondary_container']} 100%);
        border-radius: 32px;
        padding: 3rem 2rem;
        margin-bottom: 2rem;
        text-align: center;
        border: 2px solid {COLORS['outline_variant']};
        box-shadow: 0 8px 24px rgba(255, 107, 53, 0.15);
    }}
    
    .data-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 1.5rem;
        margin: 2rem 0;
    }}
    
    .feature-pill {{
        display: inline-block;
        background: {COLORS['tertiary_container']};
        color: {COLORS['on_tertiary_container']};
        padding: 0.5rem 1rem;
        border-radius: 100px;
        font-size: 0.85rem;
        font-weight: 500;
        margin: 0.25rem;
    }}
    
    /* ===== ANIMATIONS ===== */
    @keyframes slide-up {{
        from {{
            opacity: 0;
            transform: translateY(20px);
        }}
        to {{
            opacity: 1;
            transform: translateY(0);
        }}
    }}
    
    .animate-in {{
        animation: slide-up 0.6s cubic-bezier(0.34, 1.56, 0.64, 1);
    }}
    
    /* ===== SCROLLBAR ===== */
    ::-webkit-scrollbar {{
        width: 8px;
        height: 8px;
    }}
    
    ::-webkit-scrollbar-track {{
        background: {COLORS['surface_container']};
    }}
    
    ::-webkit-scrollbar-thumb {{
        background: {COLORS['outline_variant']};
        border-radius: 100px;
    }}
    
    ::-webkit-scrollbar-thumb:hover {{
        background: {COLORS['outline']};
    }}
</style>
""", unsafe_allow_html=True)

# HERO SECTION

st.markdown("""
<div class="hero-section animate-in">
    <h1 class="display-large"> Wildfire Detection and Risk Assessment</h1>
    <p class="body-large" style="color: #504349; margin-top: 0.5rem;">Professional Edition - Real-Time API Integration</p>
</div>
""", unsafe_allow_html=True)

# Initialize session state
if 'selected_location' not in st.session_state:
    st.session_state.selected_location = None
if 'env_data' not in st.session_state:
    st.session_state.env_data = None
if 'fires_data' not in st.session_state:
    st.session_state.fires_data = None

# TABS

tab1, tab2 = st.tabs([
    " Live Fire Detection",
    " Risk Assessment", 
   
])

with tab1:
    st.markdown('<div class="elevated-card animate-in">', unsafe_allow_html=True)
    st.markdown('<h2 class="headline-medium"> Active Fire Detection</h2>', unsafe_allow_html=True)
    st.markdown('<p class="body-large">Real-time monitoring using NASA FIRMS satellite constellation (VIIRS & MODIS sensors)</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2.5, 1], gap="large")
    
    with col2:
        st.markdown('<div class="material-card">', unsafe_allow_html=True)
        st.markdown('<h3 class="title-large"> Detection Parameters</h3>', unsafe_allow_html=True)
        
        fire_detector = backend["fire_detector"]
        region_map = fire_detector.get_region_map()
        
        region = st.selectbox(
            "Geographic Region",
            list(region_map.keys()),
            help="Select area of interest for fire detection"
        )
        
        hours = st.selectbox(
            "Temporal Window",
            [24, 48, 72, 168],
            format_func=lambda x: f"Last {x} hours ({x//24} days)",
            help="Time range for active fire detection"
        )
        
        # Detect fires button
        if st.button(" Detect Active Fires", type="primary", width='stretch'):
            with st.spinner(f" Analyzing {region} for active fires via NASA FIRMS API..."):
                try:
                    result = fire_detector.detect_fires(region, hours=hours)
                    
                    if result and 'error' not in result:
                        st.session_state.fires_data = result
                        fire_count = result.get('count', 0)
                        
                        if fire_count == 0:
                            st.success(f" Analysis complete: No active fires detected in {region}")
                        else:
                            st.success(f" Successfully detected {fire_count} active fires in {region}")
                        st.rerun()
                    elif 'error' in result:
                        st.error(f" API Error: {result['error']}")
                        st.info("Please verify your NASA FIRMS API credentials and network connection.")
                    else:
                        st.warning(" No data returned from API. The region may have no active fires.")
                        
                except Exception as e:
                    st.error(f" Detection failed: {str(e)}")
                    st.info(" Troubleshooting: Ensure backend.firedetect module is properly configured with NASA FIRMS API credentials.")
        
        # Show info for Whole World
        if region == "Whole World":
            st.info(" Global fire detection may take longer due to large dataset. Results displayed as heatmap for optimal performance.")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Data source info
        st.markdown(f"""
        <div class="material-card" style="background: {COLORS['info']}10; border-left: 4px solid {COLORS['info']};">
            <div class="label-large" style="color: {COLORS['info']}; margin-bottom: 0.5rem;"> Data Source</div>
            <p style="font-size: 0.9rem; color: {COLORS['on_surface_variant']}; margin: 0;">
                NASA Fire Information for Resource Management System (FIRMS) provides near real-time active fire data from
                VIIRS (375m resolution) and MODIS (1km resolution) satellite sensors with <3 hour latency.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col1:
        st.markdown('<div class="elevated-card">', unsafe_allow_html=True)
        st.markdown('<h3 class="title-large"> Live Fire Map</h3>', unsafe_allow_html=True)
        
        # Create folium map
        region_code, center, zoom = region_map[region]
        m1 = folium.Map(
            location=center,
            zoom_start=zoom,
            tiles='CartoDB positron',
            attr='© OpenStreetMap contributors, © CartoDB'
        )
        
        # Add fires if available
        if st.session_state.fires_data and st.session_state.fires_data.get('count') != "N/A" and st.session_state.fires_data.get('count', 0) > 0:
            fires = st.session_state.fires_data['fires']
            
            # For Whole World or large datasets, use heatmap only (no individual markers)
            if region == "Whole World" or len(fires) > 1000:
                # HeatMap only for performance
                heat_data = [[f['latitude'], f['longitude'], f['confidence']/100] for f in fires]
                HeatMap(
                    heat_data,
                    radius=10,
                    blur=15,
                    max_zoom=13,
                    gradient={0.4: 'yellow', 0.6: 'orange', 0.8: 'red', 1.0: 'darkred'}
                ).add_to(m1)
            else:
                # HeatMap + individual markers for smaller datasets
                heat_data = [[f['latitude'], f['longitude'], f['confidence']/100] for f in fires]
                HeatMap(
                    heat_data,
                    radius=15,
                    blur=20,
                    max_zoom=13,
                    gradient={0.4: 'yellow', 0.6: 'orange', 0.8: 'red', 1.0: 'darkred'}
                ).add_to(m1)
                
                # Individual markers (only for smaller datasets)
                for fire in fires:
                    color = 'darkred' if fire['confidence'] > 80 else 'red' if fire['confidence'] > 50 else 'orange'
                    
                    folium.CircleMarker(
                        location=[fire['latitude'], fire['longitude']],
                        radius=6,
                        color=color,
                        fill=True,
                        fillColor=color,
                        fillOpacity=0.8,
                        weight=2,
                        tooltip=f" {fire['confidence']}% confidence",
                        popup=folium.Popup(f"""
                            <div style="min-width: 200px; font-family: 'Google Sans', sans-serif;">
                                <div style="background: linear-gradient(135deg, {COLORS['primary']} 0%, {COLORS['secondary']} 100%); 
                                            color: white; padding: 0.75rem; margin: -0.5rem -0.5rem 0.75rem -0.5rem; 
                                            border-radius: 8px 8px 0 0;">
                                    <strong style="font-size: 1.1rem;"> Active Fire Detection</strong>
                                </div>
                                <table style="width: 100%; font-size: 0.9rem;">
                                    <tr><td style="padding: 0.25rem 0;"><strong>Confidence:</strong></td><td>{fire['confidence']}%</td></tr>
                                    <tr><td style="padding: 0.25rem 0;"><strong>Brightness:</strong></td><td>{fire['brightness']} K</td></tr>
                                    <tr><td style="padding: 0.25rem 0;"><strong>FRP:</strong></td><td>{fire['frp']:.2f} MW</td></tr>
                                    <tr><td style="padding: 0.25rem 0;"><strong>Satellite:</strong></td><td>{fire['satellite']}</td></tr>
                                    <tr><td style="padding: 0.25rem 0;"><strong>Sensor:</strong></td><td>{fire['instrument']}</td></tr>
                                    <tr><td style="padding: 0.25rem 0;"><strong>Detected:</strong></td><td>{fire['acq_date']} {fire['acq_time']}</td></tr>
                                </table>
                            </div>
                        """, max_width=300)
                    ).add_to(m1)
            
            # Fit bounds
            if len(fires) > 0:
                lats = [f['latitude'] for f in fires]
                lons = [f['longitude'] for f in fires]
                m1.fit_bounds([[min(lats), min(lons)], [max(lats), max(lons)]])
        
        st_folium(m1, height=600, key="fire_map", width='stretch')
        
        # Fire Statistics and Timeline (if fires detected)
        if st.session_state.fires_data and st.session_state.fires_data.get('count', 0) > 0:
            fires = st.session_state.fires_data['fires']
            
            st.markdown("---")
            st.markdown("####  Fire Detection Statistics")
            
            # Get statistics using fire_detector
            stats = fire_detector.get_statistics(st.session_state.fires_data)
            
            # Statistics cards
            stat_cols = st.columns(4)
            with stat_cols[0]:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value" style="font-size: 2rem;">{stats['total_fires']}</div>
                    <div class="metric-label">Total Fires</div>
                </div>
                """, unsafe_allow_html=True)
            
            with stat_cols[1]:
                if 'confidence' in stats:
                    high_conf = stats['confidence'].get('high_confidence_count', 0)
                    st.markdown(f"""
                    <div class="metric-card" style="border-left-color: {COLORS['error']};">
                        <div class="metric-value" style="color: {COLORS['error']}; font-size: 2rem;">{high_conf}</div>
                        <div class="metric-label">High Confidence</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            with stat_cols[2]:
                if 'brightness' in stats:
                    avg_bright = stats['brightness'].get('mean', 0)
                    st.markdown(f"""
                    <div class="metric-card" style="border-left-color: {COLORS['warning']};">
                        <div class="metric-value" style="color: {COLORS['warning']}; font-size: 2rem;">{avg_bright:.0f}</div>
                        <div class="metric-label">Avg Brightness (K)</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            with stat_cols[3]:
                if 'frp' in stats:
                    total_frp = stats['frp'].get('total_MW', 0)
                    st.markdown(f"""
                    <div class="metric-card" style="border-left-color: {COLORS['chart_3']};">
                        <div class="metric-value" style="color: {COLORS['chart_3']}; font-size: 2rem;">{total_frp:.0f}</div>
                        <div class="metric-label">Total FRP (MW)</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Detection timeline graph
            if len(fires) > 1:
                st.markdown("####  Detection Timeline")
                
                # Process fire data for timeline
                df_fires = fire_detector.get_fires_dataframe(st.session_state.fires_data)
                if not df_fires.empty and 'acq_date' in df_fires.columns:
                    # Group by date
                    timeline_data = df_fires.groupby('acq_date').size().reset_index(name='count')
                    timeline_data = timeline_data.sort_values('acq_date')
                    
                    # Create timeline chart
                    fig_timeline = go.Figure()
                    fig_timeline.add_trace(go.Bar(
                        x=timeline_data['acq_date'],
                        y=timeline_data['count'],
                        marker=dict(
                            color=COLORS['primary'],
                            line=dict(color=COLORS['primary'], width=1)
                        ),
                        name='Fire Detections'
                    ))
                    
                    fig_timeline.update_layout(
                        height=250,
                        margin=dict(l=0, r=0, t=0, b=0),
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(family='Google Sans', size=12, color=COLORS['on_surface']),
                        xaxis=dict(
                            showgrid=False,
                            title=dict(text='Detection Date', font=dict(size=11, weight=500))
                        ),
                        yaxis=dict(
                            showgrid=True,
                            gridcolor=COLORS['outline_variant'],
                            gridwidth=1,
                            title=dict(text='Number of Fires', font=dict(size=11, weight=500))
                        ),
                        hovermode='x'
                    )
                    
                    st.plotly_chart(fig_timeline, width='stretch', config={'displayModeBar': False})
        
        st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    st.markdown('<div class="elevated-card animate-in">', unsafe_allow_html=True)
    st.markdown('<h2 class="headline-medium"> Pre-Fire Risk Assessment</h2>', unsafe_allow_html=True)
    st.markdown('<p class="body-large">Machine learning-powered fire risk prediction using 20+ environmental parameters from Weather API & Google Earth Engine</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2.5, 1], gap="large")
    
    with col1:
        st.markdown('<div class="elevated-card">', unsafe_allow_html=True)
        st.markdown('<h3 class="title-large"> Select Analysis Location</h3>', unsafe_allow_html=True)
        st.markdown('<p style="font-size: 0.9rem; color: #504349; margin-bottom: 1rem;"> Click anywhere on the map to select coordinates for risk assessment</p>', unsafe_allow_html=True)
        
        # Create map restricted to Nepal
        m2 = folium.Map(
            location=[28.3949, 84.1240],  # Center of Nepal
            zoom_start=7,
            min_zoom=6,
            max_bounds=True,
            min_lat=26.0,
            max_lat=31.0,
            min_lon=80.0,
            max_lon=89.0,
            tiles='CartoDB positron'
        )
        
        # Add marker if location selected
        if st.session_state.selected_location:
            folium.Marker(
                location=st.session_state.selected_location,
                popup=f"Analysis Point<br>{st.session_state.selected_location[0]:.4f}, {st.session_state.selected_location[1]:.4f}",
                icon=folium.Icon(color='blue', icon='crosshairs', prefix='fa')
            ).add_to(m2)
        
        map_data = st_folium(m2, height=600, key="risk_map", width='stretch')
        
        # Capture map click with validation
        if map_data and map_data.get('last_clicked'):
            clicked_lat = map_data['last_clicked']['lat']
            clicked_lon = map_data['last_clicked']['lng']
            
            # Validate coordinates are within Nepal
            if 26.0 <= clicked_lat <= 31.0 and 80.0 <= clicked_lon <= 89.0:
                st.session_state.selected_location = [clicked_lat, clicked_lon]
                st.session_state.env_data = None
                st.rerun()
            else:
                st.warning("⚠️ Please select a location within Nepal boundaries.")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="material-card">', unsafe_allow_html=True)
        
        if st.session_state.selected_location:
            lat, lon = st.session_state.selected_location
            
            # Location display
            st.markdown(f"""
            <div style="background: {COLORS['primary_container']}; padding: 1rem; border-radius: 16px; margin-bottom: 1.5rem;">
                <div class="label-large" style="color: {COLORS['on_primary_container']}; margin-bottom: 0.5rem;"> Selected Coordinates</div>
                <div style="font-size: 1.1rem; font-weight: 600; color: {COLORS['primary']};">
                    {lat:.4f}° N, {lon:.4f}° E
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Step 1: Fetch data
            if st.button(" Fetch Environmental Data", type="primary", width='stretch'):
                with st.spinner(" Collecting real-time data from Weather API & Google Earth Engine..."):
                    try:
                        pre_fire = backend["pre_fire"]
                        features = pre_fire.feature_engineer.get_all_features(lat, lon)
                        
                        if features and len(features) > 0:
                            st.session_state.env_data = {
                                "latitude": lat,
                                "longitude": lon,
                                "date": datetime.now().strftime('%Y-%m-%d'),
                                "features": features
                            }
                            st.success(f" Successfully retrieved {len(features)} environmental parameters")
                            st.rerun()
                        else:
                            st.error(" No data returned. Please verify API credentials and location validity.")
                            
                    except Exception as e:
                        st.error(f" Data acquisition failed: {str(e)}")
                        st.info(" Troubleshooting:\n- Verify Google Earth Engine authentication\n- Check Weather API credentials\n- Ensure coordinates are valid")
            
            # Display environmental data
            if st.session_state.env_data:
                features = st.session_state.env_data['features']
                
                st.markdown("####  Current Weather Conditions")
                
                # Metrics grid
                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{features.get('relative_humidity_pct', 0):.1f}<span class="metric-unit">%</span></div>
                        <div class="metric-label">Relative Humidity</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col_b:
                    st.markdown(f"""
                    <div class="metric-card" style="border-left-color: {COLORS['secondary']};">
                        <div class="metric-value" style="color: {COLORS['secondary']};">{features.get('vapor_pressure_deficit_kpa', 0):.2f}<span class="metric-unit">kPa</span></div>
                        <div class="metric-label">VPD</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                col_c, col_d = st.columns(2)
                with col_c:
                    st.markdown(f"""
                    <div class="metric-card" style="border-left-color: {COLORS['tertiary']};">
                        <div class="metric-value" style="color: {COLORS['tertiary']};">{features.get('dewpoint_2m_celsius', 0):.1f}<span class="metric-unit">°C</span></div>
                        <div class="metric-label">Dew Point</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col_d:
                    st.markdown(f"""
                    <div class="metric-card" style="border-left-color: {COLORS['chart_2']};">
                        <div class="metric-value" style="color: {COLORS['chart_2']};">{features.get('clear_day_coverage', 0):.0f}<span class="metric-unit">%</span></div>
                        <div class="metric-label">Clear Sky</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Fire Proximity Metric
                col_e, col_f = st.columns(2)
                with col_e:
                    fire_count = features.get('fire_density_50km', 0)
                    
                    # Determine color based on count
                    fire_color = COLORS['low']
                    if fire_count > 5: fire_color = COLORS['medium']
                    if fire_count > 20: fire_color = COLORS['high']
                    
                    st.markdown(f"""
                    <div class="metric-card" style="border-left-color: {fire_color};">
                        <div class="metric-value" style="font-size: 2rem; color: {fire_color};">{fire_count}</div>
                        <div class="metric-label">Active Fires (50km)</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Historical trends chart
                st.markdown("####  Historical Trends (Past 7 Days)")
                
                # Create Plotly chart for humidity lag
                lag_data = {
                    'Day': ['7d ago', '3d ago', '1d ago', 'Today'],
                    'Humidity (%)': [
                        features.get('relative_humidity_pct_lag7', 0),
                        features.get('relative_humidity_pct_lag3', 0),
                        features.get('relative_humidity_pct_lag1', 0),
                        features.get('relative_humidity_pct', 0)
                    ]
                }
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=lag_data['Day'],
                    y=lag_data['Humidity (%)'],
                    mode='lines+markers',
                    name='Relative Humidity',
                    line=dict(color=COLORS['primary'], width=3),
                    marker=dict(size=10, color=COLORS['primary'], line=dict(width=2, color='white'))
                ))
                
                fig.update_layout(
                    height=250,
                    margin=dict(l=0, r=0, t=0, b=0),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(family='Google Sans', size=12, color=COLORS['on_surface']),
                    xaxis=dict(
                        showgrid=True,
                        gridcolor=COLORS['outline_variant'],
                        gridwidth=1,
                        title=dict(text='Time Period', font=dict(size=11, weight=500))
                    ),
                    yaxis=dict(
                        showgrid=True,
                        gridcolor=COLORS['outline_variant'],
                        gridwidth=1,
                        title=dict(text='Humidity (%)', font=dict(size=11, weight=500)),
                        range=[0, 100]
                    ),
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, width='stretch', config={'displayModeBar': False})
                
                # Satellite data
                st.markdown("####  Satellite Observations")
                col_g, col_h = st.columns(2)
                with col_g:
                    st.markdown(f"""
                    <div class="metric-card" style="border-left-color: {COLORS['success']};">
                        <div class="metric-value" style="color: {COLORS['success']};">{features.get('landsat_savi', 0):.3f}</div>
                        <div class="metric-label">SAVI Index</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col_h:
                    st.markdown(f"""
                    <div class="metric-card" style="border-left-color: {COLORS['error']};">
                        <div class="metric-value" style="color: {COLORS['error']};">{features.get('lst_day_c', 0):.1f}<span class="metric-unit">°C</span></div>
                        <div class="metric-label">LST (Day)</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Step 2: Risk assessment
                if st.button(" Assess Fire Risk", type="primary", width='stretch'):
                    with st.spinner(" Running CatBoost ML model for risk prediction..."):
                        try:
                            pre_fire = backend["pre_fire"]
                            result = pre_fire.predict_from_features(features)
                            
                            if "error" in result:
                                st.error(f" Model error: {result['error']}")
                                st.info(" Ensure the ML model is properly trained and feature names match.")
                            else:
                                prob = result['probability']
                                level = result['risk_level']
                                alert = result['alert_priority']
                                
                                # Risk visualization
                                risk_color = {
                                    'Critical': COLORS['critical'],
                                    'High': COLORS['high'],
                                    'Medium': COLORS['medium'],
                                    'Low': COLORS['low']
                                }.get(level, COLORS['medium'])
                                
                                st.markdown(f"""
                                <div style="background: linear-gradient(135deg, {risk_color}15 0%, {risk_color}05 100%); 
                                            padding: 1.5rem; border-radius: 20px; border-left: 4px solid {risk_color};
                                            margin: 1rem 0;">
                                    <div style="font-size: 0.85rem; font-weight: 600; color: {risk_color}; 
                                                text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 0.5rem;">
                                         Risk Assessment
                                    </div>
                                    <div style="font-size: 2rem; font-weight: 700; color: {risk_color}; margin-bottom: 0.25rem;">
                                        {level.upper()}
                                    </div>
                                    <div style="font-size: 1.5rem; font-weight: 600; color: {COLORS['on_surface']};">
                                        {prob*100:.1f}% Probability
                                    </div>
                                    <div style="margin-top: 1rem; padding-top: 1rem; border-top: 1px solid {risk_color}30;">
                                        <span class="status-chip chip-{level.lower()}" style="border-left-color: {risk_color};">
                                             Alert Level: {alert.upper()}
                                        </span>
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                # Probability gauge
                                fig_gauge = go.Figure(go.Indicator(
                                    mode="gauge+number",
                                    value=prob * 100,
                                    title={'text': "Fire Risk Probability", 'font': {'size': 14, 'family': 'Google Sans'}},
                                    number={'suffix': "%", 'font': {'size': 32, 'family': 'Google Sans Display'}},
                                    gauge={
                                        'axis': {'range': [0, 100], 'tickwidth': 2, 'tickcolor': COLORS['outline']},
                                        'bar': {'color': risk_color, 'thickness': 0.8},
                                        'bgcolor': COLORS['surface_container'],
                                        'borderwidth': 2,
                                        'bordercolor': COLORS['outline_variant'],
                                        'steps': [
                                            {'range': [0, 40], 'color': hex_to_rgba(COLORS['low'], 0.2)},
                                            {'range': [40, 60], 'color': hex_to_rgba(COLORS['medium'], 0.2)},
                                            {'range': [60, 80], 'color': hex_to_rgba(COLORS['high'], 0.2)},
                                            {'range': [80, 100], 'color': hex_to_rgba(COLORS['critical'], 0.2)}
                                        ],
                                        'threshold': {
                                            'line': {'color': "white", 'width': 4},
                                            'thickness': 0.75,
                                            'value': prob * 100
                                        }
                                    }
                                ))
                                
                                fig_gauge.update_layout(
                                    height=250,
                                    margin=dict(l=20, r=20, t=40, b=20),
                                    paper_bgcolor='rgba(0,0,0,0)',
                                    font=dict(family='Google Sans', color=COLORS['on_surface'])
                                )
                                
                                st.plotly_chart(fig_gauge, width='stretch', config={'displayModeBar': False})
                                
                                # Recommendations
                                st.markdown("####  Recommended Actions")
                                if alert == "Critical":
                                    st.markdown(f"""
                                    <div class="material-card" style="background: {COLORS['error']}10; border-left: 4px solid {COLORS['error']};">
                                        <ul style="margin: 0; padding-left: 1.5rem; color: {COLORS['on_surface']};">
                                            <li style="margin: 0.5rem 0;"><strong> Immediate mobilization recommended</strong></li>
                                            <li style="margin: 0.5rem 0;"> Alert local response teams</li>
                                            <li style="margin: 0.5rem 0;"> Monitor satellite feeds continuously</li>
                                            <li style="margin: 0.5rem 0;"> Prepare evacuation plans</li>
                                        </ul>
                                    </div>
                                    """, unsafe_allow_html=True)
                                elif alert == "High":
                                    st.markdown(f"""
                                    <div class="material-card" style="background: {COLORS['warning']}10; border-left: 4px solid {COLORS['warning']};">
                                        <ul style="margin: 0; padding-left: 1.5rem; color: {COLORS['on_surface']};">
                                            <li style="margin: 0.5rem 0;"> Prepare fire suppression resources</li>
                                            <li style="margin: 0.5rem 0;"> Increase monitoring frequency</li>
                                            <li style="margin: 0.5rem 0;"> Issue public awareness warnings</li>
                                        </ul>
                                    </div>
                                    """, unsafe_allow_html=True)
                                elif alert == "Watch":
                                    st.markdown(f"""
                                    <div class="material-card" style="background: {COLORS['info']}10; border-left: 4px solid {COLORS['info']};">
                                        <ul style="margin: 0; padding-left: 1.5rem; color: {COLORS['on_surface']};">
                                            <li style="margin: 0.5rem 0;"> Monitor weather changes closely</li>
                                            <li style="margin: 0.5rem 0;">  Verify with local ground observations</li>
                                        </ul>
                                    </div>
                                    """, unsafe_allow_html=True)
                                else:
                                    st.markdown(f"""
                                    <div class="material-card" style="background: {COLORS['success']}10; border-left: 4px solid {COLORS['success']};">
                                        <p style="margin: 0; color: {COLORS['on_surface']};">
                                             Routine monitoring sufficient<br>
                                             No immediate threat detected
                                        </p>
                                    </div>
                                    """, unsafe_allow_html=True)
                                
                                # Show ALL parameters after assessment
                                st.markdown("---")
                                with st.expander(" View Complete Parameter List", expanded=False):
                                    st.markdown("##### All Model Input Features (From Real APIs)")
                                    # Convert features to a readable dataframe
                                    params_df = pd.DataFrame(list(features.items()), columns=['Parameter', 'Value'])
                                    # Format values (round floats)
                                    params_df['Value'] = params_df['Value'].apply(lambda x: f"{x:.4f}" if isinstance(x, float) else x)
                                    st.dataframe(params_df, width='stretch', hide_index=True)
                                
                        except Exception as e:
                            st.error(f" Analysis failed: {str(e)}")
                            st.info(" Ensure ML model is loaded and feature names match the training data.")
        else:
            st.markdown(f"""
            <div class="material-card" style="text-align: center; padding: 2rem;">
                <div style="font-size: 3rem; margin-bottom: 1rem;"></div>
                <p style="color: {COLORS['on_surface_variant']}; margin: 0;">
                    Click on the map to select a location for risk assessment
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
