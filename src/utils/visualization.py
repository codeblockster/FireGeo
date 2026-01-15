import plotly.graph_objects as go
import plotly.express as px
import folium

def create_spread_heatmap(prediction_array, lat_range, lon_range):
    """Create interactive heatmap of fire spread prediction"""
    # Placeholder for heatmap generation logic
    # In a real app, this would use the grid data to create a proper overlay
    fig = go.Figure(data=go.Heatmap(
        z=prediction_array,
        x=lon_range, # Array of longitudes
        y=lat_range, # Array of latitudes
        colorscale='Hot',
        colorbar=dict(title="Fire Probability")
    ))
    
    fig.update_layout(
        title="Fire Spread Prediction - Next 24 Hours",
        xaxis_title="Longitude",
        yaxis_title="Latitude"
    )
    return fig

def create_risk_map(risk_scores, locations):
    """Create choropleth map of risk scores"""
    # locations: dataframe with 'lat', 'lon'
    fig = go.Figure(data=go.Scattergeo(
        lon=locations['lon'],
        lat=locations['lat'],
        text=risk_scores,
        mode='markers',
        marker=dict(
            size=10,
            color=risk_scores,
            colorscale='RdYlGn_r', # Red for high risk, Green for low
            showscale=True,
            colorbar=dict(title="Risk Score")
        )
    ))
    return fig

def add_fire_markers(map_obj, fire_data):
    """Add fire markers to a folium map"""
    for fire in fire_data:
        folium.CircleMarker(
            location=[fire['latitude'], fire['longitude']],
            radius=5,
            color='red',
            fill=True,
            popup=f"Confidence: {fire.get('confidence', 'N/A')}"
        ).add_to(map_obj)
    return map_obj
