import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import xgboost as xgb
import numpy as np
import datetime
from geopy.geocoders import Nominatim # Free geocoding

# 1. Configuration & Title
st.set_page_config(page_title="NHAI Live Risk Portal", layout="wide")
st.title("🛣️ Live AI Road Accident Risk Predictor")

# 2. Load Resources
@st.cache_resource
def load_assets():
    model = xgb.XGBClassifier()
    model.load_model('nhai_risk_model.json')
    hotspots = pd.read_csv('nhai_hotspots.csv')
    processed_df = pd.read_csv('nhai_processed_data.csv') 
    return model, hotspots, processed_df

model, hotspots, processed_df = load_assets()

# --- LIVE LOCATION SEARCH ---
st.sidebar.header("🔍 Live API Integration")
search_location = st.sidebar.text_input("Enter Highway/Location", "Mumbai Pune Highway")

# Free Geocoder to get real Lat/Lon for the map
geolocator = Nominatim(user_agent="nhai_hackathon_app")

# --- MOCK API FUNCTIONS (Replace with real requests.get later) ---
def fetch_live_weather_api(lat, lon):
    # Imagine this hits OpenWeatherMap API. We simulate a response.
    # In reality: requests.get(f"api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid=YOUR_KEY")
    return np.random.choice(["Fine", "Mist/Fog", "Heavy Rain", "Dust Storm"], p=[0.6, 0.1, 0.2, 0.1])

def fetch_live_traffic_api(lat, lon):
    # Imagine this hits Google Maps API (duration_in_traffic)
    return np.random.randint(50, 450) # Returns vehicles/km

def calculate_smart_hcv(hour):
    # Smart Proxy: Trucks/HCVs are much higher at night on NHAI highways
    if hour >= 22 or hour <= 6:
        return np.random.uniform(0.5, 0.8) # 50-80% trucks at night
    else:
        return np.random.uniform(0.1, 0.3) # 10-30% trucks during day

if st.sidebar.button("Fetch Live Conditions"):
    with st.spinner(f"Connecting to Satellites for {search_location}..."):
        try:
            location_data = geolocator.geocode(search_location)
            search_lat, search_lon = location_data.latitude, location_data.longitude
            
            # 1. Get Current Time
            current_hour = datetime.datetime.now().hour
            
            # 2. Fetch Live Data
            live_weather = fetch_live_weather_api(search_lat, search_lon)
            live_traffic = fetch_live_traffic_api(search_lat, search_lon)
            live_hcv = calculate_smart_hcv(current_hour)
            
            # Save to session state so UI updates
            st.session_state['lat'] = search_lat
            st.session_state['lon'] = search_lon
            st.session_state['weather'] = live_weather
            st.session_state['traffic'] = live_traffic
            st.session_state['hcv'] = live_hcv
            st.session_state['hour'] = current_hour
            st.success("Live data retrieved successfully!")
        except Exception as e:
            st.sidebar.error("Location not found. Try 'Pune' or 'Mumbai'.")

st.sidebar.markdown("---")
st.sidebar.header("Current Conditions (Live/Manual)")

# Fallback/Default values if API hasn't been called yet
weather = st.session_state.get('weather', "Fine")
traffic = st.session_state.get('traffic', 150)
hcv = st.session_state.get('hcv', 0.3)
hour = st.session_state.get('hour', 14)
map_lat = st.session_state.get('lat', 18.5204) # Default Pune
map_lon = st.session_state.get('lon', 73.8567)

# Display the fetched data, allowing user to still tweak it
weather = st.sidebar.selectbox("Weather (Live)", ["Fine", "Mist/Fog", "Heavy Rain", "Dust Storm"], index=["Fine", "Mist/Fog", "Heavy Rain", "Dust Storm"].index(weather))
road_curve = st.sidebar.selectbox("Road Geometry (Static)", ["Straight", "Slight Curve", "Sharp Curve"])
traffic = st.sidebar.slider("Traffic Density (Live)", 50, 500, int(traffic))
hcv = st.sidebar.slider("HCV Ratio (Smart Proxy)", 0.1, 0.8, float(hcv))

# --- Smart Lighting Logic ---
if 6 <= hour <= 17:
    auto_light = "Daylight"
elif (18 <= hour <= 19) or (5 <= hour <= 6):
    auto_light = "Dusk"
else:
    auto_light = "Night-Lit"

lighting = st.sidebar.selectbox("Lighting (Auto)", ["Daylight", "Dusk", "Night-Lit", "Night-Unlit"], index=["Daylight", "Dusk", "Night-Lit", "Night-Unlit"].index(auto_light))

def generate_actionable_insights(risk, weather, traffic, hcv, lighting, curve):
    suggestions = []
    
    # 1. Base Risk Level Interventions
    if risk > 0.60:
        suggestions.append("🚨 **CRITICAL:** Pre-position emergency response vehicles (ambulances/cranes) within a 5km radius of this zone immediately.")
    elif risk > 0.30:
        suggestions.append("⚠️ **WARNING:** Increase highway patrol frequency in this sector for the next 4 hours.")

    # 2. Weather & Lighting Based Solutions
    if weather in ["Heavy Rain", "Mist/Fog"]:
        suggestions.append(f"🌧️ **WEATHER ALERT:** Activate Variable Message Signs (VMS) warning drivers of '{weather}'. Mandate a temporary speed limit reduction of 20 km/h.")
    if lighting == "Night-Unlit":
        suggestions.append("💡 **INFRASTRUCTURE:** High accident probability due to zero visibility. Dispatch temporary mobile floodlights and flag this sector for priority street-lamp installation.")

    # 3. Traffic & Heavy Vehicle Interventions
    if traffic > 350 and hcv > 0.4:
        suggestions.append("🚚 **TRAFFIC CONTROL:** Dangerous mix of severe congestion and high Truck/Bus volume. Suggest diverting Heavy Commercial Vehicles (HCVs) to alternative corridors or enforcing left-lane-only discipline.")
    elif traffic > 400:
        suggestions.append("🚗 **CONGESTION:** Stop toll plaza collection temporarily if tailbacks exceed 2km to prevent rear-end collisions.")

    # 4. Road Geometry
    if curve == "Sharp Curve":
        suggestions.append("🛣️ **GEOMETRY:** Ensure chevron alignment signs (yellow/black arrows) are highly reflective. Consider installing transverse rumble strips 500m before the curve.")

    # Fallback for perfect conditions
    if not suggestions:
        suggestions.append("✅ Conditions are optimal. Maintain standard highway monitoring protocols.")
        
    return suggestions

# --- PREDICTION LOGIC ---
def predict_current_risk():
    feature_cols = [col for col in processed_df.columns if col not in ['Accident_Risk_Label', 'Latitude', 'Longitude']]
    input_df = pd.DataFrame(np.zeros((1, len(feature_cols))), columns=feature_cols)
    
    input_df['Traffic_Density'] = traffic
    input_df['HCV_Ratio'] = hcv
    input_df['Hour'] = hour
    input_df['Road_Surface_Friction'] = 0.6 
    
    if f'Weather_{weather}' in input_df.columns: input_df[f'Weather_{weather}'] = 1
    if f'Road_Curvature_{road_curve}' in input_df.columns: input_df[f'Road_Curvature_{road_curve}'] = 1
    if f'Lighting_{lighting}' in input_df.columns: input_df[f'Lighting_{lighting}'] = 1

    return model.predict_proba(input_df)[0][1]

risk_val = predict_current_risk()

# --- MAIN LAYOUT ---
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader(f"Live Map View: {search_location}")
    # Map now centers on the searched location!
    m = folium.Map(location=[map_lat, map_lon], zoom_start=11, control_scale=True)
    
    # Plot historical hotspots
    for _, row in hotspots[hotspots['Cluster'] != -1].iterrows():
        folium.CircleMarker(
            location=[row['Latitude'], row['Longitude']],
            radius=5, color='red', fill=True, popup=f"Historical Hotspot"
        ).add_to(m)
    
    # Plot current searched location
    folium.Marker([map_lat, map_lon], popup="Searched Location", icon=folium.Icon(color="blue", icon="info-sign")).add_to(m)
    
    st_folium(m, width=800, height=500)

with col2:
    st.subheader("Live Risk Assessment")
    st.metric(label="Real-Time Accident Probability", value=f"{risk_val:.2%}")
    
    # Risk Badge
    if risk_val > 0.6:
        st.error("⚠️ HIGH RISK: Immediate intervention recommended.")
    elif risk_val > 0.3:
        st.warning("⚡ MODERATE RISK: Increase highway patrolling.")
    else:
        st.success("✅ LOW RISK: Normal operations.")

    st.write("**Live Contributing Factors:**")
    st.info(f"Weather: {weather} | Lighting: {lighting} | Traffic: {int(traffic)} vehicles/km")

    # --- NEW: DYNAMIC SOLUTIONS PANEL ---
    st.markdown("---")
    st.subheader("🛡️ Automated NHAI Action Plan")
    
    # Call the engine we just built
    action_plan = generate_actionable_insights(risk_val, weather, traffic, hcv, lighting, road_curve)
    
    # Display the suggestions in a professional warning box
    if risk_val > 0.4:
        st.error('\n\n'.join(action_plan))
    elif risk_val > 0.2:
        st.warning('\n\n'.join(action_plan))
    else:
        st.success('\n\n'.join(action_plan))