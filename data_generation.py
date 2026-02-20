import pandas as pd
import numpy as np

# 1. Setup Parameters
np.random.seed(42)
num_records = 6000 # Perfect amount for a city-level dataset

# 2. Define Dense Mumbai Hotspots (Intersections, Flyovers, BKC, etc.)
# Mumbai bounds approx: Lat 18.90 (South) to 19.27 (North Dahisar), Lon 72.82 (West) to 73.00 (East Mulund/Navi Mumbai)
hotspots = []
for _ in range(150): # 150 dense spots just in Mumbai
    h_lat = np.random.uniform(18.92, 19.25)
    h_lon = np.random.uniform(72.83, 72.98)
    
    # Strict Ocean Filter for Mumbai Coastline: 
    # South Mumbai narrows heavily. If we are far South (Colaba/Worli), push Longitude East.
    if h_lat < 19.05 and h_lon < 72.82:
        h_lon = 72.82 + np.random.uniform(0.01, 0.03)
    if h_lat < 18.98 and h_lon < 72.83:
        h_lon = 72.83 + np.random.uniform(0.01, 0.02)
        
    # Radius is very small (representing city blocks/intersections)
    hotspots.append({"lat": h_lat, "lon": h_lon, "radius": np.random.uniform(0.002, 0.008)})

def generate_location():
    lat = []
    lon = []
    for _ in range(num_records):
        # Pick a random Mumbai hotspot
        spot = np.random.choice(hotspots)
        
        # Distribute accidents very tightly around that specific junction
        l_lat = spot["lat"] + np.random.normal(0, 0.004)
        l_lon = spot["lon"] + np.random.normal(0, 0.004)
        
        lat.append(l_lat)
        lon.append(l_lon)
    return np.array(lat), np.array(lon)

latitudes, longitudes = generate_location()

# 3. Generate Features
data = {
    'Latitude': latitudes, 'Longitude': longitudes,
    'Timestamp': pd.to_datetime(np.random.randint(1672531200, 1704067200, num_records), unit='s'),
    'Weather': np.random.choice(['Fine', 'Mist/Fog', 'Heavy Rain', 'Dust Storm'], num_records, p=[0.7, 0.1, 0.15, 0.05]),
    'Road_Surface_Friction': np.random.uniform(0.3, 0.8, num_records),
    'Traffic_Density': np.random.randint(50, 500, num_records),
    'HCV_Ratio': np.random.uniform(0.05, 0.4, num_records), # Less trucks in city than open highway
    'Road_Curvature': np.random.choice(['Straight', 'Slight Curve', 'Sharp Curve'], num_records),
    'Lighting': np.random.choice(['Daylight', 'Dusk', 'Night-Lit', 'Night-Unlit'], num_records),
    'Surface_Condition': np.random.choice(['Smooth', 'Minor Potholes', 'Severe Potholes'], num_records, p=[0.6, 0.3, 0.1]),
    'Historical_Accident_Cause': np.random.choice(['None', 'Speeding', 'Potholes', 'Blind Spot Collision'], num_records, p=[0.7, 0.15, 0.1, 0.05]),
    'News_Sentiment': np.random.choice(['Positive (Safe)', 'Neutral', 'Negative (Accident Prone)'], num_records, p=[0.3, 0.5, 0.2]),
    'Speed_Limit_Breaches': np.random.randint(0, 50, num_records),
    'Rash_Driving_Incidents': np.random.randint(0, 20, num_records) 
}

df = pd.DataFrame(data)

# 4. Logic: Define "Accident_Occurred"
risk_score = np.zeros(num_records)

df_lat = df['Latitude'].values
df_lon = df['Longitude'].values
for spot in hotspots:
    dist = np.sqrt((df_lat - spot['lat'])**2 + (df_lon - spot['lon'])**2)
    risk_score[dist < spot['radius']] += 0.3

# Apply risk factors
risk_score[df['Weather'] != 'Fine'] += 0.15
risk_score[df['Road_Curvature'] == 'Sharp Curve'] += 0.15
risk_score[df['Lighting'] == 'Night-Unlit'] += 0.1
risk_score[df['Road_Surface_Friction'] < 0.4] += 0.1
risk_score[df['Surface_Condition'] == 'Severe Potholes'] += 0.2
risk_score[df['Speed_Limit_Breaches'] > 30] += 0.2
risk_score[df['Rash_Driving_Incidents'] > 10] += 0.2
risk_score[df['News_Sentiment'] == 'Negative (Accident Prone)'] += 0.1

noise = np.random.normal(0, 0.05, num_records)
df['Accident_Risk_Label'] = (risk_score + noise > 0.6).astype(int)

# 5. Save and Preview
df.to_csv('nhai_accident_data.csv', index=False)
print(f"Dense Mumbai Dataset Created! Shape: {df.shape}")