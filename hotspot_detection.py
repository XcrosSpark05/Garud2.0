import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

# 1. Load the processed data from Phase 2
df = pd.read_csv('nhai_processed_data.csv')

def detect_hotspots(data):
    # We use only accidents (Label == 1) for hotspot detection
    accidents = data[data['Accident_Risk_Label'] == 1].copy()
    
    # Extract coordinates for clustering
    coords = accidents[['Latitude', 'Longitude']].values
    
    # 2. Configure DBSCAN
    # eps: The maximum distance between two samples for one to be considered as in the neighborhood of the other.
    # (0.01 roughly equals 1km depending on location)
    # min_samples: The number of samples in a neighborhood for a point to be considered a core point.
    db = DBSCAN(eps=0.01, min_samples=5).fit(np.radians(coords))
    
    # Add cluster labels back to the accident dataframe
    accidents['Cluster'] = db.labels_
    
    # -1 indicates noise/outliers (not part of a high-density hotspot)
    num_clusters = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
    print(f"Detected {num_clusters} high-risk 'Black Spots' (Hotspots).")
    
    return accidents

# Execute Detection
hotspot_df = detect_hotspots(df)

# 3. Simple Visualization of Hotspots
plt.figure(figsize=(10, 6))
# Plot outliers in grey
noise = hotspot_df[hotspot_df['Cluster'] == -1]
plt.scatter(noise['Longitude'], noise['Latitude'], c='grey', alpha=0.3, label='Isolated Incidents')

# Plot clusters in color
clusters = hotspot_df[hotspot_df['Cluster'] != -1]
plt.scatter(clusters['Longitude'], clusters['Latitude'], c=clusters['Cluster'], cmap='viridis', label='Detected Hotspots')

plt.title('NHAI Accident Hotspot Detection (Phase 3)')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend()
plt.show()

# Save for Dashboard mapping
hotspot_df.to_csv('nhai_hotspots.csv', index=False)