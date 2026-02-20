import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the dataset generated in Phase 1
df = pd.read_csv('nhai_accident_data.csv')

def preprocess_data(data):
    # Extract Hour
    data['Timestamp'] = pd.to_datetime(data['Timestamp'])
    data['Hour'] = data['Timestamp'].dt.hour
    
    # 2. Categorical Encoding (UPDATED with new features)
    categorical_cols = ['Weather', 'Road_Curvature', 'Lighting', 'Surface_Condition', 'Historical_Accident_Cause', 'News_Sentiment']
    data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)
    
    # 3. Scaling Numerical Features (UPDATED with new features)
    scaler = StandardScaler()
    numerical_cols = ['Road_Surface_Friction', 'Traffic_Density', 'HCV_Ratio', 'Hour', 'Speed_Limit_Breaches', 'Rash_Driving_Incidents']
    data[numerical_cols] = scaler.fit_transform(data[numerical_cols])
    
    # 4. Drop unused columns
    processed_df = data.drop(['Timestamp'], axis=1)
    return processed_df

processed_df = preprocess_data(df)
processed_df.to_csv('nhai_processed_data.csv', index=False)
print("Preprocessing Complete!")