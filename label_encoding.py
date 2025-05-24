import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# --- 1. Paths ---
input_path  = 'original_dataset/cybersecurity_intrusion_data.csv.xlsx'
output_dir  = 'Network_data'
output_file = 'preprocessed_dataset.csv'
output_path = os.path.join(output_dir, output_file)

# --- 2. Load data ---
df = pd.read_excel(input_path)

# --- 3. Label-encode categorical columns ---
le = LabelEncoder()
cat_cols = df.select_dtypes(include=['object', 'category']).columns
for col in cat_cols:
    df[col] = le.fit_transform(df[col].astype(str))

# --- 4. Min–Max scale + round specified numeric columns ---
scale_cols = ['network_packet_size', 'session_duration', 'ip_reputation_score']
scaler = MinMaxScaler()

# Fit & transform
df[scale_cols] = scaler.fit_transform(df[scale_cols])

# Round to 3 decimal places
df[scale_cols] = df[scale_cols].round(3)

# --- 5. Save ---
os.makedirs(output_dir, exist_ok=True)
df.to_csv(output_path, index=False)

print(f"✅ Label-encoded, MinMax-scaled, and rounded ({', '.join(scale_cols)}) dataset saved to: {output_path}")
