import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder

# Define paths
input_path = 'original_dataset/cybersecurity_intrusion_data.csv.xlsx'
output_dir = 'Network_data'
output_path = os.path.join(output_dir, 'preprocessed_dataset.csv')

# Read the Excel file
df = pd.read_excel(input_path)

# Initialize the LabelEncoder
le = LabelEncoder()

# Apply label encoding to all object or category columns
for column in df.select_dtypes(include=['object', 'category']).columns:
    df[column] = le.fit_transform(df[column].astype(str))

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Save the processed dataframe as CSV
df.to_csv(output_path, index=False)

print(f"Label-encoded dataset saved to: {output_path}")
