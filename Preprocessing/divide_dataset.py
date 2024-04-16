import pandas as pd
from sklearn.model_selection import train_test_split

# Load the CSV file into a DataFrame
df = pd.read_csv('input_mod2.csv', encoding="latin-1")

# Specify the column containing the target variable (class labels)
target_column = 'Class'

# Perform stratified split into training (40%), validation (30%), and test (30%) sets
train_data, test_data = train_test_split(df, test_size=0.6, stratify=df[target_column], random_state=42)
valid_data, test_data = train_test_split(test_data, test_size=0.5, stratify=test_data[target_column], random_state=42)

# Save the resulting sets to new CSV files
train_data.to_csv('in_train_ds2.csv', index=False)
valid_data.to_csv('in_valid_ds2.csv', index=False)
test_data.to_csv('in_test_ds2.csv', index=False)
