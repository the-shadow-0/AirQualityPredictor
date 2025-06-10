import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

def preprocess_data(file_path):
    # Read data
    df = pd.read_csv(file_path, delimiter=';', decimal=',')
    
    # Clean data
    df = df.dropna(axis=1, how='all').dropna(axis=0, how='all')
    df.columns = df.columns.str.strip()
    
    # Convert to numeric
    numeric_cols = ['CO(GT)', 'PT08.S1(CO)', 'C6H6(GT)', 'NOx(GT)', 'T', 'RH', 'PT08.S5(O3)']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Handle missing values
    df.replace(-200, np.nan, inplace=True)
    imputer = SimpleImputer(strategy='median')
    df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
    
    # Feature engineering
    df['DateTime'] = pd.to_datetime(
        df['Date'] + ' ' + df['Time'], 
        format='%d/%m/%Y %H.%M.%S',
        errors='coerce'
    )
    df['Hour'] = df['DateTime'].dt.hour
    df['DayOfWeek'] = df['DateTime'].dt.dayofweek
    
    # Select features and target
    features = df[['T', 'RH', 'CO(GT)', 'C6H6(GT)', 'NOx(GT)', 'Hour', 'DayOfWeek']]
    target = df['PT08.S5(O3)']
    
    return features, target