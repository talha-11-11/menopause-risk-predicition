# feature_engineering.py
import pandas as pd

def create_features(df):
    # Example of feature engineering
    df['age_bmi_interaction'] = df['age'] * df['BMI']
    return df
