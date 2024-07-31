# data_preparation.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(filepath):
    df = pd.read_csv(filepath)
    return df

def preprocess_data(df):
    # Handle missing values
    df = df.dropna()

    # Encode categorical variables
    df = pd.get_dummies(df, drop_first=True)

    # Split features and target
    X = df.drop('menopause_status', axis=1)
    y = df['menopause_status']

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test
