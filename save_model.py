import pickle
from model_training import train_model
from data_preparation import load_data, preprocess_data

if __name__ == "__main__":
    df = load_data('../data/menopause_data.csv')
    X_train, X_test, y_train, y_test = preprocess_data(df)
    model = train_model(X_train, y_train)
    
    # Save the model
    with open('../models/model.pkl', 'wb') as file:
        pickle.dump(model, file)

    # Save the scaler
    with open('../models/scaler.pkl', 'wb') as file:
        pickle.dump(StandardScaler().fit(X_train), file)
