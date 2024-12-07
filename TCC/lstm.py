import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from auxiliar import gen_balanced_dataset
from auxiliar import evaluate_model
from auxiliar import plot_curves
import os

def test_lstm(X, y, seed):
    # Divide em conjunto de treinos e testes
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )

    # Check the shapes of the resulting datasets
    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_train shape:", y_train.shape)
    print("y_test shape:", y_test.shape)
    
    # Define the LSTM model
    def create_lstm_model(input_shape):
        model = Sequential([
            Input(shape=input_shape),
            LSTM(64, return_sequences=True),  # LSTM layer
            Dropout(0.5),
            LSTM(32, return_sequences=False),  # LSTM layer
            Dropout(0.5),
            Dense(1, activation='sigmoid')  # Output layer for binary classification
        ])
        return model

    # Hyperparameters
    input_shape = (X_train.shape[1], X_train.shape[2])  # Number of timesteps and features
    print("X_train.shape[1] =", X_train.shape[1],
          "X_train.shape[2] =", X_train.shape[2])

    # Instantiate the model
    model = create_lstm_model(input_shape)

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Model summary
    model.summary()

    # Define early stopping to avoid overfitting
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # Train the model
    history = model.fit(X_train, y_train, 
                        epochs=50, 
                        batch_size=32, 
                        validation_split=0.2, 
                        callbacks=[early_stopping], 
                        verbose=1)

    # Predict on the test set
    y_pred_probs = model.predict(X_test)  # Predicted probabilities
    y_pred = (y_pred_probs > 0.5).astype(int).flatten()  # Convert probabilities to binary labels (0 or 1)

    return evaluate_model(y_test, y_pred)
    plot_curves(history)

file_paths = ['unfiltered_10hz.csv', 'mean_10hz.csv', 'iir_10hz.csv']
random_states = [1, 42, 30, 55, 60, 70, 80, 99, 11, 32,
                15, 19, 44, 66, 45, 33, 666, 23, 98, 90]

data = dict()
for file in file_paths:
    data[file] = pd.DataFrame()

for seed in random_states:
    for file in file_paths:
        print("==============")
        print(file)
        print("==============")
        X_balanced, y = gen_balanced_dataset("data/{fname}".format(fname=file), seed)

        # Feature Scaling (important for LSTM)
        scaler = StandardScaler()
        X_balanced_scaled = scaler.fit_transform(X_balanced)

        print(X_balanced_scaled.shape[0])
        
        # Reshape the data for LSTM input: (samples, timesteps, features)
        # Assuming each sample is a time step, we reshape the input to (samples, 1, features)
        X = X_balanced_scaled.reshape(X_balanced_scaled.shape[0], 50, 6)

        accuracy, precision, recall, f1, conf_matrix_df = test_lstm(X, y, seed)
        TN = conf_matrix_df.iat[0,0]
        FN = conf_matrix_df.iat[0,1]
        FP = conf_matrix_df.iat[1,0]
        TP = conf_matrix_df.iat[1,1]
        new_row = pd.DataFrame({"Seed": seed, "Acurácia": accuracy, "Precisão": precision, "Recall": recall, "F1-Score": f1, "TP": TP, "FP": FP, "TN": TN, "FN": FN, "Suporte(P)": TP+FP, "Suporte(N)": TN+FN}, index=[0])
        data[file] = pd.concat([data[file], new_row], ignore_index=True)

try:
    os.mkdir('resultados/lstm/')
    print(f"Directory '{'resultados/lstm/'}' created successfully.")
except FileExistsError:
    print(f"Directory '{'resultados/lstm/'}' already exists.")
except PermissionError:
    print(f"Permission denied: Unable to create '{'resultados/lstm/'}'.")
except Exception as e:
    print(f"An error occurred: {e}")
for file in file_paths:
    print(file)
    total = data[file].sum()
    total = pd.DataFrame(np.array(total).reshape(-1,len(total)))
    total[[0]] = "TOTAL"
    total = total.rename(columns={0:"Seed", 1:"Acurácia", 2:"Precisão", 3:"Recall", 4:"F1-Score", 
                                  5:"TP", 6:"FP", 7:"TN", 8:"FN", 9:"Suporte(P)", 10:"Suporte(N)"})
    mean = data[file].mean()
    mean = pd.DataFrame(np.array(mean).reshape(-1,len(mean)))
    mean[[0]] = "MÉDIA"
    mean = mean.rename(columns={0:"Seed", 1:"Acurácia", 2:"Precisão", 3:"Recall", 4:"F1-Score", 
                                5:"TP", 6:"FP", 7:"TN", 8:"FN", 9:"Suporte(P)", 10:"Suporte(N)"})
    data[file] = pd.concat([data[file], mean], ignore_index=True)
    data[file] = pd.concat([data[file], total], ignore_index=True)
    print(data[file])
    
    data[file].to_csv('resultados/lstm/{fname}'.format(fname=file), index=False)