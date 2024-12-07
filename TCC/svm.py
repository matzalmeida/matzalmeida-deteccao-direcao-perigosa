import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import pandas as pd
import shap
from auxiliar import gen_balanced_dataset
from auxiliar import evaluate_model
import matplotlib.pyplot as plt
import os

def test_svm(X, y, seed):
    # Divide em conjunto de treinos e testes
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )

    # Define and train the SVM model
    
    svm_model = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(kernel='rbf', C=1.0, random_state=seed))
    ])
    svm_model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = svm_model.predict(X_test)

    return evaluate_model(y_test, y_pred)

VERBOSE = 0

file_paths = ['unfiltered_10hz.csv', 'mean_10hz.csv', 'iir_10hz.csv']
random_states = [1, 42, 30, 55, 60, 70, 80, 99, 11, 32,
                15, 19, 44, 66, 45, 33, 666, 23, 98, 90]

data = dict()
for file in file_paths:
    data[file] = pd.DataFrame()

print("Processando...")
for seed in random_states:
    for file in file_paths:
        if (VERBOSE == 1):
            print("==============")
            print(file)
            print("==============")

        X, y = gen_balanced_dataset("data/{fname}".format(fname=file), seed)
        
        accuracy, precision, recall, f1, conf_matrix_df = test_svm(X, y, seed)
        TN = conf_matrix_df.iat[0,0]
        FN = conf_matrix_df.iat[0,1]
        FP = conf_matrix_df.iat[1,0]
        TP = conf_matrix_df.iat[1,1]
        new_row = pd.DataFrame({"Seed": seed, "Acurácia": accuracy, "Precisão": precision, "Recall": recall, "F1-Score": f1, "TP": TP, "FP": FP, "TN": TN, "FN": FN, "Suporte(P)": TP+FP, "Suporte(N)": TN+FN}, index=[0])
        data[file] = pd.concat([data[file], new_row], ignore_index=True)


try:
    os.mkdir('resultados/svm/')
    print(f"Directory '{'resultados/svm/'}' created successfully.")
except FileExistsError:
    print(f"Directory '{'resultados/svm/'}' already exists.")
except PermissionError:
    print(f"Permission denied: Unable to create '{'resultados/svm/'}'.")
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
    
    data[file].to_csv('resultados/svm/{fname}'.format(fname=file), index=False)
    