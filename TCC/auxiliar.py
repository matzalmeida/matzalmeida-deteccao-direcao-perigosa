import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

def plot_curves(history):
    # Plot the training and validation loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.show()

    # Plot the training and validation accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.show()

def evaluate_model(y_test, y_pred, VERBOSE=0):
    # Evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Print metrics
    if VERBOSE == 1:
        print("Métricas de avaliação:")
        print(f"Acurácia: {accuracy*100:.2f}")
        print(f"Precisão: {precision*100:.2f}")
        print(f"Recall: {recall*100:.2f}")
        print(f"F1 Score: {f1*100:.2f}")
        print("--------------")
        print("\nMatriz de confusão:")
    # Compute confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    # Define class labels
    class_labels = ["Perigoso", "Seguro"]  # Replace with your class names if available
    # Create a DataFrame for the labeled confusion matrix
    conf_matrix_df = pd.DataFrame(conf_matrix, index=class_labels, columns=class_labels)
    # Print labeled confusion matrix
    if VERBOSE == 1:
        print(conf_matrix_df)
        print("--------------")

    # Classification report (optional)
    if VERBOSE == 1:
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

    return accuracy, precision, recall, f1, conf_matrix_df

def gen_balanced_dataset(file_path, seed=42):
    # Load CSV file into a DataFrame
    data = pd.read_csv(file_path)

    # Inspect the data
    # print(data.head())  # Show the first few rows of the data

    # Separate features and labels
    # Assuming the last column is the label (target), and the rest are features
    X = data.iloc[:, 3:].values  # Features (all columns except the last one)
    y = data.iloc[:, 1].values   # Labels (the last column)

    # Separate by class
    class_0 = data[data['Seguro'] == 0]
    class_1 = data[data['Seguro'] == 1]

    # Find the minimum class size
    min_class_size = min(len(class_0), len(class_1))

    # Downsample each class to the minimum size
    class_0_balanced = class_0.sample(n=min_class_size, random_state=seed)
    class_1_balanced = class_1.sample(n=min_class_size, random_state=seed)

    # Combine the balanced classes
    balanced_data = pd.concat([class_0_balanced, class_1_balanced])

    # Shuffle the dataset
    balanced_data = balanced_data.sample(frac=1, random_state=seed).reset_index(drop=True)

    # Split the dataset back into features (X) and labels (y)
    X_balanced = balanced_data.drop(['Seguro', 'Arquivo', 'Motorista'], axis=1).values
    y_balanced = balanced_data['Seguro'].values
    
    return X_balanced, y_balanced