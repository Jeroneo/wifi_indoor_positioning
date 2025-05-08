# Sigmoid à la place de ReLU

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from torch.utils.data import Dataset, DataLoader
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import f1_score, precision_score  # Import additional metrics

import torch.optim as optim
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  # Import matplotlib for plotting
import time  # Import time for tracking training durations

import torch
import os
from tqdm import tqdm  # Import tqdm for progress tracking
from sklearn.metrics import confusion_matrix, classification_report

# Define the custom dataset and neural network model
class MissingValueDataset(Dataset):
    """Custom PyTorch Dataset for handling missing values"""
    
    def __init__(self, X, y, train=True):
        """
        Initialize the dataset with features and targets
        
        Parameters:
        - X: feature matrix, can contain np.nan for missing values
        - y: target labels (class indices)
        - train: whether this is training data
        """
        self.X_raw = X
        self.y = torch.tensor(y, dtype=torch.long)
        
        # Create missing indicators
        self.X_missing = torch.tensor(np.isnan(X).astype(np.float32))
        
        # Store the raw features without imputation
        self.X_scaled = torch.tensor(X, dtype=torch.float32)
        
    def __len__(self):
        return len(self.y)
        
    def __getitem__(self, idx):
        # Concatenate raw features with missing indicators
        X_combined = torch.cat([self.X_scaled[idx], self.X_missing[idx]], dim=0)
        return X_combined, self.y[idx]

class MissingValueClassifier(nn.Module):
    """Neural network classifier that handles missing values using indicator features"""
    
    def __init__(self, input_dim, output_dim=15, hidden_dims=[256, 128, 64]):
        """
        Initialize the neural network
        
        Parameters:
        - input_dim: original number of features (without missing indicators)
        - output_dim: number of output classes
        - hidden_dims: list of hidden layer sizes
        """
        super(MissingValueClassifier, self).__init__()
        
        # The input dimension is doubled because we add missing indicators
        augmented_input_dim = input_dim * 2
        
        # Build layers
        layers = []
        prev_dim = augmented_input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.Dropout(0.3))  # Add dropout for regularization
            prev_dim = hidden_dim
            
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.model(x)

def train_NN_model(model, train_loader, val_loader=None, epochs=50, learning_rate=0.001):
    """
    Train the PyTorch model
    
    Parameters:
    - model: PyTorch model
    - train_loader: DataLoader for training data
    - val_loader: DataLoader for validation data (optional)
    - epochs: number of training epochs
    - learning_rate: learning rate for optimizer
    
    Returns:
    - training history (loss and accuracy)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc)
        
        # Validation phase
        if val_loader:
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    
                    val_loss += loss.item() * inputs.size(0)
                    _, predicted = outputs.max(1)
                    val_total += targets.size(0)
                    val_correct += predicted.eq(targets).sum().item()
            
            val_epoch_loss = val_loss / val_total
            val_epoch_acc = val_correct / val_total
            history['val_loss'].append(val_epoch_loss)
            history['val_acc'].append(val_epoch_acc)
            
            scheduler.step(val_epoch_loss)
        else:
            pass
    
    return history

def load_data(file_path, test_size=0.3):
    """Load data from a CSV file"""

    data = pd.read_csv(file_path)
    X = data.drop(columns=["zone"]).values
    y = data["zone"].values
    y = pd.Categorical(y).codes  # Convert zone labels to integers

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    zone_labels = pd.Categorical(data["zone"]).categories # Store zone labels for later
    np.save('./models/zone_labels.npy', zone_labels)

    # Save the order of MAC addresses for later
    mac_address_order = data.drop(columns=["zone"]).columns.tolist()
    np.save('./models/mac_address_order.npy', mac_address_order)

    return X_train, X_test, y_train, y_test



def ensure_directory_exists(directory_path):
    """Ensure that a directory exists, creating it if necessary."""
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)


def calculate_metrics(y_true, y_pred, labels):
    """
    Calculate confusion matrix, classification report, precision, and F1 score.
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    return cm, report, precision, f1

def save_metrics_to_csv(cm, report, labels, output_path):
    """
    Save confusion matrix and classification report to CSV files.
    """
    metrics_df = pd.DataFrame(report).transpose()
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    metrics_df.to_csv(f"{output_path}_metrics.csv", index=True)
    cm_df.to_csv(f"{output_path}_confusion_matrix.csv")

def save_confusion_matrix_plot(cm, labels, output_path):
    """
    Save the confusion matrix as a plot.
    """
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)

    # Add text annotations
    fmt = 'd'
    thresh = cm.max() / 2
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], fmt),
                 ha="center", va="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(f"{output_path}_confusion_matrix.png")
    plt.close()

# Update evaluation sections in training functions
def evaluate_and_save_metrics(model, test_loader, labels, output_path):
    """
    Evaluate the model, calculate metrics, and save them to CSV and as a plot.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            y_true.extend(targets.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    # Ensure labels match the unique values in y_true
    unique_labels = sorted(set(y_true))
    label_to_int = {label: idx for idx, label in enumerate(unique_labels)}
    int_labels = list(label_to_int.values())
    y_true = [label_to_int[label] for label in y_true]
    y_pred = [label_to_int[label] for label in y_pred if label in label_to_int]

    cm, report, precision, f1 = calculate_metrics(y_true, y_pred, int_labels)
    save_metrics_to_csv(cm, report, labels, output_path)
    save_confusion_matrix_plot(cm, labels, output_path)
    return precision, f1

def save_training_history_plot(history, output_path):
    """
    Save the training history (loss and accuracy) as a plot.
    """
    plt.figure(figsize=(12, 6))

    # Plot training and validation loss
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss', color='blue')
    if 'val_loss' in history and history['val_loss']:
        plt.plot(history['val_loss'], label='Validation Loss', color='orange')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    # Plot training and validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy', color='blue')
    if 'val_acc' in history and history['val_acc']:
        plt.plot(history['val_acc'], label='Validation Accuracy', color='orange')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

# Modify train_NN_missing_data to include saving the entire model
def train_NN_missing_data(X_train, X_test, y_train, y_test, dataset_name="dataset", split_folder="./models", epochs=20):
    """
    Train a neural network to classify missing values in the dataset.
    """
    # Ensure the output directory exists
    ensure_directory_exists(f'{split_folder}/NN/{dataset_name}')

    # Create datasets and dataloaders
    train_dataset = MissingValueDataset(X_train, y_train)
    test_dataset = MissingValueDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)

    # Initialize and train model
    model = MissingValueClassifier(input_dim=X_train.shape[1])
    history = train_NN_model(model, train_loader, test_loader, epochs=epochs)

    # Evaluate model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    accuracy = correct / total

    # Save the entire model
    torch.save(model, f'{split_folder}/NN/{dataset_name}/NN_missing_value.pth')

    # Save training history to CSV
    history_df = pd.DataFrame(history)
    history_df.to_csv(f'{split_folder}/NN/{dataset_name}/training_history.csv', index=False)

    # Save training history as a plot
    save_training_history_plot(history, f'{split_folder}/NN/{dataset_name}/training_history.png')

    zone_labels = np.load('./models/zone_labels.npy', allow_pickle=True)
    precision, f1 = evaluate_and_save_metrics(model, test_loader, zone_labels, f'{split_folder}/NN/{dataset_name}/NN_missing_value')

    return model, history, accuracy, precision, f1

# Modify train_K_nearest_neighbors to include saving the entire model
def train_K_nearest_neighbors(X_train, X_test, y_train, y_test, dataset_name="dataset", split_folder="./models"):
    """
    Train a K-Nearest Neighbors classifier to classify missing values in the dataset.
    Test k values from 10 to 100 and select the best k based on accuracy.
    """
    try:
        # Ensure the output directory exists
        ensure_directory_exists(f'{split_folder}/KNN/{dataset_name}')

        # Use raw data without imputation
        X_train_raw = X_train
        X_test_raw = X_test

        # Test k values from 10 to 100
        best_k = None
        best_accuracy = 0
        best_model = None

        # Initialize progress bar for KNN
        k_values = range(10, 100)
        knn_progress_bar = tqdm(k_values, desc=f"KNN Training ({dataset_name})", unit="k")

        for k in knn_progress_bar:
            knn_model = KNeighborsClassifier(n_neighbors=k)
            knn_model.fit(X_train_raw, y_train)
            accuracy = knn_model.score(X_test_raw, y_test)

            if accuracy > best_accuracy:
                best_k = k
                best_accuracy = accuracy
                best_model = knn_model

            # Update progress bar description with the current best k and accuracy
            knn_progress_bar.set_postfix(best_k=best_k, best_accuracy=f"{best_accuracy:.4f}")

        knn_progress_bar.close()  # Close the KNN progress bar

        # Save the entire model
        torch.save(best_model, f'{split_folder}/KNN/{dataset_name}/KNN_missing_value.pth')

        zone_labels = np.load('./models/zone_labels.npy', allow_pickle=True)
        y_pred = best_model.predict(X_test_raw)
        cm, report, precision, f1 = calculate_metrics(y_test, y_pred, list(range(len(zone_labels))))
        save_metrics_to_csv(cm, report, zone_labels, f'{split_folder}/KNN/{dataset_name}/KNN_missing_value')
        save_confusion_matrix_plot(cm, zone_labels, f'{split_folder}/KNN/{dataset_name}/KNN_missing_value')

        return best_model, best_accuracy, best_k, precision, f1
    except ValueError as e:
        if "Input X contains NaN" in str(e):
            print(f"Skipping KNN model for dataset {dataset_name} due to NaN values in input.")
            return None, None, None, None, None
        else:
            raise

# Modify train_SVM to include saving the entire model
def train_SVM(X_train, X_test, y_train, y_test, dataset_name="dataset", split_folder="./models"):
    """
    Train a Support Vector Machine classifier to classify missing values in the dataset.
    """
    try:
        # Ensure the output directory exists
        ensure_directory_exists(f'{split_folder}/SVM/{dataset_name}')

        # Train SVM model
        svm_model = SVC(kernel='linear')
        svm_model.fit(X_train, y_train)

        # Evaluate model
        accuracy = svm_model.score(X_test, y_test)

        # Save the entire model
        torch.save(svm_model, f'{split_folder}/SVM/{dataset_name}/SVM_missing_value.pth')

        zone_labels = np.load('./models/zone_labels.npy', allow_pickle=True)
        y_pred = svm_model.predict(X_test)
        cm, report, precision, f1 = calculate_metrics(y_test, y_pred, list(range(len(zone_labels))))
        save_metrics_to_csv(cm, report, zone_labels, f'{split_folder}/SVM/{dataset_name}/SVM_missing_value')
        save_confusion_matrix_plot(cm, zone_labels, f'{split_folder}/SVM/{dataset_name}/SVM_missing_value')

        return svm_model, accuracy, precision, f1
    except ValueError as e:
        if "Input X contains NaN" in str(e):
            print(f"Skipping SVM model for dataset {dataset_name} due to NaN values in input.")
            return None, None, None, None
        else:
            raise

# Modify train_RandomForest to include saving the entire model
def train_RandomForest(X_train, X_test, y_train, y_test, dataset_name="dataset", split_folder="./models"):
    """
    Train a Random Forest classifier to classify missing values in the dataset.
    Test multiple n_estimators values and select the best one based on accuracy.
    """
    try:
        # Ensure the output directory exists
        ensure_directory_exists(f'{split_folder}/RandomForest/{dataset_name}')

        # Test multiple n_estimators values
        n_estimators_values = [3, 5, 10, 20, 30, 40, 50, 100, 150, 200, 250, 500]
        best_n_estimators = None
        best_accuracy = 0
        best_model = None

        # Initialize progress bar for Random Forest
        rf_progress_bar = tqdm(n_estimators_values, desc=f"RF Training ({dataset_name})", unit="n_estimators")

        for n_estimators in rf_progress_bar:
            rf_model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
            rf_model.fit(X_train, y_train)
            accuracy = rf_model.score(X_test, y_test)

            if accuracy > best_accuracy:
                best_n_estimators = n_estimators
                best_accuracy = accuracy
                best_model = rf_model

            # Update progress bar description with the current best n_estimators and accuracy
            rf_progress_bar.set_postfix(best_n_estimators=best_n_estimators, best_accuracy=f"{best_accuracy:.4f}")

        rf_progress_bar.close()  # Close the Random Forest progress bar

        # Save the entire model
        torch.save(best_model, f'{split_folder}/RandomForest/{dataset_name}/RF_n{best_n_estimators}_missing_value.pth')

        zone_labels = np.load('./models/zone_labels.npy', allow_pickle=True)
        y_pred = best_model.predict(X_test)
        cm, report, precision, f1 = calculate_metrics(y_test, y_pred, list(range(len(zone_labels))))
        save_metrics_to_csv(cm, report, zone_labels, f'{split_folder}/RandomForest/{dataset_name}/RF_n{best_n_estimators}_missing_value')
        save_confusion_matrix_plot(cm, zone_labels, f'{split_folder}/RandomForest/{dataset_name}/RF_n{best_n_estimators}_missing_value')

        return best_model, best_accuracy, best_n_estimators, precision, f1
    except ValueError as e:
        if "Input X contains NaN" in str(e):
            print(f"Skipping Random Forest model for dataset {dataset_name} due to NaN values in input.")
            return None, None, None, None, None
        else:
            raise

def plot_training_times(training_times, output_path):
    """
    Plot the training times for all models.
    """
    plt.figure(figsize=(12, 6))
    models = list(training_times.keys())
    times = list(training_times.values())
    plt.bar(models, times, color='skyblue')
    plt.xlabel("Models")
    plt.ylabel("Training Time (seconds)")
    plt.title("Training Times for All Models")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def main():
    # Define train/test splits
    splits = [(i / 100, (100 - i) / 100) for i in range(10, 100, 10)]

    for train_size, test_size in splits:
        split_folder = f'./models/split_{int(train_size * 100)}_{int(test_size * 100)}'
        ensure_directory_exists(split_folder)

        result = {}  # Reset results for the current split
        training_times = {}  # Reset training times for the current split

        # Count the total number of models to train for this split
        total_models = 0
        for dir in os.listdir('./datasets'):
            for file in os.listdir(os.path.join('./datasets', dir)):
                if file.endswith('.csv'):
                    total_models += 4  # NN, KNN, SVM, RF for each dataset
                    dataset_name = os.path.basename(file).split('.')[0]
                    ensure_directory_exists(f'{split_folder}/NN/{dataset_name}')
                    ensure_directory_exists(f'{split_folder}/KNN/{dataset_name}')
                    ensure_directory_exists(f'{split_folder}/SVM/{dataset_name}')
                    ensure_directory_exists(f'{split_folder}/RandomForest/{dataset_name}')

        # Initialize progress bar for this split
        progress_bar = tqdm(total=total_models, desc=f"Training Models (Split {int(train_size * 100)}-{int(test_size * 100)})", unit="model")

        for dir in os.listdir('./datasets'):
            for file in os.listdir(os.path.join('./datasets', dir)):
                if file.endswith('.csv'):
                    data_path = os.path.join('./datasets', dir, file)
                    dataset_name = os.path.basename(data_path).split('.')[0]

                    # Load data with the current split
                    X_train, X_test, y_train, y_test = load_data(data_path, test_size=test_size)

                    # Train NN model
                    start_time = time.time()
                    _, history, test_accuracy, precision, f1 = train_NN_missing_data(X_train, X_test, y_train, y_test, dataset_name=dataset_name, split_folder=split_folder, epochs=20)
                    end_time = time.time()
                    training_times[f'NN_{dataset_name}'] = end_time - start_time
                    train_accuracy = history['train_acc'][-1]
                    val_accuracy = history['val_acc'][-1]
                    result[f'NN_{dataset_name}'] = (train_accuracy, val_accuracy, test_accuracy, precision, f1)
                    progress_bar.update(1)

                    # Train KNN model
                    start_time = time.time()
                    _, test_accuracy, best_k, precision, f1 = train_K_nearest_neighbors(X_train, X_test, y_train, y_test, dataset_name=dataset_name, split_folder=split_folder)
                    end_time = time.time()
                    training_times[f'KNN_k{best_k}_{dataset_name}'] = end_time - start_time
                    result[f'KNN_k{best_k}_{dataset_name}'] = (None, None, test_accuracy, precision, f1)
                    progress_bar.update(1)

                    # Train SVM model
                    start_time = time.time()
                    _, test_accuracy, precision, f1 = train_SVM(X_train, X_test, y_train, y_test, dataset_name=dataset_name, split_folder=split_folder)
                    end_time = time.time()
                    training_times[f'SVM_{dataset_name}'] = end_time - start_time
                    result[f'SVM_{dataset_name}'] = (None, None, test_accuracy, precision, f1)
                    progress_bar.update(1)

                    # Train Random Forest model
                    start_time = time.time()
                    _, test_accuracy, best_n_estimators, precision, f1 = train_RandomForest(X_train, X_test, y_train, y_test, dataset_name=dataset_name, split_folder=split_folder)
                    end_time = time.time()
                    training_times[f'RF_n{best_n_estimators}_{dataset_name}'] = end_time - start_time
                    result[f'RF_n{best_n_estimators}_{dataset_name}'] = (None, None, test_accuracy, precision, f1)
                    progress_bar.update(1)

        progress_bar.close()  # Close the progress bar

        # Save results for this split
        results_df = pd.DataFrame.from_dict(result, orient='index', columns=['Train_Accuracy', 'Validation_Accuracy', 'Test_Accuracy', 'Precision', 'F1_Score'])
        results_df.index.name = 'Model'
        results_df = results_df.sort_index()
        results_df.to_csv(f'{split_folder}/results.csv', index=True)

        # Plot training times for this split
        plot_training_times(training_times, f'{split_folder}/training_times.png')

    print("\nAll splits completed.")

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()
