import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt
import time
import csv

def voxelize(pc):
    # Normalize point cloud data
    point_cloud = (pc - pc.min(axis=0)) / (pc.max(axis=0) - pc.min(axis=0))
    voxel_size = 32
    voxelized = np.zeros((voxel_size, voxel_size, voxel_size))
    point_cloud = ((voxel_size - 1) * point_cloud).astype(int)
    voxelized[point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2]] = 1
    return voxelized

def save_metrics_to_csv(epoch, train_metric, val_metric, filename):
    """
    Save metrics for each epoch to a CSV file.
    If the file does not exist, create it and write the headers.
    """
    file_exists = os.path.isfile(filename)
    start_epoch = get_last_epoch_from_csv(filename)  # Get the latest epoch from the file
    
    with open(filename, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['Epoch', 'Train Metric', 'Validation Metric'])
        writer.writerow([start_epoch + epoch, train_metric, val_metric])


class VoxelDataset(Dataset):
    def __init__(self, file_list, data_directory, subset_size=None):
        self.data_directory = data_directory
        self.file_list = file_list
        if subset_size is not None:
            self.file_list = self.file_list[:subset_size]  # Use only a subset of files

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = os.path.join(self.data_directory, self.file_list[idx])
        data = np.load(file_path, allow_pickle=True)
        part1, part2, label = data

        # Voxelizing the point clouds
        voxel1 = voxelize(part1)
        voxel2 = voxelize(part2)

        # Convert to tensors
        voxel1 = torch.tensor(voxel1, dtype=torch.float32).unsqueeze(0)  # Add channel dimension
        voxel2 = torch.tensor(voxel2, dtype=torch.float32).unsqueeze(0)  # Add channel dimension
        label = torch.tensor(label, dtype=torch.float32)

        return voxel1, voxel2, label

class Encoder3D(nn.Module):
    def __init__(self):
        super(Encoder3D, self).__init__()
        self.conv1 = nn.Conv3d(1, 4, kernel_size=3, stride=1, padding=1) # Tunable: Number of filters (4). More filters may capture more features but are computationally expensive. Make changes in progressions like 2, 4, 8, 16, 32,...
        self.conv2 = nn.Conv3d(4, 8, kernel_size=3, stride=2, padding=1)  # Tunable: Number of filters (8). Increasing may improve feature extraction at the cost of increased computation.
        #self.conv3 = nn.Conv3d(16, 32, kernel_size=3, stride=2, padding=1)
        self.fc = nn.Linear(8 * 16 * 16 * 16, 32)  # Tunable: Output features (32). Increasing this can capture more complex relationships but increases model complexity.

        self.first_forward = True  # Flag to track the first forward pass
    def forward(self, x):
        x = F.relu(self.conv1(x))
        if self.first_forward:
            print(f"Shape after conv1: {x.shape}")
        x = F.relu(self.conv2(x))
        if self.first_forward:
            print(f"Shape after conv2: {x.shape}")
        x = x.view(x.size(0), -1)  # Flatten for the fully connected layer
        if self.first_forward:
            print(f"Shape after flattening: {x.shape}")
        x = self.fc(x)
        if self.first_forward:
            print(f"Shape after fc: {x.shape}")
            self.first_forward = False  # Disable further shape printing after the first forward pass
        return x

class Siamese3DCNN(nn.Module):
    def __init__(self):
        super(Siamese3DCNN, self).__init__()
        self.encoder = Encoder3D()
        self.fc_out = nn.Linear(32, 1)  # Tunable: Number of input features (32). Match this to the output of the encoder.


    def forward(self, voxel1, voxel2):
        encoded1 = self.encoder(voxel1)
        encoded2 = self.encoder(voxel2)

        # Compute the L1 distance between the two encoded vectors
        combined = torch.abs(encoded1 - encoded2)

        # Pass through fully connected layer to get a single value
        output = self.fc_out(combined)

        # Apply sigmoid to produce a probability output
        output = torch.sigmoid(output)

        return output



def train_model_with_validation(num_epochs, learning_rate, batch_size, validate_every_n_epochs):

    data_directory = "/Users/yvonne/Downloads/curated_data"
    # Define data directory
    #  data_directory = os.path.join(os.getcwd(), "curated_data/")

    # Get all files in the directory
    all_files = [f for f in os.listdir(data_directory) if f.endswith('.npy')]

    # Split data into train, validation, and test sets
    # Tunable:
    # train_test_split `test_size`: Controls the proportion of data used for validation and test sets.
    # A smaller test_size keeps more data for training, which may improve model learning at the cost of less validation data.
    train_files, test_files = train_test_split(all_files, test_size=0.3, random_state=42)
    val_files, test_files = train_test_split(test_files, test_size=0.5, random_state=42)  # Further split to validation and test

    # Create Dataset instances
    # Corrected to pass the list of filenames along with the directory path
    train_dataset = VoxelDataset(train_files, data_directory)  # Pass list of training files and the directory path
    val_dataset = VoxelDataset(val_files, data_directory)      # Pass list of validation files and the directory path
    test_dataset = VoxelDataset(test_files, data_directory)    # Pass list of test files and the directory path

    # Create DataLoaders for each dataset
    batch_size = 8 # Tunable parameter: Increasing batch size can improve efficiency, but requires more memory.
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


    # model = Siamese3DCNN()
    #  criterion = nn.BCELoss() # Tunable: Loss function. Experimenting with other losses (e.g., MSELoss) could impact model behavior.
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate) # Tunable: Trying other optimizers (e.g., SGD) can change convergence behavior.

    # Set device to GPU if available, otherwise CPU

    # Move model to the GPU
    model = Siamese3DCNN()
    criterion = nn.BCELoss()  # Loss function
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # Optimizer

    # Check if model exists
    model_save_path = "3d_cnn_model.pth"
    if os.path.exists(model_save_path):
        print(f"Model exists. Loading model from {model_save_path}")
        model.load_state_dict(torch.load(model_save_path))
    else:
        print("No existing model found. Training from scratch.")



    # Lists to store losses for graphing
    training_losses = []
    validation_losses = []

    training_accuracies = []
    training_precisions = []
    training_recalls = []

    validation_accuracies = []
    validation_precisions = []
    validation_recalls = []

    # Updated DataLoader with increased batch size and more workers for data loading
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of validation batches: {len(val_loader)}")

    for epoch in range(num_epochs):
        # Training Phase
        model.train()
        train_loss = 0.0
        start_time = time.time()
        train_outputs_list = []
        train_labels_list = []


        voxel_ct = 0
        for voxel1, voxel2, labels in train_loader:

            optimizer.zero_grad()
            outputs = model(voxel1, voxel2)
            labels = labels.view(-1, 1)  # Ensure labels match output shape
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            voxel_ct += 1

            if voxel_ct %100 == 0:
                print(f"Voxel finished:{voxel_ct}")
            
            train_outputs_list.extend(outputs.detach().cpu().numpy())
            train_labels_list.extend(labels.detach().cpu().numpy())

        avg_train_loss = train_loss / len(train_loader)
        training_losses.append(avg_train_loss)
        end_time = time.time()

        # Convert to binary predictions for accuracy, precision, recall calculations
        train_outputs_list = [1 if output >= 0.5 else 0 for output in train_outputs_list]
        train_labels_list = [int(label) for label in train_labels_list]

        train_accuracy = accuracy_score(train_labels_list, train_outputs_list)
        train_precision = precision_score(train_labels_list, train_outputs_list)
        train_recall = recall_score(train_labels_list, train_outputs_list)

        # Append metrics to lists
        training_accuracies.append(train_accuracy)
        training_precisions.append(train_precision)
        training_recalls.append(train_recall)

        print(f"\nEpoch [{epoch + 1}/{num_epochs}], Training Loss: {avg_train_loss:.6f}, "
              f"Accuracy: {train_accuracy:.4f}, Precision: {train_precision:.4f}, Recall: {train_recall:.4f}, "
              f"Time Taken: {end_time - start_time:.2f}s")

        # Validation Phase (every N epochs)
        val_outputs_list = []
        val_labels_list = []
        if (epoch + 1) % validate_every_n_epochs == 0: # Tunable: Frequency of validation checks
            model.eval()
            val_loss = 0.0
            start_time = time.time()

            with torch.no_grad():
                for voxel1, voxel2, labels in val_loader:
                    outputs = model(voxel1, voxel2)
                    labels = labels.view(-1, 1)  # Ensure labels match output shape
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()

                    val_outputs_list.extend(outputs.cpu().numpy())
                    val_labels_list.extend(labels.cpu().numpy())

            avg_val_loss = val_loss / len(val_loader)
            validation_losses.append(avg_val_loss)
            end_time = time.time()

            # Convert to binary predictions for accuracy, precision, recall calculations
            val_outputs_list = [1 if output >= 0.5 else 0 for output in val_outputs_list]
            val_labels_list = [int(label) for label in val_labels_list]

            val_accuracy = accuracy_score(val_labels_list, val_outputs_list)
            val_precision = precision_score(val_labels_list, val_outputs_list)
            val_recall = recall_score(val_labels_list, val_outputs_list)

            # Append validation metrics to lists
            validation_accuracies.append(val_accuracy)
            validation_precisions.append(val_precision)
            validation_recalls.append(val_recall)
            print(f"Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {avg_val_loss:.6f}, "
                  f"Accuracy: {val_accuracy:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, "
                  f"Validation Time Taken: {end_time - start_time:.2f}s")
        
        # Save metrics to CSV files after each epoch
        save_metrics_to_csv(epoch + 1, train_accuracy, val_accuracy, 'accuracy.csv')
        save_metrics_to_csv(epoch + 1, train_precision, val_precision, 'precision.csv')
        save_metrics_to_csv(epoch + 1, train_recall, val_recall, 'recall.csv')

    # Save the trained model
    model_save_path = "3d_cnn_model.pth"  # Filepath where model will be saved
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

    return model, training_losses, validation_losses



def read_metrics_from_csv(filename):
    """
    Reads metrics from a CSV file and returns a list of epochs and corresponding metrics.
    """
    epochs = []
    train_metrics = []
    val_metrics = []

    with open(filename, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip the header
        for row in reader:
            epochs.append(int(row[0]))
            train_metrics.append(float(row[1]))
            val_metrics.append(float(row[2]))

    return epochs, train_metrics, val_metrics

def plot_metrics(epochs, train_metrics, val_metrics, metric_name):
    """
    Plots the training and validation metrics.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_metrics, label=f'Train {metric_name}', marker='o')
    plt.plot(epochs, val_metrics, label=f'Validation {metric_name}', marker='o')
    plt.xlabel('Epochs')
    plt.ylabel(metric_name)
    plt.title(f'{metric_name} Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_all_metrics():
    # Read and plot accuracy
    accuracy_epochs, train_accuracy, val_accuracy = read_metrics_from_csv('accuracy.csv')
    plot_metrics(accuracy_epochs, train_accuracy, val_accuracy, 'Accuracy')

    # Read and plot precision
    precision_epochs, train_precision, val_precision = read_metrics_from_csv('precision.csv')
    plot_metrics(precision_epochs, train_precision, val_precision, 'Precision')

    # Read and plot recall
    recall_epochs, train_recall, val_recall = read_metrics_from_csv('recall.csv')
    plot_metrics(recall_epochs, train_recall, val_recall, 'Recall')


def main():


    # Tunable parameters:
        # `num_epochs`: Increasing number of epochs gives the model more opportunities to learn but risks overfitting.
        # `learning_rate`: A smaller learning rate (0.001) helps the model converge slowly and stably. Higher rates (e.g., 0.01) may lead to faster but unstable convergence.
        # `batch_size`: Larger batch sizes help with convergence and are faster but need more memory.
        # `validate_every_n_epochs`: Controls how often to validate. Less frequent validation speeds up training.
    # Run the training with updated parameters
    #model, training_losses, validation_losses = train_model_with_validation(num_epochs=3, learning_rate=0.002, batch_size=128, validate_every_n_epochs=1)


    plot_all_metrics()
if __name__ == "__main__":
    main()