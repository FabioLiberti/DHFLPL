import pandas as pd
from collections import OrderedDict
from typing import List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
import os
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, ConfusionMatrixDisplay
from sklearn.tree import DecisionTreeClassifier, plot_tree
from datasets.utils.logging import disable_progress_bar
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10
import flwr as fl
from flwr.common import Metrics
from flwr_datasets import FederatedDataset
from colorama import Fore, Style, init
from torchvision.models import ResNet18_Weights

# Initialize colorama
init(autoreset=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

file_name = os.path.basename(__file__)
print(f"**************************************")
print("Name of the file: " + Fore.RED + f"{file_name}")
print(f"**************************************")

print(f"Training on {DEVICE} using PyTorch {torch.__version__} and Flower {fl.__version__}")

disable_progress_bar()


############################################
# Step 0: Loading Data
############################################

# Loading the data
print(Fore.RED + "*************************")
print(Fore.RED + "* STEP(0): Loading Data *")
print(Fore.RED + "*************************")
print(Fore.RED + "* 0-Loading data")

NUM_CLIENTS = 10
BATCH_SIZE = 32

def load_datasets():
    fds = FederatedDataset(dataset="cifar10", partitioners={"train": NUM_CLIENTS})

    def apply_transforms(batch):
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        batch["img"] = [transform(img) for img in batch["img"]]
        return batch

    trainloaders = []
    valloaders = []
    for partition_id in range(NUM_CLIENTS):
        partition = fds.load_partition(partition_id, "train")
        partition = partition.with_transform(apply_transforms)
        # Introduce heterogeneity by modifying the train/test split ratio
        train_size = 0.8 - 0.05 * (partition_id % 2)  # Alternating 80% and 75%
        partition = partition.train_test_split(train_size=train_size, seed=42)
        trainloaders.append(DataLoader(partition["train"], batch_size=BATCH_SIZE))
        valloaders.append(DataLoader(partition["test"], batch_size=BATCH_SIZE))
    testset = fds.load_split("test").with_transform(apply_transforms)
    testloader = DataLoader(testset, batch_size=BATCH_SIZE)
    return trainloaders, valloaders, testloader

trainloaders, valloaders, testloader = load_datasets()

# first batch of images and labels in the first training set
print(Fore.RED + "* 0.0-First batch of images and labels in the first training set")

batch = next(iter(trainloaders[0]))
images, labels = batch["img"], batch["label"]
images = images.permute(0, 2, 3, 1).numpy()
images = images / 2 + 0.5

fig, axs = plt.subplots(4, 8, figsize=(12, 6))

for i, ax in enumerate(axs.flat):
    ax.imshow(images[i])
    ax.set_title(trainloaders[0].dataset.features["label"].int2str([labels[i]])[0])
    ax.axis("off")

fig.tight_layout()
plt.show()


############################################
# Step 1: Centralized Training with PyTorch
# Defining the model
############################################

print(Fore.RED + "**********************************************")
print(Fore.RED + "* STEP(1): Centralized Training with PyTorch *")
print(Fore.RED + "**********************************************")


print(Fore.RED + "* 1.0-Defining the model")


class Net(nn.Module):
    def __init__(self) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 10)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def train(net, trainloader, epochs: int, verbose=False):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    net.train()
    train_losses = []
    train_accuracies = []
    for epoch in range(epochs):
        correct, total, epoch_loss = 0, 0, 0.0
        for batch in trainloader:
            images, labels = batch["img"].to(DEVICE), batch["label"].to(DEVICE, non_blocking=True)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.detach().item()
            total += labels.size(0)
            correct += (torch.max(outputs.detach().data, 1)[1] == labels).sum().item()
        epoch_loss /= len(trainloader.dataset)
        epoch_acc = correct / total
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)
        scheduler.step(epoch_loss)
        if verbose:
            print(f"Epoch {epoch+1}: train loss {epoch_loss}, accuracy {epoch_acc}")
    return train_losses, train_accuracies

def test(net, testloader):
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    all_labels = []
    all_preds = []
    net.eval()
    with torch.no_grad():
        for batch in testloader:
            images, labels = batch["img"].to(DEVICE), batch["label"].to(DEVICE, non_blocking=True)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
    loss /= len(testloader.dataset)
    accuracy = correct / total
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=1)
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    return loss, accuracy, precision, recall, f1, all_labels, all_preds

trainloader = trainloaders[0]
valloader = valloaders[0]
net = Net().to(DEVICE)

train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []
val_precisions = []
val_recalls = []
val_f1s = []


# early stopping
print(Fore.RED + "* 1.0.1-Early Stopping")

early_stopping_patience = 5
best_val_loss = float('inf')
early_stopping_counter = 0

print("Early Stopping Patience:" + Fore.RED + f"{early_stopping_patience}")
print("Best Validation Loss:" + Fore.RED + f"{best_val_loss}")
print("Early Stopping Counter:"  + Fore.RED + f"{early_stopping_counter}")


print(Fore.RED + "* 1.1-Start Training (con earl stopping)")


for epoch in range(6):  # Increased number of epochs for demonstration
    train_loss, train_acc = train(net, trainloader, 1)
    train_losses.extend(train_loss)
    train_accuracies.extend(train_acc)
    loss, accuracy, precision, recall, f1, all_labels, all_preds = test(net, valloader)
    val_losses.append(loss)
    val_accuracies.append(accuracy)
    val_precisions.append(precision)
    val_recalls.append(recall)
    val_f1s.append(f1)

    # loss, accuracy, precision, recall, f1, all_labels, all_preds = test(net, testloader)
    print(f"Epoch {epoch+1}: validation loss {loss}, accuracy {accuracy}, precision {precision}, recall {recall}, F1 {f1}")

    # Controllo early stopping
    if loss < best_val_loss:
        best_val_loss = loss
        early_stopping_counter = 0
    else:
        early_stopping_counter += 1

    if early_stopping_counter >= early_stopping_patience:
        print("Early stopping triggered")
        break




# Plotting training and validation metrics
print(Fore.RED + "* 1.2-Plotting training and validation metrics (1)")

plt.figure(figsize=(20, 5))
plt.subplot(1, 3, 1)
plt.plot(train_losses, label='Training Loss')
plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')

plt.subplot(1, 3, 2)
plt.plot(train_accuracies, label='Training Accuracy')
plt.plot(range(1, len(val_accuracies) + 1), val_accuracies, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')

plt.subplot(1, 3, 3)
plt.plot(range(1, len(val_precisions) + 1), val_precisions, label='Validation Precision')
plt.plot(range(1, len(val_recalls) + 1), val_recalls, label='Validation Recall')
plt.plot(range(1, len(val_f1s) + 1), val_f1s, label='Validation F1')
plt.xlabel('Epoch')
plt.ylabel('Metrics')
plt.legend()
plt.title('Validation Precision, Recall and F1-score')

plt.tight_layout()
plt.show()

# Plotting confusion matrix
print(Fore.RED + "* 1.3-Plotting confusion matrix (2)")

cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=range(10))
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()

# Plotting decision tree (note: this is just for visualization and does not impact model training)
# For demonstration, we'll train a simple decision tree on a subset of the data
print(Fore.RED + "* 1.4-Plotting decision tree (3)")


X_train_list = []
y_train_list = []

for batch in trainloaders[0]:
    images, labels = batch["img"], batch["label"]
    X_train_list.append(images.reshape(images.shape[0], -1).numpy())
    y_train_list.append(labels.numpy())

X_train = np.vstack(X_train_list)
y_train = np.hstack(y_train_list)

clf = DecisionTreeClassifier(max_depth=3)
clf.fit(X_train, y_train)

plt.figure(figsize=(20, 10))
plot_tree(clf, filled=True, class_names=[str(i) for i in range(10)])
plt.title("Decision Tree Visualization")
plt.show()

#########################################
# Step 2: Federated Learning with Flower
#########################################
# Updating model parameters

print(Fore.RED + "*******************************************")
print(Fore.RED + "* STEP(2): Federated Learning with Flower *")
print(Fore.RED + "*******************************************")
print(Fore.RED + "* 2.1-Updating model parameters")



def set_parameters(net, parameters: List[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)

def get_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

# Implementing a Flower client
print(Fore.RED + "* 2.2-Implementing a Flower client")


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, net, trainloader, valloader):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []
        self.val_precisions = []
        self.val_recalls = []
        self.val_f1s = []

    def get_parameters(self, config):
        return get_parameters(self.net)

    def fit(self, parameters, config):
        set_parameters(self.net, parameters)
        train_loss, train_acc = train(self.net, self.trainloader, epochs=1)
        self.train_losses.extend(train_loss)
        self.train_accuracies.extend(train_acc)
        loss, accuracy, precision, recall, f1, _, _ = test(self.net, self.valloader)
        self.val_losses.append(loss)
        self.val_accuracies.append(accuracy)
        self.val_precisions.append(precision)
        self.val_recalls.append(recall)
        self.val_f1s.append(f1)
        return get_parameters(self.net), len(self.trainloader.dataset), {"loss": loss, "accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

    def evaluate(self, parameters, config):
        set_parameters(self.net, parameters)
        loss, accuracy, precision, recall, f1, _, _ = test(self.net, self.valloader)
        return float(loss), len(self.valloader.dataset), {"accuracy": float(accuracy), "precision": float(precision), "recall": float(recall), "f1": float(f1)}

def client_fn(cid: str) -> fl.client.Client:
    net = Net().to(DEVICE)
    trainloader = trainloaders[int(cid)]
    valloader = valloaders[int(cid)]
    return FlowerClient(net, trainloader, valloader).to_client()

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    precisions = [num_examples * m["precision"] for num_examples, m in metrics]
    recalls = [num_examples * m["recall"] for num_examples, m in metrics]
    f1s = [num_examples * m["f1"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {
        "accuracy": sum(accuracies) / sum(examples),
        "precision": sum(precisions) / sum(examples),
        "recall": sum(recalls) / sum(examples),
        "f1": sum(f1s) / sum(examples)
    }

# Defining the FL Strategy
print(Fore.RED + "* 2.3-Defining the FL Strategy")


strategy = fl.server.strategy.FedAvg(
    fraction_fit=1.0,
    fraction_evaluate=0.5,
    min_fit_clients=10,
    min_evaluate_clients=5,
    min_available_clients=10,
    evaluate_metrics_aggregation_fn=weighted_average,
)

# Definizione dei dati della strategia
strategy_data = {
    "Parameter": [
        "Fraction Fit",
        "Fraction Evaluate",
        "Min Fit Clients",
        "Min Evaluate Clients",
        "Min Available Clients",
        "Evaluate Metrics Aggregation Function"
    ],
    "Value": [
        1.0,
        0.5,
        10,
        5,
        10,
        "Weighted Average"
    ]
}

# Creazione del DataFrame
strategy_df = pd.DataFrame(strategy_data)

# Stampa del DataFrame
print(strategy_df)

# Visualizzazione della tabella come grafico

print(Fore.RED + "* 2.3.1-Plotting the FL Strategy (4)")

fig, ax = plt.subplots(figsize=(15, 8))
ax.axis('tight')
ax.axis('off')

table = ax.table(cellText=strategy_df.values, colLabels=strategy_df.columns, cellLoc='center', loc='center', edges='closed')
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1.2, 1.2)

# Enhancing the table
table.auto_set_column_width(col=list(range(len(strategy_df.columns))))  # Adjust column width
for (i, j), cell in table.get_celld().items():
    cell.set_edgecolor('gray')  # Set cell border color
    cell.set_linewidth(1.5)  # Set cell border width
    if i == 0:  # Header row
        cell.set_text_props(weight='bold', color='white')
        cell.set_facecolor('black')  # Header background color
    else:
        cell.set_facecolor('white' if i % 2 == 0 else 'lightgray')  # Alternating row colors

# Set title with enhanced properties
plt.title("Federated Learning Strategy Parameters", fontsize=16, fontweight='bold', pad=20)

plt.show()


##################################
print(Fore.RED + "* 2.4-Start Simulation")


client_resources = {"num_cpus": 1, "num_gpus": 0.0}
if DEVICE.type == "cuda":
    client_resources = {"num_cpus": 1, "num_gpus": 1.0}

hist = fl.simulation.start_simulation(
    client_fn=client_fn,
    num_clients=NUM_CLIENTS,
    config=fl.server.ServerConfig(num_rounds=5),
    strategy=strategy,
    client_resources=client_resources,
)

# Debug: Print available keys in metrics
print(Fore.RED + "* 2.5-Debug: Print available keys in metrics")

print("Available keys in metrics:", hist.metrics_centralized.keys())

# Collecting individual client metrics for plotting
print(Fore.RED + "* 2.5.1-Collecting individual client metrics for plotting")

client_train_losses = []
client_train_accuracies = []
client_val_losses = []
client_val_accuracies = []
client_val_precisions = []
client_val_recalls = []
client_val_f1s = []
client_resources_allocation = []

for client_id in range(NUM_CLIENTS):
    client_net = Net().to(DEVICE)
    client = FlowerClient(client_net, trainloaders[client_id], valloaders[client_id])
    client.fit(get_parameters(client_net), {})
    client_train_losses.append(client.train_losses)
    client_train_accuracies.append(client.train_accuracies)
    client_val_losses.append(client.val_losses)
    client_val_accuracies.append(client.val_accuracies)
    client_val_precisions.append(client.val_precisions)
    client_val_recalls.append(client.val_recalls)
    client_val_f1s.append(client.val_f1s)
    # Simulate resource allocation
    cpu = 1
    gpu = 0.1 * (client_id % 2)  # Alternating between 0 and 0.1 GPU
    client_resources_allocation.append((cpu, gpu))

# Debug: Print the collected metrics for each client
print(Fore.RED + "* 2.5.2-Debug: Print the collected metrics for each client")


for i in range(NUM_CLIENTS):
    print(f"Client {i} Training Losses: {client_train_losses[i]}")
    print(f"Client {i} Training Accuracies: {client_train_accuracies[i]}")
    print(f"Client {i} Validation Losses: {client_val_losses[i]}")
    print(f"Client {i} Validation Accuracies: {client_val_accuracies[i]}")
    print(f"Client {i} Validation Precisions: {client_val_precisions[i]}")
    print(f"Client {i} Validation Recalls: {client_val_recalls[i]}")
    print(f"Client {i} Validation F1 Scores: {client_val_f1s[i]}")
    print(f"Client {i} CPU: {client_resources_allocation[i][0]}, GPU: {client_resources_allocation[i][1]}")

# Create a DataFrame to display the results in a table
print(Fore.RED + "* 2.5.3-Create a DataFrame to display the results in a table")


data = {
    "Client": list(range(NUM_CLIENTS)),
    "Training Loss": [losses[-1] for losses in client_train_losses],
    "Training Accuracy": [f"{accuracies[-1]:.4f}" for accuracies in client_train_accuracies],
    "Validation Loss": [losses[-1] for losses in client_val_losses],
    "Validation Accuracy": [f"{accuracies[-1]:.4f}" for accuracies in client_val_accuracies],
    "Validation Precision": [f"{precisions[-1]:.4f}" for precisions in client_val_precisions],
    "Validation Recall": [f"{recalls[-1]:.4f}" for recalls in client_val_recalls],
    "Validation F1": [f"{f1s[-1]:.4f}" for f1s in client_val_f1s],
    "CPU": [cpu for cpu, gpu in client_resources_allocation],
    "GPU": [gpu for cpu, gpu in client_resources_allocation]
}

df = pd.DataFrame(data)
print("\nFederated Learning Client Results:\n")
print(df)


# Plotting resource allocation
print(Fore.RED + "* 2.6-Plotting resource allocation (5)")


resource_data = {
    "Client": list(range(NUM_CLIENTS)),
    "CPU": [cpu for cpu, gpu in client_resources_allocation],
    "GPU": [gpu for cpu, gpu in client_resources_allocation]
}

resource_df = pd.DataFrame(resource_data)

# Plotting the table in a graph
print(Fore.RED + "* 2.6.1-Plotting the table in a graph")

fig, ax = plt.subplots(figsize=(18, 8)) # Increase the figure size for better readability
ax.axis('tight')
ax.axis('off')
table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center', edges='closed')
table.auto_set_font_size(False)
table.set_fontsize(10)   # Increase the font size for better readability
table.scale(1.5, 1.5)    # Increase the scale for better readability

# Imposta la larghezza delle colonne
cell_dict = table.get_celld()
for (row, col), cell in cell_dict.items():
    if col == 0:  # Colonna "Client"
        cell.set_width(0.05)
    elif col in [6, 7, 8, 9]:  # Colonne "Validation Recall", "Validation F1", "CPU", "GPU"
        cell.set_width(0.08)
    else:
        cell.set_width(0.1)

# Set font properties for the headers
for (i, j), cell in table.get_celld().items():
    if i == 0:
        cell.set_text_props(fontproperties={'weight': 'bold', 'size': 10})  # Bold and larger font for headers

# plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
plt.title("Federated Learning Client Results", fontsize=16)
plt.show()


#################################################################################################################

# Resource Allocation per Client
print(Fore.RED + "* 2.6-Plotting resource allocation (5.1)")

cpu_allocations = [cpu for cpu, gpu in client_resources_allocation]
gpu_allocations = [gpu for cpu, gpu in client_resources_allocation]

fig, axs = plt.subplots(2, 1, figsize=(12, 10))  # 2 righe, 1 colonna
axs[0].bar(range(NUM_CLIENTS), cpu_allocations, color='blue', label='CPU')
axs[0].set_ylabel('CPU Allocation')
axs[0].set_title('CPU Allocation per Client')
axs[0].legend()

axs[1].bar(range(NUM_CLIENTS), gpu_allocations, color='orange', label='GPU')
axs[1].set_ylabel('GPU Allocation')
axs[1].set_title('GPU Allocation per Client')
axs[1].legend()

fig.tight_layout()
plt.show()


##################################################################################################################
print(Fore.RED + "* 2.7-Comparison Metrics")

fig, axs = plt.subplots(4, 2, figsize=(20, 20))

# Training Loss per client
print(Fore.RED + "* 2.7.1-Training Loss per client")


for i, losses in enumerate(client_train_losses):
    axs[0, 0].plot(losses, label=f'Client {i}', marker='o')
axs[0, 0].set_xlabel('Epoch')
axs[0, 0].set_ylabel('Training Loss')
axs[0, 0].set_yscale('log')
axs[0, 0].set_title('Training Loss per Client')
axs[0, 0].legend()

# Training Accuracy per client
print(Fore.RED + "* 2.7.2-Training Accuracy per client")

for i, accuracies in enumerate(client_train_accuracies):
    axs[0, 1].plot(accuracies, label=f'Client {i}', marker='o')
axs[0, 1].set_xlabel('Epoch')
axs[0, 1].set_ylabel('Training Accuracy')
axs[0, 1].set_yscale('log')
axs[0, 1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.4f}'))
axs[0, 1].set_title('Training Accuracy per Client')
axs[0, 1].legend()

# Validation Loss per client
print(Fore.RED + "* 2.7.3-Validation Loss per client")


for i, losses in enumerate(client_val_losses):
    axs[1, 0].plot(losses, label=f'Client {i}', marker='o')
axs[1, 0].set_xlabel('Epoch')
axs[1, 0].set_ylabel('Validation Loss')
axs[1, 0].set_yscale('log')
axs[1, 0].set_title('Validation Loss per Client')
axs[1, 0].legend()

# Validation Accuracy per client
print(Fore.RED + "* 2.7.4-Validation Accuracy per client")

for i, accuracies in enumerate(client_val_accuracies):
    axs[1, 1].plot(accuracies, label=f'Client {i}', marker='o')
axs[1, 1].set_xlabel('Epoch')
axs[1, 1].set_ylabel('Validation Accuracy')
axs[1, 1].set_yscale('log')
axs[1, 1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.4f}'))
axs[1, 1].set_title('Validation Accuracy per Client')
axs[1, 1].legend()

# Validation Precision per client
print(Fore.RED + "* 2.7.5-Validation Precision per client")


for i, precisions in enumerate(client_val_precisions):
    axs[2, 0].plot(precisions, label=f'Client {i}', marker='o')
axs[2, 0].set_xlabel('Epoch')
axs[2, 0].set_ylabel('Validation Precision')
axs[2, 0].set_yscale('log')
axs[2, 0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.4f}'))
axs[2, 0].set_title('Validation Precision per Client')
axs[2, 0].legend()

# Validation Recall per client
print(Fore.RED + "* 2.7.6-Validation Recall per client")


for i, recalls in enumerate(client_val_recalls):
    axs[2, 1].plot(recalls, label=f'Client {i}', marker='o')
axs[2, 1].set_xlabel('Epoch')
axs[2, 1].set_ylabel('Validation Recall')
axs[2, 1].set_yscale('log')
axs[2, 1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.4f}'))
axs[2, 1].set_title('Validation Recall per Client')
axs[2, 1].legend()

# Validation F1 per client
print(Fore.RED + "* 2.7.7-Validation F1 per client")


for i, f1s in enumerate(client_val_f1s):
    axs[3, 0].plot(f1s, label=f'Client {i}', marker='o')
axs[3, 0].set_xlabel('Epoch')
axs[3, 0].set_ylabel('Validation F1')
axs[3, 0].set_yscale('log')
axs[3, 0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.4f}'))
axs[3, 0].set_title('Validation F1 per Client')
axs[3, 0].legend()

# Rimuovi l'ultimo subplot vuoto
fig.delaxes(axs[3, 1])

fig.tight_layout()
plt.show()



#######################################################
print(Fore.RED + "* 2.8-Comparison TEST of Centralized and Federated Learning")

# Comparison TEST of Centralized and Federated Learning
# Extract final test loss and accuracy for centralized learning
centralized_test_loss, centralized_test_accuracy, centralized_test_precision, centralized_test_recall, centralized_test_f1, _, _ = test(net, testloader)

# Extract final test loss and accuracy for federated learning
federated_test_loss = np.mean([client_val_losses[i][-1] for i in range(NUM_CLIENTS)])
federated_test_accuracy = np.mean([client_val_accuracies[i][-1] for i in range(NUM_CLIENTS)])
federated_test_precision = np.mean([client_val_precisions[i][-1] for i in range(NUM_CLIENTS)])
federated_test_recall = np.mean([client_val_recalls[i][-1] for i in range(NUM_CLIENTS)])
federated_test_f1 = np.mean([client_val_f1s[i][-1] for i in range(NUM_CLIENTS)])

# Plotting comparison
labels = ['Centralized', 'Federated']
test_losses = [centralized_test_loss, federated_test_loss]
test_accuracies = [centralized_test_accuracy, federated_test_accuracy]
test_precisions = [centralized_test_precision, federated_test_precision]
test_recalls = [centralized_test_recall, federated_test_recall]
test_f1s = [centralized_test_f1, federated_test_f1]

# Additional plot for better comparison
fig, ax = plt.subplots(1, 5, figsize=(25, 5))

# Test Loss comparison
ax[0].plot(labels, test_losses, color='blue', marker='o')
ax[0].set_yscale('log')
ax[0].set_ylabel('Test Loss')
ax[0].set_title('Test Loss Comparison')

# Test Accuracy comparison
ax[1].plot(labels, test_accuracies, color='blue', marker='o')
ax[1].set_yscale('log')
ax[1].set_ylabel('Test Accuracy')
ax[1].set_title('Test Accuracy Comparison')

# Test Precision comparison
ax[2].plot(labels, test_precisions, color='blue', marker='o')
ax[2].set_yscale('log')
ax[2].set_ylabel('Test Precision')
ax[2].set_title('Test Precision Comparison')

# Test Recall comparison
ax[3].plot(labels, test_recalls, color='blue', marker='o')
ax[3].set_yscale('log')
ax[3].set_ylabel('Test Recall')
ax[3].set_title('Test Recall Comparison')

# Test F1 comparison
ax[4].plot(labels, test_f1s, color='blue', marker='o')
ax[4].set_yscale('log')
ax[4].set_ylabel('Test F1')
ax[4].set_title('Test F1 Comparison')

for axis in ax:
    axis.set_xlabel('Learning Type')
    axis.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.4f}'))

fig.tight_layout()
plt.show()

################################################################################
print(Fore.RED + "* 2.9-Distribution of Training Metrics")


# Estrazione delle metriche dell'ultima epoca per ogni client
train_losses_last_epoch = [losses[-1] for losses in client_train_losses]
train_accuracies_last_epoch = [accuracies[-1] for accuracies in client_train_accuracies]
val_losses_last_epoch = [losses[-1] for losses in client_val_losses]
val_accuracies_last_epoch = [accuracies[-1] for accuracies in client_val_accuracies]
val_precisions_last_epoch = [precisions[-1] for precisions in client_val_precisions]
val_recalls_last_epoch = [recalls[-1] for recalls in client_val_recalls]
val_f1s_last_epoch = [f1s[-1] for f1s in client_val_f1s]

fig, axs = plt.subplots(3, 3, figsize=(20, 15))

# Distribuzione delle Perdite di Addestramento tra i Client
axs[0, 0].hist(train_losses_last_epoch, bins=10, edgecolor='black')
axs[0, 0].set_xlabel('Training Loss')
axs[0, 0].set_ylabel('Number of Clients')
axs[0, 0].set_title('Distribution of Training Losses among Clients')

# Distribuzione delle Accuratezze di Addestramento tra i Client
axs[0, 1].hist(train_accuracies_last_epoch, bins=10, edgecolor='black')
axs[0, 1].set_xlabel('Training Accuracy')
axs[0, 1].set_ylabel('Number of Clients')
axs[0, 1].set_title('Distribution of Training Accuracies among Clients')

# Distribuzione delle Perdite di Validazione tra i Client
axs[0, 2].hist(val_losses_last_epoch, bins=10, edgecolor='black')
axs[0, 2].set_xlabel('Validation Loss')
axs[0, 2].set_ylabel('Number of Clients')
axs[0, 2].set_title('Distribution of Validation Losses among Clients')

# Distribuzione delle Accuratezze di Validazione tra i Client
axs[1, 0].hist(val_accuracies_last_epoch, bins=10, edgecolor='black')
axs[1, 0].set_xlabel('Validation Accuracy')
axs[1, 0].set_ylabel('Number of Clients')
axs[1, 0].set_title('Distribution of Validation Accuracies among Clients')

# Distribuzione delle Precisioni di Validazione tra i Client
axs[1, 1].hist(val_precisions_last_epoch, bins=10, edgecolor='black')
axs[1, 1].set_xlabel('Validation Precision')
axs[1, 1].set_ylabel('Number of Clients')
axs[1, 1].set_title('Distribution of Validation Precisions among Clients')

# Distribuzione dei Recall di Validazione tra i Client
axs[1, 2].hist(val_recalls_last_epoch, bins=10, edgecolor='black')
axs[1, 2].set_xlabel('Validation Recall')
axs[1, 2].set_ylabel('Number of Clients')
axs[1, 2].set_title('Distribution of Validation Recalls among Clients')

# Distribuzione dei F1-score di Validazione tra i Client
axs[2, 0].hist(val_f1s_last_epoch, bins=10, edgecolor='black')
axs[2, 0].set_xlabel('Validation F1-score')
axs[2, 0].set_ylabel('Number of Clients')
axs[2, 0].set_title('Distribution of Validation F1-scores among Clients')

fig.tight_layout()
plt.show()


# Grafico completo finale
print(Fore.RED + "* 2.11-Grafico completo finale")

fig, axs = plt.subplots(7, 2, figsize=(20, 35))

# Training Loss per Client
for i, losses in enumerate(client_train_losses):
    axs[0, 0].plot(losses, label=f'Client {i}', marker='o')
axs[0, 0].set_xlabel('Epoch')
axs[0, 0].set_ylabel('Training Loss')
axs[0, 0].set_yscale('log')
axs[0, 0].set_title('Training Loss per Client')
axs[0, 0].legend()

# Training Accuracy per Client
for i, accuracies in enumerate(client_train_accuracies):
    axs[0, 1].plot(accuracies, label=f'Client {i}', marker='o')
axs[0, 1].set_xlabel('Epoch')
axs[0, 1].set_ylabel('Training Accuracy')
axs[0, 1].set_yscale('log')
axs[0, 1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.4f}'))
axs[0, 1].set_title('Training Accuracy per Client')
axs[0, 1].legend()

# Validation Loss per Client
for i, losses in enumerate(client_val_losses):
    axs[1, 0].plot(losses, label=f'Client {i}', marker='o')
axs[1, 0].set_xlabel('Epoch')
axs[1, 0].set_ylabel('Validation Loss')
axs[1, 0].set_yscale('log')
axs[1, 0].set_title('Validation Loss per Client')
axs[1, 0].legend()

# Validation Accuracy per Client
for i, accuracies in enumerate(client_val_accuracies):
    axs[1, 1].plot(accuracies, label=f'Client {i}', marker='o')
axs[1, 1].set_xlabel('Epoch')
axs[1, 1].set_ylabel('Validation Accuracy')
axs[1, 1].set_yscale('log')
axs[1, 1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.4f}'))
axs[1, 1].set_title('Validation Accuracy per Client')
axs[1, 1].legend()

# Validation Precision per Client
for i, precisions in enumerate(client_val_precisions):
    axs[2, 0].plot(precisions, label=f'Client {i}', marker='o')
axs[2, 0].set_xlabel('Epoch')
axs[2, 0].set_ylabel('Validation Precision')
axs[2, 0].set_yscale('log')
axs[2, 0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.4f}'))
axs[2, 0].set_title('Validation Precision per Client')
axs[2, 0].legend()

# Validation Recall per Client
for i, recalls in enumerate(client_val_recalls):
    axs[2, 1].plot(recalls, label=f'Client {i}', marker='o')
axs[2, 1].set_xlabel('Epoch')
axs[2, 1].set_ylabel('Validation Recall')
axs[2, 1].set_yscale('log')
axs[2, 1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.4f}'))
axs[2, 1].set_title('Validation Recall per Client')
axs[2, 1].legend()

# Validation F1 per Client
for i, f1s in enumerate(client_val_f1s):
    axs[3, 0].plot(f1s, label=f'Client {i}', marker='o')
axs[3, 0].set_xlabel('Epoch')
axs[3, 0].set_ylabel('Validation F1')
axs[3, 0].set_yscale('log')
axs[3, 0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.4f}'))
axs[3, 0].set_title('Validation F1 per Client')
axs[3, 0].legend()

# Training Loss Distribution among Clients
train_losses_last_epoch = [losses[-1] for losses in client_train_losses]
axs[4, 0].hist(train_losses_last_epoch, bins=10, edgecolor='black')
axs[4, 0].set_xlabel('Training Loss')
axs[4, 0].set_ylabel('Number of Clients')
axs[4, 0].set_title('Distribution of Training Losses among Clients')

# Training Accuracy Distribution among Clients
train_accuracies_last_epoch = [accuracies[-1] for accuracies in client_train_accuracies]
axs[4, 1].hist(train_accuracies_last_epoch, bins=10, edgecolor='black')
axs[4, 1].set_xlabel('Training Accuracy')
axs[4, 1].set_ylabel('Number of Clients')
axs[4, 1].set_title('Distribution of Training Accuracies among Clients')

# Validation Loss Distribution among Clients
val_losses_last_epoch = [losses[-1] for losses in client_val_losses]
axs[5, 0].hist(val_losses_last_epoch, bins=10, edgecolor='black')
axs[5, 0].set_xlabel('Validation Loss')
axs[5, 0].set_ylabel('Number of Clients')
axs[5, 0].set_title('Distribution of Validation Losses among Clients')

# Validation Accuracy Distribution among Clients
val_accuracies_last_epoch = [accuracies[-1] for accuracies in client_val_accuracies]
axs[5, 1].hist(val_accuracies_last_epoch, bins=10, edgecolor='black')
axs[5, 1].set_xlabel('Validation Accuracy')
axs[5, 1].set_ylabel('Number of Clients')
axs[5, 1].set_title('Distribution of Validation Accuracies among Clients')


# Comparison TEST of Centralized and Federated Learning
print(Fore.RED + "* 2.12-Comparison TEST of Centralized and Federated Learning")

labels = ['Centralized', 'Federated']
test_losses = [centralized_test_loss, federated_test_loss]
test_accuracies = [centralized_test_accuracy, federated_test_accuracy]
test_precisions = [centralized_test_precision, federated_test_precision]
test_recalls = [centralized_test_recall, federated_test_recall]
test_f1s = [centralized_test_f1, federated_test_f1]

# Additional plot for better comparison
fig, ax = plt.subplots(1, 5, figsize=(25, 5))

# Test Loss comparison
ax[0].bar(labels, test_losses, color=['blue', 'orange'])
ax[0].set_yscale('log')
ax[0].set_ylabel('Test Loss')
ax[0].set_title('Test Loss Comparison')

# Test Accuracy comparison
ax[1].bar(labels, test_accuracies, color=['blue', 'orange'])
ax[1].set_yscale('log')
ax[1].set_ylabel('Test Accuracy')
ax[1].set_title('Test Accuracy Comparison')

# Test Precision comparison
ax[2].bar(labels, test_precisions, color=['blue', 'orange'])
ax[2].set_yscale('log')
ax[2].set_ylabel('Test Precision')
ax[2].set_title('Test Precision Comparison')

# Test Recall comparison
ax[3].bar(labels, test_recalls, color=['blue', 'orange'])
ax[3].set_yscale('log')
ax[3].set_ylabel('Test Recall')
ax[3].set_title('Test Recall Comparison')

# Test F1 comparison
ax[4].bar(labels, test_f1s, color=['blue', 'orange'])
ax[4].set_yscale('log')
ax[4].set_ylabel('Test F1')
ax[4].set_title('Test F1 Comparison')

for axis in ax:
    axis.set_xlabel('Learning Type')
    axis.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.4f}'))

fig.tight_layout()
plt.show()


#################################################################################

file_name = os.path.basename(__file__)
print(f"**************************************")
print("Name of the file: " + Fore.RED + f"{file_name}")
print(f"**************************************")
