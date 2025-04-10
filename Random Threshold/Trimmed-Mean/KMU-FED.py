import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torchvision import transforms, models
import matplotlib.pyplot as plt
from PIL import Image
import random

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Emotion label mapping based on filename
emotion_mapping = {
    "HA": 0,  # Happy
    "SA": 1,  # Sad
    "FE": 2,  # Fear
    "DI": 3,  # Disgust
    "SU": 4,  # Surprised
    "AN": 5   # Angry
}

# Custom dataset class for KMU-FED
class KMUFEDDataset(data.Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]

    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)

        # Load image
        image = Image.open(img_path).convert("L")  # Convert to grayscale

        # Extract label from filename
        label_tag = img_name.split('_')[1]  # e.g., "04_FE_s03_074" → "FE"
        label = emotion_mapping.get(label_tag, -1)  # Default to -1 if label is not recognized

        if label == -1:
            raise ValueError(f"Unknown emotion label in filename: {img_name}")

        # Apply transformations
        if self.transform:
            image = self.transform(image)

        return image, label

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize for ResNet
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize grayscale images
])

# Load dataset
dataset_path = "ADAS Dataset"  # Change this to your dataset path
dataset = KMUFEDDataset(dataset_path, transform=transform)

# Split dataset (80% train, 20% test)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

# Create DataLoaders
train_loader = data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
test_loader = data.DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4)

# Modify ResNet-18 for single-channel input and 6 output classes
class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        self.model = models.resnet18(pretrained=False)
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, 6)
    
    def forward(self, x):
        return self.model(x)

# Decentralized SGD-based learning node
class DSGDNode:
    def __init__(self, node_id, lr=0.001, is_malicious=False, flip_percentage=0.2):
        self.node_id = node_id
        self.model = ResNet18().to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()
        self.peers = []
        self.is_malicious = is_malicious
        self.flip_percentage = flip_percentage
    
    def set_peers(self, peers):
        self.peers = peers
    
    def train_step(self, data, target):
        # If the node is malicious, flip a percentage of the labels
        if self.is_malicious:
            target = self.flip_labels(target)
        
        data, target = data.to(device), target.to(device)  # Move batch to GPU
        self.optimizer.zero_grad()
        output = self.model(data)
        loss = self.criterion(output, target)
        loss.backward()
        self.optimizer.step()
        return loss.item(), output, target
    
    def flip_labels(self, target):
        # Flip a percentage of the labels
        flipped_target = target.clone()
        num_flips = int(self.flip_percentage * target.size(0))
        flip_indices = random.sample(range(target.size(0)), num_flips)
        
        for idx in flip_indices:
            original_label = target[idx].item()
            # Randomly select a different label to flip to
            new_label = random.choice([i for i in range(6) if i != original_label])
            flipped_target[idx] = new_label
        
        return flipped_target
    
    def decentralized_sgd_update(self, trim=1):
        """Trimmed Mean aggregation for robustness"""
        if not self.peers:
            return
        with torch.no_grad():
            # Include self and all peers
            models = [self.model] + [peer.model for peer in self.peers]
            # Process each parameter
            for param_idx, param_self in enumerate(self.model.parameters()):
                # Gather parameters from all models
                all_params = [list(model.parameters())[param_idx].data for model in models]
                stacked = torch.stack(all_params)  # Shape: [num_models, *param_shape]
                # Sort and trim extremes
                sorted_params = torch.sort(stacked, dim=0).values
                trimmed = sorted_params[trim:-trim]  # Remove extremes
                avg_param = torch.mean(trimmed, dim=0)
                # Update parameter
                param_self.data.copy_(avg_param)
    
    def accuracy(self, output, target):
        _, predicted = output.max(1)
        correct = (predicted == target).sum().item()
        return correct / target.size(0)

    def evaluate(self, test_loader):
        self.model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)  # Move test batch to GPU
                output = self.model(data)
                _, predicted = output.max(1)
                correct += (predicted == target).sum().item()
                total += target.size(0)
        return correct / total

# Simulation setup
num_nodes = 20
num_neighbors = 3
malicious_percentage = 0
flip_percentage = 0
nodes = [DSGDNode(i, is_malicious=(random.random() < malicious_percentage), flip_percentage=flip_percentage) for i in range(num_nodes)]

# Random threshold topology setup
for i in range(num_nodes):
    neighbors = random.sample([node for node in nodes if node != nodes[i]], num_neighbors)
    nodes[i].set_peers(neighbors)

# Visualization setup
loss_history = [[] for _ in range(num_nodes)]
accuracy_history = [[] for _ in range(num_nodes)]

# Training loop
for epoch in range(50):
    print(f"Epoch {epoch+1}")
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)  # Ensure data is on GPU
        for i, node in enumerate(nodes):
            loss, output, target = node.train_step(data, target)
            loss_history[i].append(loss)
            accuracy = node.accuracy(output, target)
            accuracy_history[i].append(accuracy)
            node.decentralized_sgd_update(trim=1)

# Test evaluation after training
print("\nStarting test evaluation...")
test_accuracies = []
for i, node in enumerate(nodes):
    accuracy = node.evaluate(test_loader)
    test_accuracies.append(accuracy)
    print(f"Node {i} Test Accuracy: {accuracy:.4f}")

# Plot loss and accuracy history
print("\nPlotting loss and accuracy graphs...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Loss Plot
for i in range(num_nodes):
    ax1.plot(loss_history[i], label=f'Node {i}')
ax1.set_xlabel("Iterations")
ax1.set_ylabel("Loss")
ax1.set_title("Loss Evolution in Decentralized SGD (Ring Topology)")
#ax1.legend()

# Accuracy Plot
for i in range(num_nodes):
    ax2.plot(accuracy_history[i], label=f'Node {i}')
ax2.set_xlabel("Iterations")
ax2.set_ylabel("Accuracy")
ax2.set_title("Accuracy Evolution in Decentralized SGD (Ring Topology)")
#ax2.legend()

plt.tight_layout()
plt.savefig("ADAS.png") 
#plt.show()
