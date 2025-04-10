import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import random 

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Use ResNet-18 for CIFAR-10
class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        self.model = models.resnet18(pretrained=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, 10)  # Output for 10 classes
    
    def forward(self, x):
        return self.model(x)

# Decentralized SGD-based learning node
class DSGDNode:
    def __init__(self, node_id, lr=0.001, is_malicious=False, flip_percentage=0.2):
        self.node_id = node_id
        self.model = ResNet18().to(device)  # Move model to GPU
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()
        self.peers = []  # Two neighbors in a ring topology
        self.is_malicious = is_malicious  # Flag for malicious nodes
        self.flip_percentage = flip_percentage  # Percentage of labels to flip for malicious nodes
    
    def set_peers(self, left_neighbor, right_neighbor):
        self.peers = [left_neighbor, right_neighbor]
    
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
        if not self.peers:
            return
        with torch.no_grad():
            models = [self.model] + [peer.model for peer in self.peers]
            for param_idx, param_self in enumerate(self.model.parameters()):
                all_params = [list(model.parameters())[param_idx].data for model in models]
                stacked = torch.stack(all_params)
                sorted_params = torch.sort(stacked, dim=0).values
                trimmed = sorted_params[trim:-trim]
                avg_param = torch.mean(trimmed, dim=0)
                param_self.data.copy_(avg_param)
                # Free memory
                del stacked, sorted_params, trimmed, avg_param
            torch.cuda.empty_cache()  # Clear CUDA cache after all parameters

    def flip_labels(self, target):
        flipped_target = target.clone()
        num_flips = int(self.flip_percentage * target.size(0))
        flip_indices = random.sample(range(target.size(0)), num_flips)
        
        for idx in flip_indices:
            original_label = target[idx].item()
            new_label = random.choice([i for i in range(10) if i != original_label])  # Corrected to 10 classes
            flipped_target[idx] = new_label
        
        return flipped_target
        
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

# Load CIFAR-10 dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
train_dataset = datasets.CIFAR10(root="./data", train=True, transform=transform, download=True)
train_loader = data.DataLoader(train_dataset, batch_size=512, shuffle=True, num_workers=4)

test_dataset = datasets.CIFAR10(root="./data", train=False, transform=transform, download=True)
test_loader = data.DataLoader(test_dataset, batch_size=512, shuffle=False, num_workers=4)

# Simulation setup
num_nodes = 10
malicious_percentage = 0.1  # Percentage of malicious nodes
flip_percentage = 1  # Percentage of labels to flip for malicious nodes

# Randomly assign malicious nodes
nodes = []
for i in range(num_nodes):
    is_malicious = random.random() < malicious_percentage  # Randomly decide if the node is malicious
    node = DSGDNode(i, is_malicious=is_malicious, flip_percentage=flip_percentage)
    nodes.append(node)

# Ring topology setup
for i in range(num_nodes):
    left_neighbor = nodes[(i - 1) % num_nodes]  # Wrap around
    right_neighbor = nodes[(i + 1) % num_nodes]  # Wrap around
    nodes[i].set_peers(left_neighbor, right_neighbor)

# Visualization setup
loss_history = [[] for _ in range(num_nodes)]
accuracy_history = [[] for _ in range(num_nodes)]

# Training loop
for epoch in range(20):
    print(f"Epoch {epoch+1}")
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
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
plt.savefig("CIFAR-10")
#plt.show()