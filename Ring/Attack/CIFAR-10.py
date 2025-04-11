import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import random

# --------------------------
# Hardware Configuration
# --------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------
# Model Definition
# --------------------------
class ResNet18(nn.Module):
    """ResNet-18 architecture adapted for CIFAR-10 classification"""
    def __init__(self):
        super(ResNet18, self).__init__()
        self.model = models.resnet18(pretrained=False)
        # Modify final fully connected layer for 10 classes
        self.model.fc = nn.Linear(self.model.fc.in_features, 10)
    
    def forward(self, x):
        return self.model(x)

# --------------------------
# Decentralized Learning Node
# --------------------------
class DSGDNode:
    """Decentralized SGD node with potential malicious behavior"""
    
    def __init__(self, node_id, lr=0.001, is_malicious=False, flip_percentage=0.2):
        """
        Initialize a decentralized learning node
        Args:
            node_id: Unique identifier for the node
            lr: Learning rate for Adam optimizer
            is_malicious: Flag indicating adversarial behavior
            flip_percentage: Fraction of labels to flip if malicious
        """
        self.node_id = node_id
        self.model = ResNet18().to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()
        self.peers = []  # Neighbors in ring topology
        self.is_malicious = is_malicious
        self.flip_percentage = flip_percentage

    def set_peers(self, left_neighbor, right_neighbor):
        """Define neighbors in the ring topology"""
        self.peers = [left_neighbor, right_neighbor]

    def train_step(self, data, target):
        """Execute one training step with optional label flipping"""
        if self.is_malicious:
            target = self._flip_labels(target)
            
        data, target = data.to(device), target.to(device)
        self.optimizer.zero_grad()
        output = self.model(data)
        loss = self.criterion(output, target)
        loss.backward()
        self.optimizer.step()
        return loss.item(), output, target

    def _flip_labels(self, target):
        """Adversarial label corruption mechanism"""
        flipped_target = target.clone()
        num_flips = int(self.flip_percentage * target.size(0))
        flip_indices = random.sample(range(target.size(0)), num_flips)
        
        for idx in flip_indices:
            original_label = target[idx].item()
            new_label = random.choice([i for i in range(10) if i != original_label])
            flipped_target[idx] = new_label
            
        return flipped_target

    def decentralized_sgd_update(self):
        """Parameter synchronization with neighbors using simple averaging"""
        if not self.peers:
            return
            
        with torch.no_grad():
            # Average parameters with neighbors' parameters
            for param_self, param_left, param_right in zip(
                self.model.parameters(),
                self.peers[0].model.parameters(),
                self.peers[1].model.parameters()
            ):
                param_self.data = (param_self.data + 
                                  param_left.data + 
                                  param_right.data) / 3

    def evaluate(self, test_loader):
        """Evaluate model performance on test dataset"""
        self.model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = self.model(data)
                _, predicted = output.max(1)
                correct += (predicted == target).sum().item()
                total += target.size(0)
        return correct / total

# --------------------------
# Data Loading
# --------------------------
def load_cifar10():
    """Load and normalize CIFAR-10 dataset"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    train_dataset = datasets.CIFAR10(
        root="./data", 
        train=True, 
        transform=transform, 
        download=True
    )
    
    test_dataset = datasets.CIFAR10(
        root="./data", 
        train=False, 
        transform=transform, 
        download=True
    )
    
    return (
        data.DataLoader(train_dataset, batch_size=1024, shuffle=True, num_workers=4),
        data.DataLoader(test_dataset, batch_size=1024, shuffle=False, num_workers=4)
    )

# --------------------------
# Simulation Setup
# --------------------------
def initialize_nodes(num_nodes=10, malicious_percentage=0.1):
    """Create network nodes with random malicious assignment"""
    nodes = []
    for i in range(num_nodes):
        is_malicious = random.random() < malicious_percentage
        nodes.append(DSGDNode(
            i,
            is_malicious=is_malicious,
            flip_percentage=1.0  # Full corruption if malicious
        ))
    return nodes

def setup_ring_topology(nodes):
    """Configure ring topology connections"""
    for i, node in enumerate(nodes):
        left = nodes[(i - 1) % len(nodes)]
        right = nodes[(i + 1) % len(nodes)]
        node.set_peers(left, right)

# --------------------------
# Training & Evaluation
# --------------------------
def training_loop(nodes, train_loader, epochs=20):
    """Execute decentralized training process"""
    loss_history = [[] for _ in nodes]
    accuracy_history = [[] for _ in nodes]
    
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            for i, node in enumerate(nodes):
                loss, output, target = node.train_step(data, target)
                loss_history[i].append(loss)
                accuracy = (output.argmax(1) == target).float().mean().item()
                accuracy_history[i].append(accuracy)
                node.decentralized_sgd_update()
                
    return loss_history, accuracy_history

def visualize_results(loss_history, accuracy_history):
    """Generate training metrics visualization"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    for i, losses in enumerate(loss_history):
        ax1.plot(losses, label=f'Node {i}')
    ax1.set_xlabel("Iterations")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training Loss Evolution")
    
    # Accuracy plot
    for i, accuracies in enumerate(accuracy_history):
        ax2.plot(accuracies, label=f'Node {i}')
    ax2.set_xlabel("Iterations")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Training Accuracy Evolution")
    
    plt.tight_layout()
    plt.legend()
    plt.savefig("cifar10_training_metrics.png")

# --------------------------
# Main Execution
# --------------------------
if __name__ == "__main__":
    # Load datasets
    train_loader, test_loader = load_cifar10()
    
    # Initialize decentralized network
    nodes = initialize_nodes(num_nodes=10, malicious_percentage=0.1)
    setup_ring_topology(nodes)
    
    # Run training process
    loss_hist, acc_hist = training_loop(nodes, train_loader)
    
    # Evaluate final performance
    print("\nFinal Test Accuracies:")
    for i, node in enumerate(nodes):
        accuracy = node.evaluate(test_loader)
        print(f"Node {i}: {accuracy:.2%}")
    
    # Generate visualization
    visualize_results(loss_hist, acc_hist)