import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torchvision import datasets, transforms, models 
import matplotlib.pyplot as plt
import random

# --------------------------
# Configuration
# --------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_NODES = 20
NUM_NEIGHBORS = 3
MALICE_PROB = 0.1
BATCH_SIZE = 1024
EPOCHS = 20

# --------------------------
# Model Architecture
# --------------------------
class CIFARResNet(nn.Module):
    """ResNet-18 adapted for CIFAR-10 classification"""
    
    def __init__(self):
        super(CIFARResNet, self).__init__()
        self.model = models.resnet18(pretrained=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, 10)
        
    def forward(self, x):
        return self.model(x)

# --------------------------
# Decentralized Node
# --------------------------
class DecentralizedNode:
    """Node in decentralized network with Byzantine capabilities"""
    
    def __init__(self, node_id, lr=0.001, is_malicious=False, corruption_rate=1.0):
        """
        Args:
            node_id: Unique node identifier
            lr: Learning rate for Adam optimizer
            is_malicious: Flag for adversarial behavior
            corruption_rate: Fraction of labels to corrupt
        """
        self.node_id = node_id
        self.model = CIFARResNet().to(DEVICE)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()
        self.peers = []
        self.is_malicious = is_malicious
        self.corruption_rate = corruption_rate

    def set_peers(self, peers):
        """Configure neighbors in random threshold topology"""
        self.peers = peers

    def train_step(self, images, labels):
        """Execute training iteration with optional label corruption"""
        if self.is_malicious:
            labels = self._corrupt_labels(labels)
            
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        
        self.optimizer.zero_grad()
        outputs = self.model(images)
        loss = self.criterion(outputs, labels)
        loss.backward()
        self.optimizer.step()
        
        return loss.item(), outputs, labels

    def _corrupt_labels(self, labels):
        """Adversarial label flipping mechanism"""
        corrupted = labels.clone()
        num_corrupt = int(self.corruption_rate * labels.size(0))
        corrupt_indices = random.sample(range(labels.size(0)), num_corrupt)
        
        for idx in corrupt_indices:
            original = labels[idx].item()
            new_label = random.choice([i for i in range(10) if i != original])
            corrupted[idx] = new_label
            
        return corrupted

    def synchronize_parameters(self):
        """Average parameters with randomly selected neighbors"""
        if not self.peers:
            return
            
        with torch.no_grad():
            for self_param, *peer_params in zip(
                self.model.parameters(),
                *[peer.model.parameters() for peer in self.peers]
            ):
                # Calculate mean of self + neighbor parameters
                param_sum = self_param.data.clone()
                for p in peer_params:
                    param_sum += p.data
                self_param.data.copy_(param_sum / (len(peer_params) + 1))

    def evaluate(self, test_loader):
        """Assess model performance on test data"""
        self.model.eval()
        correct, total = 0, 0
        
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = self.model(images)
                _, predicted = outputs.max(1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
                
        return correct / total

# --------------------------
# Data Pipeline
# --------------------------
def create_data_loaders():
    """Create augmented CIFAR-10 data loaders"""
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    train_set = datasets.CIFAR10(
        root="./data", 
        train=True, 
        transform=transform, 
        download=True
    )
    
    test_set = datasets.CIFAR10(
        root="./data", 
        train=False, 
        transform=transform
    )
    
    return (
        data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4),
        data.DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    )

# --------------------------
# Network Setup
# --------------------------
def initialize_network():
    """Create decentralized node network with random topology"""
    nodes = [DecentralizedNode(i, is_malicious=(random.random() < MALICE_PROB)) 
            for i in range(NUM_NODES)]
    
    # Configure random threshold topology
    for i, node in enumerate(nodes):
        potential_peers = [n for n in nodes if n != node]
        neighbors = random.sample(potential_peers, NUM_NEIGHBORS)
        node.set_peers(neighbors)
        
    return nodes

# --------------------------
# Training & Visualization
# --------------------------
def train_network(nodes, train_loader):
    """Execute decentralized training process"""
    loss_history = [[] for _ in nodes]
    accuracy_history = [[] for _ in nodes]
    
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch+1}/{EPOCHS}")
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            for i, node in enumerate(nodes):
                loss, outputs, labels = node.train_step(images, labels)
                acc = (outputs.argmax(1) == labels).float().mean().item()
                
                loss_history[i].append(loss)
                accuracy_history[i].append(acc)
                node.synchronize_parameters()
                
    return loss_history, accuracy_history

def visualize_performance(loss_hist, acc_hist):
    """Generate training metrics visualization"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    for i, losses in enumerate(loss_hist):
        ax1.plot(losses, alpha=0.7, label=f'Node {i}')
    ax1.set_title("Training Loss Evolution")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Loss")
    
    # Accuracy plot
    for i, accuracies in enumerate(acc_hist):
        ax2.plot(accuracies, alpha=0.7, label=f'Node {i}')
    ax2.set_title("Training Accuracy Evolution")
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Accuracy")
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig("cifar_training.png", bbox_inches='tight')
    plt.close()

# --------------------------
# Main Execution
# --------------------------
if __name__ == "__main__":
    # Initialize components
    train_loader, test_loader = create_data_loaders()
    nodes = initialize_network()
    
    # Train network
    loss_hist, acc_hist = train_network(nodes, train_loader)
    
    # Evaluate performance
    print("\nFinal Test Accuracies:")
    for i, node in enumerate(nodes):
        accuracy = node.evaluate(test_loader)
        print(f"Node {i}: {accuracy:.2%}")
    
    # Generate visualization
    visualize_performance(loss_hist, acc_hist)