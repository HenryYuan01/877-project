import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torchvision import transforms, models
import matplotlib.pyplot as plt
from PIL import Image
import random

# --------------------------
# Configuration
# --------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_NODES = 20
NUM_NEIGHBORS = 3
MALICE_PROB = 0.1
FLIP_PERCENTAGE = 1.0
BATCH_SIZE = 128
EPOCHS = 20
IMG_SIZE = 224
NUM_CLASSES = 6

# Emotion label mapping based on filename prefixes
EMOTION_MAP = {
    "HA": 0,  # Happy
    "SA": 1,  # Sad
    "FE": 2,  # Fear
    "DI": 3,  # Disgust
    "SU": 4,  # Surprised
    "AN": 5   # Angry
}

# --------------------------
# Data Pipeline
# --------------------------
class KMUFEDDataset(data.Dataset):
    """Custom dataset loader for KMU-FED facial expression data"""
    
    def __init__(self, data_root, transform=None):
        """
        Args:
            data_root: Path to directory containing images
            transform: Optional transform to be applied
        """
        self.data_root = data_root
        self.transform = transform
        self.image_files = [f for f in os.listdir(data_root) 
                          if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.data_root, img_name)
        
        # Load and convert to grayscale
        image = Image.open(img_path).convert('L')
        
        # Extract emotion label from filename
        label_code = img_name.split('_')[1]
        label = EMOTION_MAP.get(label_code, -1)
        
        if label == -1:
            raise ValueError(f"Invalid emotion code in {img_name}")
            
        if self.transform:
            image = self.transform(image)
            
        return image, label

def create_data_loaders(data_root="./KMU-FED", test_split=0.2):
    """Create train/test loaders with normalization"""
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    full_dataset = KMUFEDDataset(data_root, transform=transform)
    train_size = int((1 - test_split) * len(full_dataset))
    test_size = len(full_dataset) - train_size
    
    train_set, test_set = data.random_split(full_dataset, [train_size, test_size])
    
    return (
        data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4),
        data.DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    )

# --------------------------
# Model Architecture
# --------------------------
class EmotionResNet(nn.Module):
    """Modified ResNet-18 for facial emotion recognition"""
    
    def __init__(self):
        super(EmotionResNet, self).__init__()
        base_model = models.resnet18(pretrained=False)
        
        # Adapt first layer for grayscale input
        base_model.conv1 = nn.Conv2d(1, 64, kernel_size=3, 
                                   stride=1, padding=1, bias=False)
                                   
        # Modify final layer for emotion classes
        base_model.fc = nn.Linear(base_model.fc.in_features, NUM_CLASSES)
        
        self.model = base_model

    def forward(self, x):
        return self.model(x)

# --------------------------
# Decentralized Node
# --------------------------
class FederatedNode:
    """Node participating in federated learning with security features"""
    
    def __init__(self, node_id, lr=0.001, is_malicious=False, flip_rate=0.2):
        """
        Args:
            node_id: Unique identifier
            lr: Learning rate for optimizer
            is_malicious: Flag for Byzantine behavior
            flip_rate: Percentage of labels to corrupt if malicious
        """
        self.node_id = node_id
        self.model = EmotionResNet().to(DEVICE)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()
        self.peers = []
        self.is_malicious = is_malicious
        self.flip_rate = flip_rate

    def set_peers(self, peers):
        """Configure network neighbors"""
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
        """Adversarial label corruption mechanism"""
        corrupted = labels.clone()
        num_corrupt = int(self.flip_rate * labels.size(0))
        targets = random.sample(range(labels.size(0)), num_corrupt)
        
        for idx in targets:
            original = labels[idx].item()
            candidates = [i for i in range(NUM_CLASSES) if i != original]
            corrupted[idx] = random.choice(candidates)
            
        return corrupted

    def aggregate_parameters(self, trim=1):
        """Robust parameter aggregation using trimmed mean"""
        if not self.peers:
            return
            
        with torch.no_grad():
            # Include self and peer models
            all_models = [self.model] + [peer.model for peer in self.peers]
            
            for param_idx, self_param in enumerate(self.model.parameters()):
                # Collect corresponding parameters
                params = [list(m.parameters())[param_idx].data for m in all_models]
                stacked = torch.stack(params)
                
                # Trim extreme values and average
                sorted_params = torch.sort(stacked, dim=0).values
                trimmed = sorted_params[trim:-trim]
                self_param.data.copy_(torch.mean(trimmed, dim=0))

    def evaluate(self, test_loader):
        """Model performance assessment on test data"""
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
# Network Setup
# --------------------------
def initialize_network():
    """Create federated learning network with random topology"""
    nodes = [FederatedNode(i, 
                         is_malicious=(random.random() < MALICE_PROB),
                         flip_rate=FLIP_PERCENTAGE)
           for i in range(NUM_NODES)]
    
    # Create random peer connections
    for node in nodes:
        candidates = [n for n in nodes if n != node]
        peers = random.sample(candidates, NUM_NEIGHBORS)
        node.set_peers(peers)
        
    return nodes

# --------------------------
# Training & Evaluation
# --------------------------
def train_network(nodes, train_loader):
    """Execute federated training process"""
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
                node.aggregate_parameters(trim=1)
                
    return loss_history, accuracy_history

def visualize_performance(loss_hist, acc_hist, filename="ADAS.png"):
    """Generate training metrics visualization"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    for i, losses in enumerate(loss_hist):
        ax1.plot(losses, alpha=0.7, label=f'Node {i}')
    ax1.set_title("Training Loss Progression")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Loss")
    
    # Accuracy plot
    for i, accuracies in enumerate(acc_hist):
        ax2.plot(accuracies, alpha=0.7, label=f'Node {i}')
    ax2.set_title("Training Accuracy Progression")
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Accuracy")
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    plt.close()

# --------------------------
# Main Execution
# --------------------------
if __name__ == "__main__":
    # Initialize components
    train_loader, test_loader = create_data_loaders()
    nodes = initialize_network()
    
    # Conduct federated training
    loss_hist, acc_hist = train_network(nodes, train_loader)
    
    # Evaluate final performance
    print("\nFinal Test Accuracies:")
    for i, node in enumerate(nodes):
        accuracy = node.evaluate(test_loader)
        print(f"Node {i}: {accuracy:.2%}")
    
    # Generate visualizations
    visualize_performance(loss_hist, acc_hist)