import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torchvision import transforms, models 
from PIL import Image
import random
import matplotlib.pyplot as plt

# --------------------------
# Configuration
# --------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_PATH = "./KMU-FED"
NUM_NODES = 20
NUM_NEIGHBORS = 3
MALICE_PROB = 0.1
BATCH_SIZE = 128
EPOCHS = 20

# --------------------------
# Dataset Configuration
# --------------------------
EMOTION_LABELS = {
    "HA": 0,  # Happy
    "SA": 1,  # Sad
    "FE": 2,  # Fear
    "DI": 3,  # Disgust
    "SU": 4,  # Surprised
    "AN": 5   # Angry
}

class EmotionDataset(data.Dataset):
    """Custom dataset loader for KMU-FED facial expressions"""
    
    def __init__(self, data_root, transform=None):
        self.data_root = data_root
        self.transform = transform
        self.samples = self._validate_files()
        
    def _validate_files(self):
        """Validate files and extract labels"""
        valid_files = []
        for f in os.listdir(self.data_root):
            if f.lower().endswith(('.jpg', '.png', '.jpeg')):
                try:
                    label_code = f.split('_')[1]
                    if label_code in EMOTION_LABELS:
                        valid_files.append(f)
                except IndexError:
                    continue
        if not valid_files:
            raise ValueError(f"No valid images found in {self.data_root}")
        return valid_files
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_name = self.samples[idx]
        img_path = os.path.join(self.data_root, img_name)
        
        # Load and preprocess
        image = Image.open(img_path).convert('L')  # Grayscale
        label_code = img_name.split('_')[1]
        label = EMOTION_LABELS[label_code]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

# --------------------------
# Model Architecture
# --------------------------
class EmotionResNet(nn.Module):
    """ResNet-18 adapted for facial emotion recognition"""
    
    def __init__(self):
        super(EmotionResNet, self).__init__()
        base_model = models.resnet18(pretrained=False)
        
        # Modify first convolutional layer for grayscale
        base_model.conv1 = nn.Conv2d(1, 64, kernel_size=3, 
                                   stride=1, padding=1, bias=False)
                                   
        # Adjust final layer for 6 classes
        base_model.fc = nn.Linear(base_model.fc.in_features, 6)
        
        self.model = base_model

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
        self.model = EmotionResNet().to(DEVICE)
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
            new_label = random.choice([i for i in range(6) if i != original])
            corrupted[idx] = new_label
            
        return corrupted

    def aggregate_parameters(self):
        """Dynamic parameter aggregation with all peers"""
        if not self.peers:
            return
            
        with torch.no_grad():
            models = [self.model] + [peer.model for peer in self.peers]
            
            for param_idx, param_self in enumerate(self.model.parameters()):
                # Gather parameters from all models
                all_params = [list(model.parameters())[param_idx].data for model in models]
                stacked_params = torch.stack(all_params)
                
                # Calculate mean across all peers
                param_self.data.copy_(torch.mean(stacked_params, dim=0))

    def evaluate(self, test_loader):
        """Assess model performance on validation data"""
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
# Training Utilities
# --------------------------
def create_data_pipeline():
    """Create transformed data loaders"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    full_dataset = EmotionDataset(DATA_PATH, transform=transform)
    
    # Split dataset
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_set, test_set = data.random_split(full_dataset, [train_size, test_size])
    
    return (
        data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4),
        data.DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    )

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
                node.aggregate_parameters()
                
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
    plt.savefig("emotion_training.png", bbox_inches='tight')
    plt.close()

# --------------------------
# Main Execution
# --------------------------
if __name__ == "__main__":
    # Initialize components
    train_loader, test_loader = create_data_pipeline()
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