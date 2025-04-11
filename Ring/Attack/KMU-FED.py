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
# Hardware Configuration
# --------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------
# Dataset Configuration
# --------------------------
EMOTION_MAPPING = {
    "HA": 0,  # Happy
    "SA": 1,  # Sad
    "FE": 2,  # Fear
    "DI": 3,  # Disgust
    "SU": 4,  # Surprised
    "AN": 5   # Angry
}

class KMUFEDDataset(data.Dataset):
    """Custom dataset loader for KMU-FED facial expression data"""
    
    def __init__(self, data_root, transform=None):
        """
        Args:
            data_root (str): Path to directory containing images
            transform (callable): Optional transform to apply to images
        """
        self.data_root = data_root
        self.transform = transform
        self.image_files = [f for f in os.listdir(data_root) 
                           if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        
        # Validate dataset
        if not self.image_files:
            raise ValueError(f"No images found in {data_root}")

    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.data_root, img_name)
        
        # Load and convert image to grayscale
        image = Image.open(img_path).convert('L')
        
        # Extract emotion label from filename
        try:
            label_code = img_name.split('_')[1]
            label = EMOTION_MAPPING[label_code]
        except (IndexError, KeyError) as e:
            raise ValueError(f"Invalid filename format: {img_name}") from e
            
        # Apply transformations
        if self.transform:
            image = self.transform(image)
            
        return image, label

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
                                   
        # Modify final layer for 6 emotion classes
        base_model.fc = nn.Linear(base_model.fc.in_features, 6)
        
        self.model = base_model

    def forward(self, x):
        return self.model(x)

# --------------------------
# Decentralized Learning Node
# --------------------------
class DecentralizedNode:
    """Node in decentralized network with potential adversarial behavior"""
    
    def __init__(self, node_id, lr=0.001, is_malicious=False, corruption_rate=1.0):
        """
        Args:
            node_id: Unique identifier for the node
            lr: Learning rate for Adam optimizer
            is_malicious: Flag for adversarial behavior
            corruption_rate: Fraction of labels to corrupt if malicious
        """
        self.node_id = node_id
        self.model = EmotionResNet().to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()
        self.peers = []
        self.is_malicious = is_malicious
        self.corruption_rate = corruption_rate

    def set_peers(self, left_peer, right_peer):
        """Configure neighbors in ring topology"""
        self.peers = [left_peer, right_peer]

    def train_step(self, images, labels):
        """Execute one training iteration with optional label corruption"""
        if self.is_malicious:
            labels = self._corrupt_labels(labels)
            
        images, labels = images.to(device), labels.to(device)
        
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

    def synchronize_parameters(self):
        """Average model parameters with neighbors"""
        if not self.peers:
            return
            
        with torch.no_grad():
            for self_param, left_param, right_param in zip(
                self.model.parameters(),
                self.peers[0].model.parameters(),
                self.peers[1].model.parameters()
            ):
                self_param.data = (self_param.data + 
                                 left_param.data + 
                                 right_param.data) / 3

    def evaluate(self, test_loader):
        """Assess model performance on validation data"""
        self.model.eval()
        correct, total = 0, 0
        
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = self.model(images)
                _, predicted = outputs.max(1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
                
        return correct / total

# --------------------------
# Training Utilities
# --------------------------
def create_data_pipeline(data_path, batch_size=128):
    """Create transformed data loaders"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    full_dataset = KMUFEDDataset(data_path, transform=transform)
    
    # Split dataset
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_set, test_set = data.random_split(full_dataset, [train_size, test_size])
    
    return (
        data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4),
        data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4)
    )

def initialize_network(num_nodes=10, malice_prob=0.1):
    """Create decentralized node network"""
    nodes = []
    for node_id in range(num_nodes):
        is_malicious = random.random() < malice_prob
        nodes.append(DecentralizedNode(
            node_id,
            is_malicious=is_malicious,
            corruption_rate=1.0
        ))
    return nodes

def configure_topology(nodes):
    """Set up ring topology connections"""
    num_nodes = len(nodes)
    for i, node in enumerate(nodes):
        left = nodes[(i - 1) % num_nodes]
        right = nodes[(i + 1) % num_nodes]
        node.set_peers(left, right)

def train_network(nodes, train_loader, epochs=20):
    """Execute decentralized training process"""
    loss_history = [[] for _ in nodes]
    accuracy_history = [[] for _ in nodes]
    
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            for i, node in enumerate(nodes):
                loss, outputs, labels = node.train_step(images, labels)
                acc = (outputs.argmax(1) == labels).float().mean().item()
                
                loss_history[i].append(loss)
                accuracy_history[i].append(acc)
                node.synchronize_parameters()
                
    return loss_history, accuracy_history

def visualize_training(loss_hist, acc_hist, save_path="training_metrics.png"):
    """Generate training performance visualization"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss curves
    for i, losses in enumerate(loss_hist):
        ax1.plot(losses, label=f'Node {i}')
    ax1.set_title("Training Loss Evolution")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Loss")
    
    # Accuracy curves
    for i, accuracies in enumerate(acc_hist):
        ax2.plot(accuracies, label=f'Node {i}')
    ax2.set_title("Training Accuracy Evolution")
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Accuracy")
    
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# --------------------------
# Main Execution
# --------------------------
if __name__ == "__main__":
    # Configuration
    DATA_PATH = "./KMU-FED"  # Update with actual path
    NUM_NODES = 10
    MALICE_PROBABILITY = 0.1
    
    # Initialize components
    train_loader, test_loader = create_data_pipeline(DATA_PATH)
    nodes = initialize_network(NUM_NODES, MALICE_PROBABILITY)
    configure_topology(nodes)
    
    # Train network
    loss_history, acc_history = train_network(nodes, train_loader)
    
    # Evaluate performance
    print("\nFinal Test Accuracies:")
    for i, node in enumerate(nodes):
        accuracy = node.evaluate(test_loader)
        print(f"Node {i}: {accuracy:.2%}")
    
    # Generate visualization
    visualize_training(loss_history, acc_history, "emotion_training.png")