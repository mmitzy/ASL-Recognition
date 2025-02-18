import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
from Evaluate import Evaluate
import matplotlib.pyplot as plt

class CNNModel(nn.Module):
    def __init__(self, num_classes):
        super(CNNModel, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)  
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1) 
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1) 

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Halves the spatial dimensions

        self.fc1 = nn.Linear(128 * 8 * 8, 512)  
        self.fc2 = nn.Linear(512, num_classes)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        
        x = x.view(x.size(0), -1)  
        
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
    def save_model(self, file_path):
        torch.save(self.state_dict(), file_path)
        print(f"Model saved to {file_path}")

    def load_model(self, file_path, device):
        self.load_state_dict(torch.load(file_path))
        self.to(device)  # Move the model to the specified device
        print(f"Model loaded from {file_path}")


# -------------------------------
# Example Usage

# Hyperparameters
num_classes = 29  # Number of classes (sign language letters)
learning_rate = 0.001
num_epochs = 5
batch_size = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((64, 64)), 
    transforms.ToTensor(),  
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]) 
])

train_data_path = "archive/asl_alphabet_train"
test_data_path = "archive/asl_alphabet_test"
model_save_path = "training_files/cnn_model.pth"

if not os.path.exists(train_data_path):
    raise ValueError(f"Training data path {train_data_path} does not exist!")
if not os.path.exists(test_data_path):
    raise ValueError(f"Test data path {test_data_path} does not exist!")

train_dataset = datasets.ImageFolder(root=train_data_path, transform=transform)
test_dataset = datasets.ImageFolder(root=test_data_path, transform=transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

model = CNNModel(num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

training_losses = []
training_accuracies = []

if not os.path.exists(model_save_path):
    print("Training the model...")
    model.to(device)
    model.train()  

    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            
            # Compute the loss
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        # Calculate and store training loss and accuracy for this epoch
        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = 100 * correct / total
        training_losses.append(epoch_loss)
        training_accuracies.append(epoch_accuracy)

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")
    
    model.save_model(model_save_path)
else:
    print("Model already exists, skipping training.")

print("Loading the model...")
model.load_model(model_save_path, device)

def plot_training_metrics(training_losses, training_accuracies):
    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss', color='tab:blue')
    ax1.plot(training_losses, color='tab:blue', label='Training Loss')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Accuracy (%)', color='tab:orange')
    ax2.plot(training_accuracies, color='tab:orange', label='Training Accuracy')
    ax2.tick_params(axis='y', labelcolor='tab:orange')

    plt.title('Training Loss and Accuracy')
    plt.tight_layout()
    plt.savefig('training_metrics.png', dpi=300)
    print("Training metrics saved as 'training_metrics.png'")
    plt.show()

plot_training_metrics(training_losses, training_accuracies)

print("Evaluating the model...")
evaluator = Evaluate()  
accuracy = evaluator.evaluate_model(model, test_loader, device)  
