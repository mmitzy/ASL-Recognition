import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
from Evaluate import Evaluate
import matplotlib.pyplot as plt

class SimpleNeuralNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, learning_rate=0.001):
        super(SimpleNeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, num_classes)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        x = x.view(x.size(0), -1) 
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

    def train_model(self, train_loader, num_epochs, device):
        self.to(device)  
        self.train() 
        
        training_losses = []
        training_accuracies = []
        
        for epoch in range(num_epochs):
            running_loss = 0.0
            correct = 0
            total = 0
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                
                self.optimizer.zero_grad()
                
                outputs = self(images)
                
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                
                running_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            epoch_loss = running_loss / len(train_loader)
            epoch_accuracy = 100 * correct / total
            training_losses.append(epoch_loss)
            training_accuracies.append(epoch_accuracy)

            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")
        
        self.plot_training_metrics(training_losses, training_accuracies)

    def save_model(self, file_path):
        torch.save(self.state_dict(), file_path)
        print(f"Model saved to {file_path}")

    def load_model(self, file_path, device):
        self.load_state_dict(torch.load(file_path))
        self.to(device)
        print(f"Model loaded from {file_path}")

    def plot_training_metrics(self, training_losses, training_accuracies):
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
        plt.savefig('training_metrics_nn.png', dpi=300)
        print("Training metrics saved as 'training_metrics_nn.png'")
        plt.show()

# -------------------------------
# Example Usage

# Hyperparameters
input_dim = 64 * 64 * 3  
hidden_dim = 512  
num_classes = 29  
learning_rate = 0.001
num_epochs = 10
batch_size = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((64, 64)),  
    transforms.ToTensor(),  
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  
])

train_data_path = "archive/asl_alphabet_train"
test_data_path = "archive/asl_alphabet_test"
model_save_path = "training_files/simple_nn_model.pth"

if not os.path.exists(train_data_path):
    raise ValueError(f"Training data path {train_data_path} does not exist!")
if not os.path.exists(test_data_path):
    raise ValueError(f"Test data path {test_data_path} does not exist!")

train_dataset = datasets.ImageFolder(root=train_data_path, transform=transform)
test_dataset = datasets.ImageFolder(root=test_data_path, transform=transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

model = SimpleNeuralNetwork(input_dim, hidden_dim, num_classes, learning_rate)

if os.path.exists(model_save_path):
    print("Model already exists, skipping training.")
    model.load_model(model_save_path, device)
else:
    print("Training the model...")
    model.train_model(train_loader, num_epochs, device)
    model.save_model(model_save_path)

print("Evaluating the model...")
evaluator = Evaluate() 
accuracy = evaluator.evaluate_model(model, test_loader, device) 
