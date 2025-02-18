import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_recall_fscore_support
import numpy as np

class Evaluate:
    def evaluate_model(self, model, test_loader, device):
        model.to(device)
        model.eval()
        correct = 0
        total = 0
        all_labels = []
        all_predictions = []

        class_names = test_loader.dataset.classes

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())

        accuracy = 100 * correct / total
        print(f'Accuracy of the model on the test set: {accuracy:.2f}%')

        cm = confusion_matrix(all_labels, all_predictions, labels=range(len(class_names)))

        plt.figure(figsize=(25, 20)) 

        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        disp.plot(cmap=plt.cm.Blues, xticks_rotation=0, ax=plt.gca())
        plt.title('Confusion Matrix (Results)', fontsize=18)
        plt.xlabel('Predicted Label', fontsize=14)
        plt.ylabel('True Label', fontsize=14)

        plt.tight_layout()
        plt.savefig('model_evaluation.png', dpi=300)
        print("Evaluation plots saved as 'model_evaluation.png'")
        plt.show()

        tp = cm.diagonal()  
        fp = cm.sum(axis=0) - tp  
        fn = cm.sum(axis=1) - tp  

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)

        precision = np.nan_to_num(precision, nan=0.0)
        recall = np.nan_to_num(recall, nan=0.0)

        overall_precision, overall_recall, overall_f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average='weighted', zero_division=0
        )

        print("\nOverall Metrics:")
        print(f"Precision: {overall_precision * 100:.2f}")
        print(f"Recall: {overall_recall * 100:.2f}")
        print(f"F1-Score: {overall_f1 * 100:.2f}")

        return accuracy, overall_precision, overall_recall, overall_f1
