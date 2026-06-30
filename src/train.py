import torch 
import torch.nn as nn 
from model import CNNModel 
from dataset import get_dataloader
import torch.optim as optim 
from torchmetrics import Recall , Precision 



def train_pipeline():
    NUM_EPOCHS = 10
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    NUM_CLASSES = 7 
    TRAIN_DIR=r"data\clouds_train"
    TEST_DIR=r"data\clouds_test"


    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print ("the device used : ", device)

    train_loader , test_loader , class_to_idx = get_dataloader(TRAIN_DIR , TEST_DIR , batch_size=BATCH_SIZE)

    model=CNNModel(num_classes=NUM_CLASSES).to(device)
    criterion=nn.CrossEntropyLoss()
    optimizer=optim.Adam(model.parameters(), lr = LEARNING_RATE)

    metric_precision = Precision(task="multiclass", num_classes=NUM_CLASSES, average="macro").to(device)
    metric_recall = Recall(task="multiclass", num_classes=NUM_CLASSES , average="macro").to(device)

    print("Starting training...")
    print ("#"*60)


    for epoch in range(NUM_EPOCHS):
        # Training Phase
        model.train()
        running_loss = 0.0
        for images , labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs=model(images)
            loss=criterion(outputs , labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            optimizer.zero_grad()

        mean_train_loss = running_loss / len(train_loader)

        # Evaluation Phase
        model.eval()
        running_test_loss = 0.0

        with torch.no_grad():
            for images , labels in test_loader:
                images, labels = images.to(device) , labels.to(device)
                outputs=model(images)
                loss=criterion(outputs , labels)
                running_test_loss += loss.item()

                _, preds = torch.max(outputs, 1)
                metric_precision.update(preds, labels)
                metric_recall.update(preds, labels)

        mean_test_loss=running_test_loss / len(test_loader)
# Finalize overall metrics calculation for the complete epoch
        epoch_precision = metric_precision.compute()
        epoch_recall = metric_recall.compute()
        
        # Flush evaluation states to zero for the next incoming epoch
        metric_precision.reset()
        metric_recall.reset()
        
        # 5. Print Epoch Metrics Dashboard
        print(f"Epoch [{epoch+1:02d}/{NUM_EPOCHS:02d}]")
        print(f"  -> Train Loss: {mean_train_loss:.4f}")
        print(f"  -> Test Loss:  {mean_test_loss:.4f}")
        print(f"  -> Test Macro-Precision: {epoch_precision:.2%} | Test Macro-Recall: {epoch_recall:.2%}")
        print("-" * 60)
        
    print("Training Cycle Successfully Complete!")

if __name__ == "__main__":
    train_pipeline()
            