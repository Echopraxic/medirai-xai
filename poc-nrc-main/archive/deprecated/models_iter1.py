import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torchvision import datasets, transforms
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

import matplotlib.pyplot as plt
import datetime


class MelonomaModelIter1(nn.Module):

    def __init__(self, img_width=224, img_height=224):
        super(MelonomaModelIter1, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * (img_width // 8) * (img_height // 8), 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 1)
        self.pred = nn.Sigmoid()


    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.pred(x)
        return x


class train_iter1():

    def __init__(self, 
                train_dir, 
                test_dir,
                model,
                img_width = 224, 
                img_height = 224,
                batch_size = 32,
                epochs=20,
                verbose=True):

        transform_train = transforms.Compose([
            transforms.Resize((img_width, img_height)),
            transforms.RandomRotation(20),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomResizedCrop(img_width),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        transform_test = transforms.Compose([
            transforms.Resize((img_width, img_height)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        train_dataset = datasets.ImageFolder(
            train_dir, 
            transform=transform_train
        )
        test_dataset = datasets.ImageFolder(
            test_dir, 
            transform=transform_test
        )

        self.train_loader = data.DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            pin_memory=True)
        self.test_loader = data.DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            pin_memory=True)
        
        self.model = model
        self.verbose = verbose
        self.epochs = epochs
        self.img_w = img_width
        self.img_h = img_height
        self.batch_size = batch_size
        self.cuda = torch.cuda.is_available() 


    def _evaluate_model(self, criterion):

        self.model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in self.test_loader:
                if self.cuda:
                    inputs, labels = inputs.cuda(non_blocking=True), labels.float().cuda(non_blocking=True)
                labels = labels.float()
                outputs = self.model(inputs).squeeze()
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        
        return val_loss / len(self.test_loader)
    
        
    def train_model(self):

        self.model.train()
        train_loss, val_loss = [], []
        train_date = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        best_model_outname = f'best_model-training_started-{train_date}.pkt'
        last_model_outname = f'last_model-training_started-{train_date}.pkt'

        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        min_val_loss = 0
        for epoch in range(self.epochs):
            running_loss = 0.0
            
            for inputs, labels in self.train_loader:
                if self.cuda:
                    inputs, labels = inputs.cuda(non_blocking=True), labels.float().cuda(non_blocking=True)
                labels = labels.float()
                optimizer.zero_grad()
                outputs = self.model(inputs).squeeze()
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            
            epoch_loss = running_loss / len(self.train_loader)
            val_loss_epoch = self._evaluate_model(criterion)
            
            train_loss.append(epoch_loss)
            val_loss.append(val_loss_epoch)
            
            if self.verbose:
                print(f"Epoch {epoch+1}/{self.epochs}, Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss_epoch:.4f}")
            
            if epoch_loss < min_val_loss or epoch == 0:
                torch.save(self.model, best_model_outname)
                min_val_loss = epoch_loss
            #self.evaluate_metrics(self.model, self.test_loader)

        torch.save(self.model, last_model_outname)  
        return train_loss, val_loss


    def evaluate_metrics(self):

        self.model.eval()
        y_true, y_pred = [], []
        
        with torch.no_grad():
            for inputs, labels in self.test_loader:
                
                if self.cuda:
                    inputs = inputs.cuda()
                    labels = labels.cuda()
                    outputs = self.model(inputs).squeeze().cuda().numpy()
                else:
                    outputs = self.model(inputs).squeeze().cpu().numpy()
 
                preds = (outputs > 0.5).astype(int)
                
                y_pred.extend(preds.flatten())
                y_true.extend(labels.cpu().numpy())
        
        # Classification metrics
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred))
        
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_true, y_pred))
        
        print(f"\nROC-AUC Score: {roc_auc_score(y_true, y_pred)}")