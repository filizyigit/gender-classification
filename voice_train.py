import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Adım 1: Veri setinin yüklenmesi ve incelenmesi
df = pd.read_csv('C:/Users/filizyigit/Desktop/Bitirme Ödevi/voice/voice.csv')
print("Veri Seti Örneği:")
print(df.head())

# Adım 2: Veri ön işleme
X = df.drop('label', axis=1)
y = df['label']

# Verilerin ölçeklendirilmesi
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, 'scaler.pkl')
print("Scaler dosyası kaydedildi.")

# Eğitim, doğrulama ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)  # 0.25 x 0.8 = 0.2

# Verileri PyTorch tensorlerine dönüştürme
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.replace({'male': 0, 'female': 1}).values, dtype=torch.long)

X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val.replace({'male': 0, 'female': 1}).values, dtype=torch.long)

X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.replace({'male': 0, 'female': 1}).values, dtype=torch.long)

# Model oluşturma
class SVM(nn.Module):
    def __init__(self, input_dim):
        super(SVM, self).__init__()
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.fc(x)

# Modelin ve kayıp fonksiyonunun tanımlanması
input_dim = X_train.shape[1]
model = SVM(input_dim)
criterion = nn.BCEWithLogitsLoss()

# Optimizasyon algoritmasının seçilmesi
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Eğitim döngüsü
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs.squeeze(), y_train_tensor.float())
    loss.backward()
    optimizer.step()
    
    # Doğrulama seti üzerinde değerlendirme
    model.eval()
    with torch.no_grad():
        outputs_val = model(X_val_tensor)
        val_loss = criterion(outputs_val.squeeze(), y_val_tensor.float())
        predicted_val = torch.round(torch.sigmoid(outputs_val))
        val_accuracy = accuracy_score(y_val_tensor.numpy(), predicted_val.numpy())
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, Validation Loss: {val_loss.item():.4f}, Validation Accuracy: {val_accuracy * 100:.2f}%')

# Modelin test seti üzerinde değerlendirilmesi
model.eval()
with torch.no_grad():
    outputs_test = model(X_test_tensor)
    test_loss = criterion(outputs_test.squeeze(), y_test_tensor.float())
    predicted_test = torch.round(torch.sigmoid(outputs_test))
    test_accuracy = accuracy_score(y_test_tensor.numpy(), predicted_test.numpy())
    
print(f'Test Loss: {test_loss.item():.4f}, Test Accuracy: {test_accuracy * 100:.2f}%')
print('Test Classification Report:')
print(classification_report(y_test_tensor.numpy(), predicted_test.numpy(), target_names=['male', 'female']))
print('Test Confusion Matrix:')
print(confusion_matrix(y_test_tensor.numpy(), predicted_test.numpy()))

# Adım 6: Model ağırlıklarını kaydetme
torch.save(model.state_dict(), 'voice_model_weights.pth')
print("Model ağırlıkları başarıyla kaydedildi.")
