import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, models, transforms
import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score
import os
import cv2
import torchvision.transforms as transforms


if __name__ == '__main__':
    # Veri yollarını ve diğer parametreleri tanımla
    train_dir = r"C:/Users/filizyigit/Desktop/Bitirme Ödevi/image data/train"
    test_dir = r"C:/Users/filizyigit/Desktop/Bitirme Ödevi/image data/test"
    val_dir = r"C:/Users/filizyigit/Desktop/Bitirme Ödevi/image data/validation"
    batch_size = 32
    num_epochs = 10
    learning_rate = 0.001
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Veri augmentasyonu ve ön işleme işlemlerini tanımla
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # Veri yükleyici oluştur
    image_datasets = {
        'train': datasets.ImageFolder(train_dir, data_transforms['train']),
        'val': datasets.ImageFolder(val_dir, data_transforms['val'])
    }
    dataloaders = {
        'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=batch_size, shuffle=True, num_workers=4),
        'val': torch.utils.data.DataLoader(image_datasets['val'], batch_size=batch_size, shuffle=False, num_workers=4)
    }

    # Sınıf etiketlerini al
    class_names = image_datasets['train'].classes

    # CNN Modeli tanımla
    model_conv = models.resnet18(pretrained=True)
    num_ftrs = model_conv.fc.in_features
    model_conv.fc = nn.Linear(num_ftrs, 2)
    model_conv = model_conv.to(device)

    # Loss fonksiyonu ve optimizer tanımla
    criterion = nn.CrossEntropyLoss()
    optimizer_conv = optim.SGD(model_conv.parameters(), lr=learning_rate, momentum=0.9)

    # CNN modelini eğit
    for epoch in range(num_epochs):
        for phase in ['train', 'val']:
            if phase == 'train':
                model_conv.train()
            else:
                model_conv.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer_conv.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model_conv(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer_conv.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(image_datasets[phase])
            epoch_acc = running_corrects.double() / len(image_datasets[phase])

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))


    # SVM Modeli tanımla ve eğit
    X_train = []
    y_train = []
    for inputs, labels in dataloaders['train']:
        inputs = inputs.numpy()
        labels = labels.numpy()
        X_train.extend(inputs)
        y_train.extend(labels)
    X_train = np.array(X_train)
    y_train = np.array(y_train)

    X_train = np.reshape(X_train, (X_train.shape[0], -1))

    svm_model = svm.SVC(kernel='linear')
    svm_model.fit(X_train, y_train)

    # Test veri seti üzerinde performansı değerlendir
    X_test = []
    y_test = []
    for inputs, labels in dataloaders['val']:
        inputs = inputs.numpy()
        labels = labels.numpy()
        X_test.extend(inputs)
        y_test.extend(labels)
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    X_test = np.reshape(X_test, (X_test.shape[0], -1))

    svm_predictions = svm_model.predict(X_test)
    svm_accuracy = accuracy_score(y_test, svm_predictions)
    print('SVM Accuracy: {:.4f}'.format(svm_accuracy))

    # CNN modeli üzerinde performansı değerlendir
    model_conv.eval()
    running_corrects = 0

    for inputs, labels in dataloaders['val']:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model_conv(inputs)
        _, preds = torch.max(outputs, 1)

        running_corrects += torch.sum(preds == labels.data)

    cnn_accuracy = running_corrects.double() / len(image_datasets['val'])
    print('CNN Accuracy: {:.4f}'.format(cnn_accuracy))


# Modeli yükle
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_conv = models.resnet18(pretrained=True)
num_ftrs = model_conv.fc.in_features
model_conv.fc = nn.Linear(num_ftrs, 2)
model_conv.load_state_dict(torch.load("image_weights.pth", map_location=device))
model_conv.eval()
model_conv = model_conv.to(device)

# Cinsiyet etiketleri
class_names = ['male', 'female']

# Dönüşümler
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Kamera bağlantısı
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Gri tonlamalı görüntü oluştur
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        # Yüz bölgesini çıkar
        face = frame[y:y+h, x:x+w]
        
        # Yüz bölgesini dönüştür
        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face_tensor = transform(face_rgb).unsqueeze(0).to(device)

        # Modelden tahmin al
        with torch.no_grad():
            outputs = model_conv(face_tensor)
            _, predicted = torch.max(outputs, 1)
            prediction = class_names[predicted.item()]

        # Kare ve cinsiyet etiketini ekle
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, prediction, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Görüntüyü ekranda göster
    cv2.imshow('Gender Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Temizleme işlemleri
cap.release()
cv2.destroyAllWindows()