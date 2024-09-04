import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
import numpy as np
import os

if __name__ == '__main__':
    # Veri yollarını ve diğer parametreleri tanımla
    train_dir = r"C:/Users/filizyigit/Desktop/Bitirme Ödevi/image data/train"
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

    # Model ağırlıklarını kaydet
    torch.save(model_conv.state_dict(), 'model_weights.pth')
    print("Model weights saved successfully!")
