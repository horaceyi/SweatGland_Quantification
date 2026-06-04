import numpy as np
import torch
import torch.nn as nn
import Model_Architecture
import os
from torch import FloatTensor, optim, LongTensor
from dataset import LiverDataset
from torch.utils.data import DataLoader
from CustomLoss import DiceLoss
import matplotlib.pyplot as plt

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def map_labels(labels):
    temp = labels.clone()
    labels[temp == 1] = 0
    labels[temp == 2] = 1
    labels[temp > 2] = 0
    return labels


def train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=150, save_weights_path="",
                error_path=""):
    hist = np.zeros([num_epochs, 3])
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60, 90, 120], gamma=0.5)

    for _epoch in range(num_epochs):
        epoch = _epoch + 1
        print(f'Epoch {epoch}/{num_epochs}')

        model.train()
        epoch_loss = 0
        for x, y in train_loader:
            optimizer.zero_grad()
            inputs = x.to(device).float()
            labels = map_labels(torch.squeeze(y.to(device).long()))

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        epoch_loss /= len(train_loader)

        model.eval()
        val_epoch_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                inputs = x.to(device).float()
                labels = map_labels(torch.squeeze(y.to(device).long()))
                outputs = model(inputs)
                val_epoch_loss += criterion(outputs, labels).item()
        val_epoch_loss /= len(val_loader)
        
        scheduler.step()
        print(f"Loss: {epoch_loss:.4f} | Val_Loss: {val_epoch_loss:.4f}")
        hist[_epoch, :] = [epoch, epoch_loss, val_epoch_loss]

    torch.save({"net": model.state_dict()}, save_weights_path)
    np.savetxt(error_path, hist, delimiter=",")
    return model


def test_model(model, criterion, test_loader, save_weights_path):
    checkpoint = torch.load(save_weights_path, map_location=device)
    model.load_state_dict(checkpoint['net'])
    model.eval()

    os.makedirs("train_result/test_result", exist_ok=True)
    with torch.no_grad():
        for i, (x, y) in enumerate(test_loader):
            inputs = x.to(device).float()
            labels = map_labels(torch.squeeze(y.to(device).long()))
            outputs = model(inputs)

            if i < 5:
                pred = torch.argmax(outputs, dim=1)[0].cpu().numpy()
                plt.imsave(f"train_result/test_result/res_{i}.png", pred, cmap='gray')


def train_entry():
    model = Model_Architecture.DepthNet(3, 2, 32).to(device)
    data_path = r"...\train_data" # Modify here

    for p in ["train_result/weights", "train_result/error"]:
        os.makedirs(p, exist_ok=True)

    dataset = LiverDataset(data_path)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = DiceLoss()

    save_path = ".../train_result/weights/final_model.pth" # Modify here
    train_model(model, criterion, optimizer, train_loader, val_loader, save_weights_path=save_path,
                error_path=".../train_result/error/log.csv") # Modify here
    test_model(model, criterion, val_loader, save_path)


if __name__ == '__main__':
    train_entry()
