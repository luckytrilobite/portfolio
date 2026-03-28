import torch

import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.utils.data as data
import torch.optim as optim
import torch.nn as nn

from model import LeNet
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
import time
import copy

device = torch.device('cpu')

data_transform = transforms.Compose([
    transforms.RandomRotation(0),         
    transforms.ToTensor(),
    transforms.Lambda(lambda x: torch.flip(x, [1])),          
    transforms.Lambda(lambda x: torch.rot90(x, k=3, dims=[1, 2])),  
    transforms.Normalize((0.1307,), (0.3081,))
])

output_dim = 10
model = LeNet(output_dim).to(device)

train_data = datasets.EMNIST(root='./LeNet/data', split='digits', train=True, transform=data_transform, download=True)
test_data = datasets.EMNIST(root='./LeNet/data', split='digits', train=False, transform=data_transform, download=True)

valid_ratio = 0.9
n_train_examples = int(len(train_data) * valid_ratio)
n_valid_examples = len(train_data) - n_train_examples

train_data, valid_data = data.random_split(train_data, [n_train_examples, n_valid_examples])

batch_size = 64
train_loader = data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
valid_loader = data.DataLoader(valid_data, batch_size=batch_size)
test_loader = data.DataLoader(test_data, batch_size=batch_size)

learning_rate = 1e-3
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
loss_func = nn.CrossEntropyLoss()

def calculate_accuracy(y_pred, y):
    top_pred = y_pred.argmax(1, keepdim=True)
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    acc = correct.float() / y.shape[0]
    return acc

def train(model, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0
    model.train()
    for (x, y) in tqdm(iterator, desc="Training", leave=False):
        x = x.to(device)
        y = y.to(device)
        y_pred = model(x)
        loss = criterion(y_pred, y)
        acc = calculate_accuracy(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc += acc.item()
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def evaluate(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0
    model.eval()
    with torch.no_grad():
        for (x, y) in tqdm(iterator, desc="Evaluating", leave=False):
            x = x.to(device)
            y = y.to(device)
            y_pred = model(x)
            loss = criterion(y_pred, y)
            acc = calculate_accuracy(y_pred, y)
            epoch_loss += loss.item()
            epoch_acc += acc.item()
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

epochs = 5
best_valid_loss = float('inf')
train_acc_list = []
valid_acc_list = []
train_loss_list = []
valid_loss_list = []

for epoch in trange(epochs, desc="Epochs"):
    start_time = time.monotonic()
    train_loss, train_acc = train(model, train_loader, optimizer, loss_func)
    valid_loss, valid_acc = evaluate(model, valid_loader, loss_func)

    train_acc_list.append(train_acc)
    valid_acc_list.append(valid_acc)
    train_loss_list.append(train_loss)
    valid_loss_list.append(valid_loss)

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'LeNet.pt')

    end_time = time.monotonic()
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    print(f'\nEpoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
    print(f'\tValid Loss: {valid_loss:.3f} | Valid Acc: {valid_acc*100:.2f}%')

model.load_state_dict(torch.load('LeNet.pt'))
model = model.to(device)
test_loss, test_acc = evaluate(model, test_loader, loss_func)
print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')

plt.plot(train_acc_list)
plt.plot(valid_acc_list)
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Valid'], loc='upper left') 
plt.show()

plt.plot(train_loss_list)
plt.plot(valid_loss_list)
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Valid'], loc='upper left')
plt.show()
