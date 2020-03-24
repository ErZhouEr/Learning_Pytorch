import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import datasets,transforms

# get data

batch_size=200

train_loader=torch.utils.data.DataLoader(
    datasets.MNIST('../data',train=True,transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,),(0.3081,))
    ]),download=True), batch_size=batch_size,shuffle=True
)

test_loader=torch.utils.data.DataLoader(
    datasets.MNIST('../data',train=False,transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,),(0.3081,))
    ])), batch_size=batch_size,shuffle=True
)

# class NN

class MLP(nn.Module):

    def __init__(self):
        super(MLP,self).__init__()

        self.model=nn.Sequential(
            nn.Linear(784,200),
            nn.ReLU(inplace=True),
            nn.Linear(200,200),
            nn.ReLU(inplace=True),
            nn.Linear(200,10),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x=self.model(x)
        # x.requires_grad=True
        return x

net=MLP()

learning_rate=0.01
optimizor=optim.SGD(net.parameters(),lr=learning_rate)

criterion=nn.CrossEntropyLoss()

epochs=30
for epoch in range(epochs):
    for idx,(train_data,target) in enumerate(train_loader):
        data=train_data.view(-1,28*28)
        # print(data.requires_grad)
        logits=net.forward(data)
        # print(logits.requires_grad)
        loss=criterion(logits,target)
        # print(loss.requires_grad)

        optimizor.zero_grad()
        loss.backward()
        optimizor.step()

        if idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, idx * len(train_data), len(train_loader.dataset),
                       100. * idx / len(train_loader), loss.item()))

    test_loss = 0
    accuracy = 0

    for data, target in test_loader:
        data = data.view(-1, 28 * 28)
        logits = net.forward(data)
        test_loss += criterion(logits, target).item()
        pred = logits.data.max(1)[1]
        accuracy += pred.eq(target.data).sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, accuracy, len(test_loader.dataset),
        100. * accuracy / len(test_loader.dataset)))
