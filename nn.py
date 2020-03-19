import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets,transforms

batch_size=200

train_loader=torch.utils.data.DataLoader(                  # load minist data, dont know the meaning of 'transforms.Normalize((0.1307,),(0.3081,))'
    datasets.MNIST('../data',train=True,download=True, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,),(0.3081,))
    ])),batch_size=batch_size,shuffle=True
)

test_loader=torch.utils.data.DataLoader(
    datasets.MNIST('../data',train=False,transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,),(0.3081,))
    ])),batch_size=batch_size,shuffle=True
)

# 创建一个Tensor时，使用requires_grad参数指定是否记录对其的操作，以便之后利用backward()方法进行梯度求解
w1,b1=torch.randn(200,784,requires_grad=True),torch.randn(200,requires_grad=True)
w2,b2=torch.randn(200,200,requires_grad=True),torch.randn(200,requires_grad=True)
w3,b3=torch.randn(10,200,requires_grad=True),torch.randn(10,requires_grad=True)

torch.nn.init.kaiming_normal_(w1)
torch.nn.init.kaiming_normal_(w2)
torch.nn.init.kaiming_normal_(w3)

def forward(input_tensor):
    a=input_tensor@w1.t()+b1   # torch.matmul(a,b) is the same as torch.matmul(b,a.t())
    a=F.relu(a)
    a=a@w2.t()+b2
    a=F.relu(a)
    a=a@w3.t()+b3
    a=F.relu(a)
    return a

learning_rate=0.03
optimizor=optim.SGD([w1,b1,w2,b2,w3,b3],lr=learning_rate)
criterion=nn.CrossEntropyLoss()

epochs=30
for epoch in range(epochs):
    for batch_id,(train_data,target) in enumerate(train_loader):
        train_data=train_data.view(-1,28*28)
        logits=forward(train_data)
        loss=criterion(logits,target)     # input the predicted value and the target
        optimizor.zero_grad()
        loss.backward()           # Tensor执行自身的backward()函数
        optimizor.step()

        if batch_id%100==0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_id * len(train_data), len(train_loader.dataset),
                       100. * batch_id / len(train_loader), loss.item()))

    test_loss=0
    accuracy=0

    for data,target in test_loader:
        data=data.view(-1,28*28)
        logits=forward(data)
        test_loss+=criterion(logits,target).item()
        pred=logits.data.max(1)[1]
        accuracy+=pred.eq(target.data).sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, accuracy, len(test_loader.dataset),
        100. * accuracy / len(test_loader.dataset)))
