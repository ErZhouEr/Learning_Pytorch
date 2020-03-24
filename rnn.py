import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

input_size=1
hidden_size=10
output_size=1


class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()

        self.rnn=nn.RNN(
            input_size=1,
            hidden_size=10,
            num_layers=1,
            batch_first=True   # make batch as the first param
        )
        self.linear=nn.Linear(hidden_size,output_size)

    def forward(self, x,hidden_prev):
        output,hidden_prev=self.rnn(x,hidden_prev)
        # print('output: ',output.shape,'hidden_size: ',hidden_prev.shape)
        output=output.view(-1,hidden_size)
        # print('output2: ', output.shape)
        output=self.linear(output)
        # print('output3: ', output.shape)
        output=output.unsqueeze(dim=0)
        # print('output4: ', output.shape)
        return output,hidden_prev


model=Net()

learning_rate=0.01
optimizer=optim.Adam(model.parameters(),lr=learning_rate)
criterion=nn.MSELoss()

hidden_prev=torch.zeros(1,1,hidden_size)   # [batch, num_layer, hidden_size]

epochs=6000
for epoch in range(epochs):
    time_steps=50     # 50-1 is the sequence length
    start = np.random.randint(5)
    datalst = np.linspace(start, start + 10, time_steps)
    data = np.sin(datalst)
    x = torch.from_numpy(data[:-1]).float().view(1,time_steps-1,1)    # [batch, seq_len, vec_size]
    y = torch.from_numpy(data[1:]).float().view(1,time_steps-1,1)

    output,hidden_prev=model.forward(x,hidden_prev)
    hidden_prev=hidden_prev.detach()

    loss=criterion(output,y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch%100==0:
        print(f'epoch: {epoch},loss: {loss}')

start = np.random.randint(5)
datalst = np.linspace(start, start + 10, time_steps)
data = np.sin(datalst)
x = torch.from_numpy(data[:-1]).float().view(1,time_steps-1,1)    # [batch, seq_len, vec_size]
y = torch.from_numpy(data[1:]).float().view(1,time_steps-1,1)

n=x.shape[1]
predict=[]
input=x[:,0,:]
print('input ',input)
for _ in range(n):
    input=input.view(1,1,1)
    output,hidden_prev=model(input,hidden_prev)
    input=output
    predict.append(output.detach().numpy().ravel()[0])

x=x.data.numpy().ravel()
y=y.data.numpy()
plt.scatter(datalst[:-1],x,s=90)
plt.plot(datalst[:-1],x)

plt.scatter(datalst[1:],predict)
plt.show()

