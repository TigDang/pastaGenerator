import torch as torch
import components.makeDataset as datas
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(datas.countOfIngrs, 1000)
        self.fc2 = nn.Linear(1000, 1000)
        self.fc3 = nn.Linear(1000, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)


net = Net()

trainX = torch.Tensor(datas.dataset[:10, :])
Y = torch.ones(datas.countOfPastas)
testX = torch.Tensor(datas.dataset[10:, :])

trainX = trainX.type(torch.float)
Y = Y.type(torch.float)
testX = testX.type(torch.float)

trainset = torch.utils.data.DataLoader(TensorDataset(trainX, Y[:10]), batch_size=10, shuffle=True)
testset = torch.utils.data.DataLoader(TensorDataset(testX, Y[10:]), batch_size=10, shuffle=False)

loss_function = nn.L1Loss()
optimizer = optim.Adam(net.parameters(), lr=1)

for epoch in range(30):  # три полных прохода по нашим данным
    for data in trainset:  # `data` это батч наших данных
        X, y = data  # X это батч свойств, y это батч целевых переменных.
        net.zero_grad()  # устанавливаем значение градиента в 0 перед вычислением функции потерь. Вам следует делать это на каждом шаге.
        output = net(X.view(-1, datas.countOfIngrs))  # передаем выпрямленный батч
        loss = loss_function(output, y)  # вычисляем функцию потерь
        loss.backward()  # передаем это значение назад по сети
        optimizer.step()  # пытаемся оптимизировать значение весов исходя из потерь и градиента
    print(loss)  # выводим на экран значение функции потерь. Мы надеемся, что оно убывает!
