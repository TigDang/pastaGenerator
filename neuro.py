import torch as torch
import components.makeDataset as datas
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# Оверрайтинг абстрактной модели нейросети под нашу задачу
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # Объявление слоёв, их размера
        self.fc1 = nn.Linear(datas.countOfIngrs, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)

    # Определение функции прохода вперед по слоям
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = nn.Flatten(1, -1)(x)
        return x


# Объявление и вывод экземляра нашей модели
net = Net()
print(net)

# Обёртывание данных в тензор, объявление тензора флажков
trainX = torch.Tensor(datas.dataset[:, :]).view(-1, datas.countOfIngrs)
Y = torch.ones(datas.countOfPastas)
Y=torch.cat((Y, torch.zeros(1)), 0)

# testX = torch.Tensor(datas.dataset[10:, :]).view(-1, datas.countOfIngrs)

# Обёртывание тензора в объект DataLoader, предназначенный для использования в модели
trainset = torch.utils.data.DataLoader(TensorDataset(trainX, Y[:]), batch_size=1, shuffle=True)
# testset = torch.utils.data.DataLoader(TensorDataset(testX, Y[10:]), batch_size=1, shuffle=False)

# Определение функции ошибки и LR
loss_function = nn.L1Loss()
optimizer = optim.Adam(net.parameters(), lr=0.00001)

# Процесс обучения
for epoch in range(5000):  # столько полных проходов по нашим данным
    for data in trainset:  # `data` это батч наших данных
        X, y = data  # X это батч свойств, y это батч целевых переменных.
        net.zero_grad()  # устанавливаем значение градиента в 0 перед вычислением функции потерь.
        output = net(X.view(-1, datas.countOfIngrs))  # передаем выпрямленный батч
        loss = loss_function(output, y)  # вычисляем функцию потерь
        loss.backward()  # передаем это значение назад по сети
        optimizer.step()  # пытаемся оптимизировать значение весов исходя из потерь и градиента
    print(loss)  # выводим на экран значение функции потерь


# Цикл вывода по отправке символа с клавиатуры нового рецепта пасты по критерию
while input() == 'c':
    randomRecipe = torch.Tensor(datas.GetRandomRecipe(datas.countOfIngrs + 1)).view(-1, datas.countOfIngrs)
    ans = net.forward(x=randomRecipe)
    # Критерий вывода в негативном формате
    while ans > 1.000001:
        print(ans)
        print('... and his recipe is ' + str(datas.GetInterpretArray(datas.c, randomRecipe.numpy().transpose().astype(int))))

        randomRecipe = torch.Tensor(datas.GetRandomRecipe(datas.countOfIngrs + 1)).view(-1, datas.countOfIngrs)
        ans = net.forward(x=randomRecipe)

    print(ans)
    print(datas.GetInterpretArray(datas.c, randomRecipe.numpy().transpose().astype(int)))
