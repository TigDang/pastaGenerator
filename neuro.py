import torch as torch
import components.makeDataset as datas
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


# Гиперпараметры
EPOCHS = 500
BATCH_SIZE = 64
LEARNING_RATE = 0.1

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Оверрайтинг абстрактной модели нейросети под нашу задачу
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Объявление слоёв, их размера
        self.layer_1 = nn.Linear(datas.countOfIngrs, 64)
        self.layer_2 = nn.Linear(64, 64)
        self.layer_out = nn.Linear(64, 1)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        self.batchnorm1 = nn.BatchNorm1d(64)
        self.batchnorm2 = nn.BatchNorm1d(64)

    # Определение функции прохода вперед по слоям
    def forward(self, inputs):
        x = self.layer_1(inputs)
        x = self.batchnorm1(x)
        x = self.relu(self.layer_2(x))
        x = self.batchnorm2(x)
        x = self.dropout(x)
        x = torch.sigmoid(self.layer_out(x))

        return x


def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))

    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum / y_test.shape[0]
    acc = torch.round(acc * 100)

    return acc


# Объявление и вывод экземляра нашей модели
model = Net()
model.to(device)
print(model)

# Обёртывание данных в тензор, объявление тензора флажков
trainX = torch.Tensor(datas.dataset[:, :]).view(-1, datas.countOfIngrs)
Y = torch.ones(datas.countOfPastas)
Y=torch.cat((Y, torch.zeros(1)), 0)

# Обёртывание тензора в объект DataLoader, предназначенный для использования в модели
train_loader = DataLoader(dataset=TensorDataset(trainX, Y[:]), batch_size=BATCH_SIZE, shuffle=True)

# Определение функции ошибки и LR
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

model.train()
for e in range(1, EPOCHS + 1):
    epoch_loss = 0
    epoch_acc = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        # Обнуляем градиент перед работой
        optimizer.zero_grad()

        # Получем предсказания модели
        y_pred = model(X_batch)

        # Считаем величину ошибки предсказания относительно датасета
        loss = criterion(y_pred, y_batch.unsqueeze(1))
        acc = binary_acc(y_pred, y_batch.unsqueeze(1))

        # Распространяем ошибку назад по сети и оптимизируем
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    print(f'Epoch {e + 0:03}: | Loss: {epoch_loss / len(train_loader):.5f} | Acc: {epoch_acc / len(train_loader):.3f}')

# Проверка работы модели на обучающей выборке. Должны быть выведены в основном единицы
ansOfPastas = []
model.eval()
for i in range(datas.countOfPastas):
    a = model(torch.Tensor(datas.dataset[i, :]).view(-1, datas.countOfIngrs))
    ansOfPastas.append(a.cpu().detach().numpy()[0][0])
    print('{} have output is'.format(i) + str(torch.round(a)))

# Цикл вывода ста сгенерированных паст
for t in range(100):
    # Оборачиваем случайный рецепт в тензор, помещаем на вход модели и записываем результат в ans
    randomRecipe = torch.Tensor(datas.GetRandomRecipe(datas.countOfIngrs + 1)).view(-1, datas.countOfIngrs)
    model.eval()
    ans = torch.round(model.forward(inputs=randomRecipe))
    # Критерий вывода в негативном формате (пока модель не скажет, что рецепт - паста, делаем новый рецепт
    while ans != 1:
        randomRecipe = torch.Tensor(datas.GetRandomRecipe(datas.countOfIngrs + 1)).view(-1, datas.countOfIngrs)
        ans = torch.round(model.forward(inputs=randomRecipe))

    # print(ans)
    print(datas.GetInterpretArray(datas.c, randomRecipe.numpy().transpose().astype(int)))
