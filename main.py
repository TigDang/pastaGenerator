import torch as torch
import components.makeDataset as datas
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

X = datas.dataset[:, :]
y = torch.ones(datas.countOfPastas)
y = torch.cat((y, torch.zeros(1)), 0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=69)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

EPOCHS = 50
BATCH_SIZE = 64
LEARNING_RATE = 0.001


class TrainData(Dataset):

    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

    def __len__(self):
        return len(self.X_data)


train_data = TrainData(torch.FloatTensor(X_train),
                       torch.FloatTensor(y_train))


## test data
class TestData(Dataset):

    def __init__(self, X_data):
        self.X_data = X_data

    def __getitem__(self, index):
        return self.X_data[index]

    def __len__(self):
        return len(self.X_data)


test_data = TestData(torch.FloatTensor(X_test))

train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=1)


class BinaryClassification(nn.Module):
    def __init__(self):
        super(BinaryClassification, self).__init__()
        # Number of input features is 12.
        self.layer_1 = nn.Linear(datas.countOfIngrs, 64)
        self.layer_2 = nn.Linear(64, 64)
        self.layer_out = nn.Linear(64, 1)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        self.batchnorm1 = nn.BatchNorm1d(64)
        self.batchnorm2 = nn.BatchNorm1d(64)

    def forward(self, inputs):
        x = self.relu(self.layer_1(inputs))
        x = self.batchnorm1(x)
        x = self.relu(self.layer_2(x))
        x = self.batchnorm2(x)
        x = self.dropout(x)
        x = self.layer_out(x)

        return x

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

model = BinaryClassification()
model.to(device)
print(model)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))

    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum / y_test.shape[0]
    acc = torch.round(acc * 100)

    return acc


model.train()
for e in range(1, EPOCHS + 1):
    epoch_loss = 0
    epoch_acc = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()

        y_pred = model(X_batch)

        loss = criterion(y_pred, y_batch.unsqueeze(1))
        acc = binary_acc(y_pred, y_batch.unsqueeze(1))

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    print(f'Epoch {e + 0:03}: | Loss: {epoch_loss / len(train_loader):.5f} | Acc: {epoch_acc / len(train_loader):.3f}')


y_pred_list = []
model.eval()
with torch.no_grad():
    for X_batch in test_loader:
        X_batch = X_batch.to(device)
        y_test_pred = model(X_batch)
        y_test_pred = torch.sigmoid(y_test_pred)
        y_pred_tag = torch.round(y_test_pred)
        y_pred_list.append(y_pred_tag.cpu().numpy())

y_pred_list = [a.squeeze().tolist() for a in y_pred_list]

confusion_matrix(y_test, y_pred_list)

print(classification_report(y_test, y_pred_list))

ansOfPastas = []
model.eval()
for i in range(datas.countOfPastas):
    a = model(torch.Tensor(datas.dataset[i, :]).view(-1, datas.countOfIngrs))
    ansOfPastas.append(a.cpu().detach().numpy()[0][0])
    print('{} have output is'.format(i) + str(a))

# Цикл вывода по отправке символа с клавиатуры нового рецепта пасты по критерию
while input() == 'c':
    # Оборачиваем случайный рецепт в тензор, помещаем на вход модели и записываем результат в ans
    randomRecipe = torch.Tensor(datas.GetRandomRecipe(datas.countOfIngrs + 1)).view(-1, datas.countOfIngrs)
    model.eval()
    ans = model.forward(inputs=randomRecipe)
    # Критерий вывода в негативном формате
    # while not(min(ansOfPastas, key=lambda i: float(i)) < ans < max(ansOfPastas, key=lambda i: float(i))):
    while torch.round(a) < 0:
        print(ans)
        print('... and his recipe is ' + str(datas.GetInterpretArray(datas.c, randomRecipe.numpy().transpose().astype(int))))

        randomRecipe = torch.Tensor(datas.GetRandomRecipe(datas.countOfIngrs + 1)).view(-1, datas.countOfIngrs)
        ans = torch.round(model.forward(inputs=randomRecipe))

    print(ans)
    print(datas.GetInterpretArray(datas.c, randomRecipe.numpy().transpose().astype(int)))