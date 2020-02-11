import torch.nn as nn
import torch.utils.data
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable

input_size = 784        # Размеры изображения = 28 * 28 = 784
hidden_size = 500       # Количество узлов на скрытом слое
num_classes = 10        # Число классов на выходе
num_epochs = 5          # Количество тренировок всего набора данных
batch_size = 100        # Размер входных данных данных для одной итерации
learning_rate = 0.001   # Скорость конвергенции

# Загружаем тренировочный и тестовый наборы данных
train_dataset = dsets.FashionMNIST(
	root='./data',
	train=True,
	transform=transforms.ToTensor(),
	download=True
)

test_dataset = dsets.FashionMNIST(
	root='./data',
	train=False,
	transform=transforms.ToTensor()
)

# Загружаем данные в код
train_loader = torch.utils.data.DataLoader(
	dataset=train_dataset,
	batch_size=batch_size,
	shuffle=True  # Перемешивание элементов
)

test_loader = torch.utils.data.DataLoader(
	dataset=test_dataset,
	batch_size=batch_size,
	shuffle=False
)

class_names = ['Футболка / топ', "Шорты", "Свитер", "Платье", "Плащ", "Сандали", "Рубашка", "Кроссовок", "Сумка", "Ботинок"]


class Net(nn.Module):
	def __init__(self, input_size, hidden_size, num_classes):
		super(Net, self).__init__()  # Наследуемый родительским классом
		self.fc1 = nn.Linear(input_size, hidden_size)  # 1й связанный слой: 784 (данные входа) -> 500 (скрытый)
		self.relu = nn.ReLU()  # Нелинейный слой ReLU max(0, x)
		self.fc2 = nn.Linear(hidden_size, num_classes)  # 2й связанный слой: 500 (скрытый) -> 10 (вывод)

	def forward(self, x):  # Складывание слоёв
		out = self.fc1(x)
		out = self.relu(out)
		out = self.fc2(out)
		# print(out)
		return out


net = Net(input_size, hidden_size, num_classes)

# функция потерь
criterion = nn.CrossEntropyLoss()

# Оптимизатор
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
	for i, (images, labels) in enumerate(train_loader):  # Загрузка партии изображений с индексом, данными, классом
		images = Variable(images.view(-1, 28 * 28))  # Конвертация тензора в переменную: изменяем изображение вектора размером 784 на матрицу 28 *28
		labels = Variable(labels)
		optimizer.zero_grad()  # Инициализация скрытых масс до нулей
		outputs = net(images)  # Передний пропуск: определение выходного класса, данного изображения
		loss = criterion(outputs, labels)  # Определение потерь: разница между выходным классом и предварительно заданной переменной
		loss.backward()  # Обратный переход: определение параметра weight
		optimizer.step()  # Оптимизатор: обносление параметров веса в скрытых узлах


correct = 0
total = 0
for images, labels in test_loader:
	images = Variable(images.view(-1, 28 * 28))
	outputs = net(images)
	_, predicted = torch.max(outputs.data, 1)  # Выбор лучшего класса из выходных данных: класс с лучшим счётом
	total += labels.size(0)  # Увеличиваем суммарный счёт
	correct += (predicted == labels).sum()  # Увеличиваем корректный счёт

torch.save(net, 'fnn_model.pkl')
