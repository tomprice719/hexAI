import torch
import torch.utils.data as utils
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

training_data = np.load("training_data.npz")

positions = torch.from_numpy(training_data["positions"])
winners = torch.from_numpy(training_data["winners"])

dataset = utils.TensorDataset(positions, winners)
dataloader = utils.DataLoader(dataset)

input_size = np.prod(positions.size()[1:])


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 100)
        self.fc4 = nn.Linear(100, 1)

    def forward(self, x):
        x = x.view(-1, input_size)
        x = F.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x

net = Net()

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(net.parameters())

correct = 0
total = 0

for epoch in range(10):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(dataloader):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)

        total += labels.size[0]
        correct += ((labels == 1) == (outputs > 0.5)).sum().item()

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print(correct, total, correct / total)

