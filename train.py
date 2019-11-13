import torch
import torch.utils.data as utils
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from time import time

training_data = np.load("training_data.npz")

positions = torch.tensor(training_data["positions"], dtype = torch.float)
winners = torch.tensor(training_data["winners"], dtype = torch.float)

dataset = utils.TensorDataset(positions, winners)
dataloader = utils.DataLoader(dataset)

input_size = np.prod(positions.size()[1:])

class ConvNet(nn.Module):
    def __init__(self, hidden_layers, breadth):
        super(ConvNet, self).__init__()
        self.num_hidden = hidden_layers
        self.conv = nn.ModuleList([nn.Conv2d(2, breadth, 3)] +
                                  [nn.Conv2d(breadth, breadth, 3)
                                   for _ in range(hidden_layers - 1)])
        self.fc_out = nn.ModuleList([nn.Linear(breadth, 1)] +
                                    [nn.Linear(breadth, 1, bias=False) for _ in range(hidden_layers - 1)])

        self.fc_out[0].bias.data.fill_(0)
        for i in range(hidden_layers):
            self.fc_out[i].weight.data.fill_(0)

        self.padding = nn.ConstantPad2d((1, 1, 1, 1), 0)

    def forward(self, x):
        out = 0
        for i in range(self.num_hidden):
            #x = torch.tanh(self.conv[i](self.padding(x)))
            x = self.conv[i](self.padding(x))
            x = F.relu(x)
            #x = x * torch.sigmoid(x)
            out += self.fc_out[i](torch.sum(x, (2, 3)))
        return out


class FCNet(nn.Module):
    def __init__(self, breadth):
        super(FCNet, self).__init__()
        self.fc1 = nn.Linear(input_size, breadth)
        self.fc2 = nn.Linear(breadth, breadth)
        self.fc3 = nn.Linear(breadth, breadth)
        self.fc4 = nn.Linear(breadth, 1)

    def forward(self, x):
        x = x.view(-1, input_size)
        x = torch.tanh(self.fc1(x))
        #x = F.relu(self.fc2(x))
        #x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

net = ConvNet(5, 40)
#net = FCNet(100)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(net.parameters())

correct = 0
total = 0

start_time = time()

for epoch in range(10):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(dataloader):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)

        total += labels.size()[0]
        correct += ((labels == 1) == (outputs > 0.5)).sum().item()

        loss = criterion(outputs.view(-1), labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print(time() - start_time)
            last_time = time()
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0
            print(correct, total, correct / total)
            correct = 0
            total = 0

