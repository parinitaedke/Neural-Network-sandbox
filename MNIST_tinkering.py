import torch
import torchvision
from torchvision import transforms, datasets
import matplotlib as plt
from matplotlib import pyplot

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

train = datasets.MNIST("./Datasets/", train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))

test = datasets.MNIST("./Datasets/", train=False, download=True, transform=transforms.Compose([transforms.ToTensor()]))

trainset = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True)
testset = torch.utils.data.DataLoader(test, batch_size=10, shuffle=True)

# # Just trying to print a data image
# for data in trainset:
#     print(data)
#     break
#
# x, y = data[0][0], data[1][0]
# print(y)
#
# pyplot.imshow(x.view(28, 28))
# pyplot.show()


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # self.fc1 = nn.Linear(input output)
        # input = 28 x 28 = 784
        # output
        self.fc1 = nn.Linear(784, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)

        return F.log_softmax(x, dim=1)


net = Net()
# print(net)

optimizer = optim.Adam(net.parameters(), lr=0.001)

EPOCHS = 3

for epoch in range(EPOCHS):
    for data in trainset:
        # data is a batch of feature sets and labels
        X, y = data

        net.zero_grad()

        # the -1 says be prepared for any amount of data to be passed through
        output = net(X.view(-1, 28*28))
        loss = F.nll_loss(output, y)

        # backwards propagation
        loss.backward()

        # adjusts the network weights
        optimizer.step()

    print(loss)

correct = 0
total = 0

# validating data by running the model without calculating gradients
with torch.no_grad():
    for data in trainset:
        X, y = data

        output = net(X.view(-1, 28*28))
        # print("***********************************")
        # print(output)
        for idx, i in enumerate(output):
            if torch.argmax(i) == y[idx]:
                correct += 1
            total += 1

print("Accuracy: ", round(correct/total, 3))

print(torch.argmax(net(X[3].view(-1, 784))[0]))
pyplot.imshow(X[3].view(28,28))
pyplot.show()


