import os
import cv2
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import transforms, datasets
import matplotlib.pyplot as plt

REBUILD_DATA = False


class DogsVSCats:
    IMG_SIZE = 50
    CATS = "./Datasets/PetImages/Cat"
    DOGS = "./Datasets/PetImages/Dog"
    LABELS = {CATS: 0, DOGS: 1}

    training_data = []
    cat_count = 0
    dog_count = 0

    def make_training_data(self):
        for label in self.LABELS:
            print(label)

            for f in tqdm(os.listdir(label)):

                try:
                    path = os.path.join(label, f)

                    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                    img = cv2.resize(img, (self.IMG_SIZE, self.IMG_SIZE))

                    # Creates a one hot vector with the corresponding label
                    # np.eye(2)[self.LABELS[label]]
                    self.training_data.append([np.array(img), np.eye(2)[self.LABELS[label]]])

                    if label == self.CATS:
                        self.cat_count += 1
                    elif label == self.DOGS:
                        self.dog_count += 1

                except Exception as e:
                    pass
                # print(str(e))

        np.random.shuffle(self.training_data)
        np.save("./Datasets/PetImages/training_data.npy", self.training_data)
        print("Cats: ", self.cat_count)
        print("Dogs: ", self.dog_count)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 5)  # kernel size is 5x5
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.conv3 = nn.Conv2d(64, 128, 5)

        x = torch.randn(50, 50).view(-1, 1, 50, 50)
        self._to_linear = None
        self.convs(x)

        self.fc1 = nn.Linear(self._to_linear, 512)
        self.fc2 = nn.Linear(512, 2)

    def convs(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv3(x)), (2, 2))

        # print("X[0].shape: ", x[0].shape)
        if self._to_linear is None:
            self._to_linear = x[0].shape[0] * x[0].shape[1] * x[0].shape[2]

        return x

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self._to_linear)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return F.softmax(x, dim=1)


if REBUILD_DATA:
    dogsvcats = DogsVSCats()
    dogsvcats.make_training_data()


training_data = np.load("./Datasets/PetImages/training_data.npy", allow_pickle=True)
print(len(training_data))

# print(training_data[28])
# plt.imshow(training_data[28][0], cmap="gray")
# plt.show()

net = Net()

optimizer = optim.Adam(net.parameters(), lr=0.001)
loss_function = nn.MSELoss()

X = torch.Tensor([i[0] for i in training_data]).view(-1, 50, 50)
X = X/255.0  # Scaling the images from 0 to 255 to 0 to 1

y = torch.Tensor([i[1] for i in training_data])

VAL_PCT = 0.1
val_size = int(len(X)*VAL_PCT)
# print(val_size)

train_X = X[:-val_size]
train_y = y[:-val_size]

test_X = X[-val_size:]
test_y = y[-val_size:]

# print(len(train_X))
# print(len(test_X))


def train(net):
    BATCH_SIZE = 100
    EPOCHS = 1

    # Training the CNN
    for epoch in range(EPOCHS):
        loss, in_sample_acc = 0, 0
        for i in tqdm(range(0, len(train_X), BATCH_SIZE)):
            # print(i, i+BATCH_SIZE)
            batch_X = train_X[i: i+BATCH_SIZE].view(-1, 1, 50, 50)
            batch_y = train_y[i: i+BATCH_SIZE]

            net.zero_grad()
            outputs = net(batch_X)

            matches = [torch.argmax(i) == torch.argmax(j) for i, j in zip(outputs, batch_y)]
            in_sample_acc = matches.count(True) / len(matches)

            loss = loss_function(outputs, batch_y)
            loss.backward()

            # Does the update
            optimizer.step()

        print(loss)
        print("In-sample acc: ", round(in_sample_acc, 2))


def test(net):
    correct = 0
    total = 0

    # Testing the CNN
    with torch.no_grad():
        for i in tqdm(range(len(test_X))):
            real_class = torch.argmax(test_y[i])
            net_out = net(test_X[i].view(-1, 1, 50, 50))[0]
            predicted_class = torch.argmax(net_out)

            if predicted_class == real_class:
                correct += 1

            total += 1

    print("Accuracy: ", round(correct/total, 3))


def fwd_pass(X, y, train=False):
    if train:
        net.zero_grad()

    outputs = net(X)

    # we want in and out of sample accuracy
    # In-sample accuracy: This is the accuracy on the data we're actually feeding through the model for training.
    #                     This is the data that we're "fitting" against
    # Out-of-sample accuracy: This is the accuracy on data that we've set aside that the model will never see/fit
    #                         against
    matches = [torch.argmax(i) == torch.argmax(j) for i, j in zip(outputs, y)]
    acc = matches.count(True)/len(matches)

    loss = loss_function(outputs, y)

    if train:
        loss.backward()
        optimizer.step()

    return acc, loss


def test(size=32):
    random_start = np.random.randint(len(test_X)-size)
    X, y = test_X[random_start:random_start+size], test_y[random_start:random_start+size]

    with torch.no_grad():
        val_acc, val_loss = fwd_pass(X.view(-1, 1, 50, 50), y)
    return val_acc, val_loss

val_acc, val_loss = test(size=32)
print(val_acc, val_loss)
