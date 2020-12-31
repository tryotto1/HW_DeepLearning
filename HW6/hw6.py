import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt
import numpy as np
import time


########################################
# You can define whatever classes if needed
########################################

class View(nn.Module):
    def __init__(self, *shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(x.shape[0], *self.shape)


class BlockInsideStage(nn.Module):
    def __init__(self, input_channel, output_channel, kern_size, stride_size, pad_size):
        super(BlockInsideStage, self).__init__()

        modules = []

        modules.append(nn.BatchNorm2d(num_features=input_channel))
        modules.append(nn.ReLU(inplace=True))
        modules.append(nn.Conv2d(in_channels=input_channel, out_channels=input_channel,
                                      kernel_size=kern_size, stride=stride_size, padding=pad_size))
        modules.append(nn.BatchNorm2d(num_features=input_channel))
        modules.append(nn.ReLU(inplace=True))
        modules.append(nn.Conv2d(in_channels=input_channel, out_channels=input_channel,
                                      kernel_size=kern_size, stride=stride_size, padding=pad_size))
        self.sequential = nn.Sequential(*modules)

    def forward(self, input_data):
        return input_data + self.sequential(input_data)


class BlockInsideStageHalf(nn.Module):
    def __init__(self, input_channel, output_channel, kern_size, stride_size, pad_size):
        super(BlockInsideStageHalf, self).__init__()

        modules = []
        modules_identity = []

        # 일반
        modules.append(nn.BatchNorm2d(num_features=input_channel))
        modules.append(nn.ReLU(inplace=True))
        modules.append(nn.Conv2d(in_channels=input_channel, out_channels=output_channel
                                      , kernel_size=kern_size, stride=2, padding=pad_size))
        modules.append(nn.BatchNorm2d(num_features=output_channel))
        modules.append(nn.ReLU(inplace=True))
        modules.append(nn.Conv2d(in_channels=output_channel, out_channels=output_channel
                                      , kernel_size=kern_size, stride=stride_size, padding=pad_size))
        self.sequential = nn.Sequential(*modules)

        # identity용
        modules_identity.append(nn.BatchNorm2d(num_features=input_channel))
        modules_identity.append(nn.ReLU(inplace=True))
        modules_identity.append(nn.Conv2d(in_channels=input_channel, out_channels=output_channel,
                                               kernel_size=1, stride=2, padding=0))
        self.identity = nn.Sequential(*modules_identity)

    def forward(self, input_data):
        return self.sequential(input_data) + self.identity(input_data)


class IdentityResNet(nn.Module):
    # __init__ takes 4 parameters
    # nblk_stage1: number of blocks in stage 1, nblk_stage2.. similar
    def __init__(self, nblk_stage1, nblk_stage2, nblk_stage3, nblk_stage4):
        super(IdentityResNet, self).__init__()

        layers = []

        # stage0 : conv 3x3
        layers.append(nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1))

        # stage1
        for cnt in range(nblk_stage1):
            tmp_block = BlockInsideStage(input_channel=64, output_channel=64, kern_size=3, stride_size=1, pad_size=1)
            layers.append(tmp_block)

        # stage2
        layers.append(
            BlockInsideStageHalf(input_channel=64, output_channel=128, kern_size=3, stride_size=1, pad_size=1))
        for cnt in range(nblk_stage2 - 1):
            layers.append(
                BlockInsideStage(input_channel=128, output_channel=128, kern_size=3, stride_size=1, pad_size=1))

        # stage3
        layers.append(
            BlockInsideStageHalf(input_channel=128, output_channel=256, kern_size=3, stride_size=1, pad_size=1))
        for cnt in range(nblk_stage3 - 1):
            layers.append(
                BlockInsideStage(input_channel=256, output_channel=256, kern_size=3, stride_size=1, pad_size=1))

        # stage4
        layers.append(
            BlockInsideStageHalf(input_channel=256, output_channel=512, kern_size=3, stride_size=1, pad_size=1))
        for cnt in range(nblk_stage4 - 1):
            layers.append(
                BlockInsideStage(input_channel=512, output_channel=512, kern_size=3, stride_size=1, pad_size=1))

        # avg pool + fc
        layers.append(nn.AdaptiveAvgPool2d((1, 1)))
        layers.append(View(-1))
        layers.append(nn.Linear(in_features=512, out_features=10))

        # 등록
        self.network = nn.Sequential(*layers)
        print(self.network)

    ########################################
    # Implement the network
    # You can declare whatever variables
    ########################################

    ########################################
    # You can define whatever methods
    ########################################

    def forward(self, x):
        ########################################
        # Implement the network
        # You can declare or define whatever variables or methods
        ########################################

        return self.network(x)


########################################
# Q1. set device
# First, check availability of GPU.
# If available, set dev to "cuda:0";
# otherwise set dev to "cpu"
########################################

# Assuming that we are on a CUDA machine, this should print a CUDA device:
dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('current device: ', dev)

########################################
# data preparation: CIFAR10
########################################

########################################
# Q2. set batch size
# set batch size for training data
########################################
batch_size = 10
PATH = './checkpoint'

# preprocessing
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# load training data
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True)

# load test data
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# define network
net = IdentityResNet(nblk_stage1=2, nblk_stage2=2,
                     nblk_stage3=2, nblk_stage4=2)

########################################
# Q3. load model to GPU
# Complete below to load model to GPU
########################################
# net.load_state_dict(torch.load(PATH))
net.to(dev)

# set loss function
criterion = nn.CrossEntropyLoss()

########################################
# Q4. optimizer
# Complete below to use SGD with momentum (alpha= 0.9)
# set proper learning rate
########################################
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# start training
t_start = time.time()

for epoch in range(5):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(dev), data[1].to(dev)

        ########################################
        # Q5. make sure gradients are zero!
        # zero the parameter gradients
        ########################################
        optimizer.zero_grad()

        ########################################
        # Q6. perform forward pass
        ########################################
        outputs = net(inputs)

        # set loss
        loss = criterion(outputs, labels)

        ########################################
        # Q7. perform backprop
        ########################################
        loss.backward()

        ########################################
        # Q8. take a SGD step
        ########################################
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        print(str(i) + "  " + str(loss.item()))
        if i % 2000 == 1999:  # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0
            t_end = time.time()
            print('elapsed:', t_end - t_start, ' sec')
            t_start = t_end
    torch.save(net.state_dict(), PATH)

print('Finished Training')

# now testing
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))

########################################
# Q9. complete below
# when testing, computation is done without building graphs
########################################
with torch.no_grad():
    for data in testloader:
        images, labels = data[0].to(dev), data[1].to(dev)
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

# per-class accuracy
for i in range(10):
    print('Accuracy of %5s' % (classes[i]), ': ',
          100 * class_correct[i] / class_total[i], '%')

# overall accuracy
print('Overall Accurracy: ', (sum(class_correct) / sum(class_total)) * 100, '%')


