import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
import sys
from untitled.t_est import t_est1
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from time import time
from torch.autograd import Variable
from PIL import Image
import time as tim
import cv2
from untitled.RBT import getRotateLabelx,get_degree,getRotateLabely,RGBT
from PIL import Image
import cv2
import math
import torch.functional
import torch.utils.model_zoo as model_zoo

def conv3x3(in_planes, out_planes, stride = 1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size = 3, stride = stride, padding = 1,bias = False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet_2(nn.Module):
    def __init__(self, block, layers, num_classes=8):
        self.inplanes = 64
        super(ResNet_2, self).__init__()
        #input = 3*64*128
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,#class torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
                                                                                        #二维卷积层, 输入的尺度是(N, C_in,H,W)，输出尺度（N,C_out,H_out,W_out）的计算方式：
                               bias=False)#output = 32*32*64
        self.bn1 = nn.BatchNorm2d(64)#output = 32*32*64
        #上面的参数修改过了
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)#output = 32*32*64
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.layer5 = self._make_layer(block, 1024, layers[4], stride=2)
        self.avgpool = nn.MaxPool2d(2, stride=2)
        self.fc1 = nn.Linear(2048, 1024)
        # self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(1024,2*4)
        self.sigmoid = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
#        print(np.shape(x))
#         x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
#        print(np.shape(x))

        x = self.layer1(x)
 #       print(np.shape(x))
        x = self.layer2(x)
#        print(np.shape(x))
        x = self.layer3(x)
#        print(np.shape(x))
        x = self.layer4(x)
#        print(np.shape(x))
        x = self.layer5(x)
#        print(np.shape(x))
        x = self.avgpool(x)
#        print(np.shape(x))


        x = x.view(x.size(0), -1)
#        print(np.shape(x))
        x = self.fc1(x)
#        print(np.shape(x))
        x = self.relu(x)
        # x = self.dropout(x)
        x = self.fc2(x)

        return x

def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_2(BasicBlock, [2, 2, 2, 2 ,2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url('https://download.pytorch.org/models/resnet18-5c106cde.pth'))
    return model


class LandmarksDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir,
                                self.landmarks_frame.iloc[idx, 0])
        image = io.imread(img_name)
        #image = image.transpose((2, 1, 0))
        landmarks = self.landmarks_frame.iloc[idx, 1:].values
        landmarks = landmarks.astype('float')
        sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            sample = self.transform(sample)

        return sample

class ToTensor(object):
    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        #image = image.transpose((1,0,2))
        image = transforms.ToTensor()(image)
        landmarks = torch.from_numpy(landmarks)
        #image = image.transpose((2, 1, 0))
        return {'image': image,
                'landmarks': landmarks}

class Normalize():

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        for c in range(3):
            image[:, c] = (image[:, c] - self.mean[c]) / self.std[c]
        return {'image': image, 'landmarks': landmarks}

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(  # input shape (3, 128, 64)
            nn.Conv2d(
                in_channels=3,  # input height
                out_channels=32,  # n_filters
                kernel_size=3,  # filter size
                stride=1,  # filter movement/step
                padding=1,
            ),  # output shape (32, 128, 64)
            nn.ReLU(),  # activation
            nn.MaxPool2d(kernel_size=2),  # choose max value in 2x2 area, output shape (32, 64, 32)
        )
        self.conv2 = nn.Sequential(  # input shape (32, 64, 32)
            nn.Conv2d(32, 64, 3, 1, 1),  # output shape (64, 64, 32)
            nn.ReLU(),  # activation
            nn.MaxPool2d(2),  # output shape (64, 32, 16)
        )
        self.conv3 = nn.Sequential(  # input shape (64, 32, 16)
            nn.Conv2d(64, 128, 3, 1, 1),  # output shape (128, 32, 16)
            nn.ReLU(),  # activation
            nn.MaxPool2d(2),  # output shape (128, 16, 8)
        )
        self.conv4 = nn.Sequential(  # input shape (128, 16, 8)
            nn.Conv2d(128, 256, 3, 1, 1),  # output shape (256, 16, 8)
            nn.ReLU(),  # activation
            nn.MaxPool2d(2),  # output shape (256, 8, 4)
        )
        self.conv5 = nn.Sequential(  # input shape (256, 8, 4)
            nn.Conv2d(256, 512, 3, 1, 1),  # output shape (512, 8, 4)
            nn.ReLU(),  # activation
            nn.MaxPool2d(2),  # output shape (512, 4, 2)
        )
        self.conv6 = nn.Sequential(  # input shape (512, 4, 2)
            nn.Conv2d(512, 1024, 3, 1, 1),  # output shape (1024, 4, 2)
            nn.ReLU(),  # activation
            nn.MaxPool2d(2),  # output shape (1024, 2, 1)
        )
        self.fc = nn.Sequential(
            nn.Linear(1024 * 1 * 2, 1024),  # fully connected layer, output 1024
            nn.ReLU(),
            #nn.Dropout(0.5),
        )
        self.out = nn.Linear(1024, 2 * 4)  # fully connected layer, output 4 * 2 points

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = x.view(x.size(0), -1)  # flatten the output of convolution6 to (batch_size, 1024 * 2 * 1)
        fc = self.fc(x)
        output = self.out(fc)
        #tim.sleep(2000)
        return output

def tensor_to_PIL(tensor):
    unloader = transforms.ToPILImage()
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    #print(type(image))
    image = unloader(image)
    #print(type(image))
    return image


# def showpic(pic, landmarks)
#     i = landmarks.cpu().detach().numpy()
#     # print(i[0][0],i[0][1],i[0][2],i[0][3],i[0][4],i[0][5],i[0][6],i[0][7])
#     plt.imshow(tensor_to_PIL(pic))
#     plt.scatter(i[0][0], i[0][1])
#     plt.scatter(i[0][2], i[0][3])
#     plt.scatter(i[0][4], i[0][5])
#     plt.scatter(i[0][6], i[0][7])
#     plt.show()

# def studyRate(index):
#     if(index<)

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    train_loss = 0
    for batch_idx, batch in enumerate(train_loader):
        data, target = batch['image'], batch['landmarks']
        target = target.type(torch.FloatTensor)
        data = Variable(data).cuda()
        target = Variable(target).cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = F.mse_loss(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()  # sum up batch loss
    return train_loss / len(train_loader)
'''

def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for batch in test_loader:
            data, target = batch['image'].to(device).float(), batch['landmarks'].to(device).float()
            output = model(data)
            test_loss += F.l1_loss(output, target).item() # sum up batch loss

    print('test loss:', test_loss)
'''

def main():

    parser = argparse.ArgumentParser(description='Chepai')
    parser.add_argument('--batch-size', type=int, default=16 ,metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=65, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    device = torch.device("cuda" if use_cuda else "cpu")

    mean = torch.Tensor([0.485, 0.456, 0.406])
    std = torch.Tensor([0.229, 0.224, 0.225])


    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    transformed_dataset1 = LandmarksDataset(csv_file='chepai_landmarks.csv',
                                           root_dir='data/',
                                           transform=transforms.Compose([
                                               ToTensor()
                                           ]))
    transformed_dataset2 = LandmarksDataset(csv_file='chepai_landmarks3.csv',
                                           root_dir='data/',
                                           transform=transforms.Compose([
                                               ToTensor()
                                           ]))

    train_loader = DataLoader(transformed_dataset1, batch_size=args.batch_size,
                        shuffle=True, **kwargs)
    test_loader = DataLoader(transformed_dataset2, batch_size=args.test_batch_size,
                        shuffle=True, **kwargs)

    # model = torch.load('./cnn_2')#load the model
    # t_est1(args, model, device, test_loader)#do test work
    #
    # return 0

    torch.manual_seed(args.seed)


    model = resnet18().cuda()
    #model = CNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    loss_x = []
    for epoch in range(1, args.epochs + 1):
        begin = time()
        if epoch % 10 == 0:
            optimizer = optim.Adam(model.parameters(), lr=0.01 * pow(0.9, epoch))
            if epoch > 49:
                optimizer = optim.Adam(model.parameters(), lr=0.01 * pow(0.9, 40))
        loss = train(args, model, device, train_loader, optimizer, epoch)
        loss_x.append(loss)
        end = time()
        print('epoch {}/{} complete in {:02f}s, loss:{:03f}'.format(epoch , args.epochs, end - begin, loss))

    torch.save(model, './cnn_3')
    ax = plt.subplot(111)
    ax.plot(loss_x)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()
if __name__ == '__main__':
    main()

