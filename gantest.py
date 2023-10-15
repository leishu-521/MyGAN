import torch
from torch import nn, optim, autograd
import numpy as np
import visdom
from torch.nn import functional as F
from matplotlib import pyplot as plt
import random

h_dim = 400
batchsz = 512
viz = visdom.Visdom()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()

        self.net = nn.Sequential(
            #
            nn.Linear(2, h_dim),
            nn.ReLU(True),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(True),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(True),
            nn.Linear(h_dim, 2),
        )

    def forward(self, z):
        output = self.net(z)
        return output


class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(2, h_dim),
            nn.ReLU(True),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(True),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(True),
            nn.Linear(h_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        output = self.net(x)
        return output.view(-1)

def data_generator():

    scale = 2.
    centers = [
        (1, 0),
        (-1, 0),
        (0, 1),
        (0, -1),
        (1. / np.sqrt(2), 1. / np.sqrt(2)),
        (1. / np.sqrt(2), -1. / np.sqrt(2)),
        (-1. / np.sqrt(2), 1. / np.sqrt(2)),
        (-1. / np.sqrt(2), -1. / np.sqrt(2))
    ]
    centers = [(scale * x, scale * y) for x, y in centers]
    while True:
        dataset = []
        for i in range(batchsz):
            point = np.random.randn(2) * .02    #随机产生2个正态分布的数值，并乘以0.02
            center = random.choice(centers)
            # 对point加入高斯噪声
            point[0] += center[0]
            point[1] += center[1]
            dataset.append(point)
        dataset = np.array(dataset, dtype='float32') #把列表转换成array
        dataset /= 1.414  # stdev
        yield dataset   #类似于return的用法，只不过下次会继续从这块接着执行

def main():
    torch.manual_seed(23)
    np.random.seed(23)

    data_iter = data_generator()
    x = next(data_iter)
    # print(x.shape)

    G = Generator().cuda()
    D = Discriminator().cuda()
    # print(G)
    # print(D)

    for epoch in range(100):

        # 1.train Discriminator firstly
        for _ in range(5):
            # 1.1 train on real data
            xr = next(data_iter)
            xr = torch.from_numpy(xr).cuda()
            # [b, 2] => [b, 1]
            predr = D(xr)
            # max predr
            lossr = -predr.mean()


            # 1.2 train on fake data
            # [b,]
            z = torch.randn(batchsz, 2).cuda()
            xf = G(z)






        # 2. train Generator





if __name__ == "__main__":
    # a = torch.randn([32, 2])
    # G = Generator()
    # D = Discriminator()
    # b = G(a)
    # print(a)
    # d = D(b)
    # print("----------------------------")
    # print(d)
    main()
