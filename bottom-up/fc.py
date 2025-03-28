
import torch.nn as nn
# from torch.nn.utils.weight_norm import weight_norm  # 替换过时模块
from torch.nn.utils.parametrizations import weight_norm

class FCNet(nn.Module):
    """Simple class for non-linear fully connect network
    """
    def __init__(self, dims):   # default: ReLU
        super(FCNet, self).__init__()

        layers = []
        for i in range(len(dims)-2):
            in_dim = dims[i]
            out_dim = dims[i+1]
            layers.append(weight_norm(nn.Linear(in_dim, out_dim), dim=None))
            layers.append(nn.ReLU())
        layers.append(weight_norm(nn.Linear(dims[-2], dims[-1]), dim=None))
        layers.append(nn.ReLU())

        self.main = nn.Sequential(*layers)

    # def __init__(self, dims):   # tanh
    #     super(FCNet, self).__init__()

    #     layers = []
    #     for i in range(len(dims)-2):
    #         in_dim = dims[i]
    #         out_dim = dims[i+1]
    #         layers.append(weight_norm(nn.Linear(in_dim, out_dim), dim=None))
    #         layers.append(nn.Tanh())  # 替换为 Tanh 激活函数
    #     layers.append(weight_norm(nn.Linear(dims[-2], dims[-1]), dim=None))
    #     layers.append(nn.Tanh())  # 替换为 Tanh 激活函数

    #     self.main = nn.Sequential(*layers)

    # def __init__(self, dims):   # GLU
    #     super(FCNet, self).__init__()

    #     layers = []
    #     for i in range(len(dims)-2):
    #         in_dim = dims[i]
    #         out_dim = dims[i+1] * 2  # 输出维度加倍，以适应 GLU
    #         layers.append(weight_norm(nn.Linear(in_dim, out_dim), dim=None))
    #         layers.append(nn.GLU())  # 使用 GLU 激活
    #     layers.append(weight_norm(nn.Linear(dims[-2], dims[-1] * 2), dim=None))  # 输出维度加倍
    #     layers.append(nn.GLU())  # 使用 GLU 激活

    #     self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


if __name__ == '__main__':
    fc1 = FCNet([10, 20, 10])
    print(fc1)

    print('============')
    fc2 = FCNet([10, 20])
    print(fc2)
