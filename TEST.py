from torch import nn


def func(m):
    print(m.x)


class A(nn.Module):
    def __init__(self):
        super(A, self).__init__()
        self.x = 234


a = A()
a.apply(func)