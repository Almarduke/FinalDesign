import torch


a = torch.FloatTensor([[1,2,3],[4,5,6],[7,8,9]])
b = torch.FloatTensor([[1,2,2],[4,2,6],[0,1,9]])
print(a * b)
print(torch.mul(a, b))
print(a.exp())