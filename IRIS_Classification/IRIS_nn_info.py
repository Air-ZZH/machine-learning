import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.autograd import Variable
import tqdm # 下面的进度条

iris = load_iris()
X = iris['data'] # 150个样本每个 #150个 [[5.1 3.5 1.4 0.2]
y = iris['target'] # 0,1,2代表的labels

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X) #使用 StandardScaler 将特征标准化（均值为 0，方差为 1）

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=2)
'''
X_train 如下一共120行  150*0.8 = 120
[ 4.32165405e-01 -5.92373012e-01  5.92245988e-01  7.90670654e-01]
 [-9.00681170e-01  5.58610819e-01 -1.16971425e+00 -9.20547742e-01]
 [-2.94841818e-01 -3.62176246e-01 -8.98031345e-02  1.32509732e-01]
'''

# 转换为 PyTorch 张量
X_train = torch.from_numpy(X_train).float()
y_train = torch.from_numpy(y_train).long()
X_test = torch.from_numpy(X_test).float()
y_test = torch.from_numpy(y_test).long()

'''
print(X_train.shape) --> torch.Size([120, 4])
print(y_train.shape) --> torch.Size([120])
'''

class Model(nn.Module):
    def __init__(self, input_dim):  # dim 就是 dimension（维度）的缩写，input_dim 是一种约定俗成的变量命名方式，用来表示输入特征的维度。

        super(Model, self).__init__()
        self.layer1 = nn.Linear(input_dim, 64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, 3)

    def forward(self, x):
        # Input → FC（隐藏层1）→ ReLU → FC（隐藏层2）→ ReLU → FC（输出层）→ Softmax
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)  # 注意：不加 softmax
        return x

# 实例化模型
model = Model(X_train.shape[1]) #.shape[0] 是样本数（比如 120）// .shape[1] 是每个样本的特征数（比如 IRIS 是 4 个特征）
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()
#这就是 PyTorch 框架中 nn.Module 的一个 魔法功能 —— 它在内部重写了 __call__() 方法，自动帮你调用 forward()。
'''
print(model)
Model(
  (layer1): Linear(in_features=4, out_features=64, bias=True)
  (layer2): Linear(in_features=64, out_features=64, bias=True)
  (layer3): Linear(in_features=64, out_features=3, bias=True)
)'''

EPOCHS = 100
loss_list = np.zeros(EPOCHS)
''' length为100
[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
 0. 0. 0. 0.]
'''
accuracy_list = np.zeros(EPOCHS)

for epoch in tqdm.trange(EPOCHS): #EPOCHS，EPOCHS = 50 就训练 50 轮
    #上面一行等效于 for epoch in range(EPOCHS):
    model.train()
    y_pred = model(X_train)
    #y_pred = model.forward(X_train) 等同

    loss = loss_fn(y_pred, y_train)

    '''
    tensor(0.1540, grad_fn=<NllLossBackward0>) grad_fn=<NllLossBackward0> 表示这个张量（loss）是通过某个计算图生成的结果
    tensor(0.1506, grad_fn=<NllLossBackward0>) 而 NllLossBackward0 是 负责“反向传播”这一步骤的函数名。
    tensor(0.1474, grad_fn=<NllLossBackward0>) grad_fn 就是这个“操作”的反向传播函数（backward function）
    '''
    loss_list[epoch] = loss.item() #得到loss的值了 如0.1339760571718216

    optimizer.zero_grad() #清空模型中所有参数的梯度信息，防止梯度累加。
    loss.backward() #反向传播
    optimizer.step() #这一步是“真正学习”的那一刻。PyTorch 就根据你选择的优化算法Adam 用梯度调整参数，使 loss 下降一点点。

    model.eval()
    with torch.no_grad(): #禁用梯度计算 测试阶段不需要反向传播，所以不需要计算梯度
        y_pred_test = model(X_test)

        predictions = torch.argmax(y_pred_test, dim=1)
        '''
        tensor([2, 2, 1, 2, 2, 2, 2, 1, 1, 2, 2, 2, 2, 2, 2, 1, 1, 2, 2, 1, 2, 2, 2, 1,
        1, 2, 2, 2, 2, 2])
        tensor([2, 2, 1, 2, 2, 2, 2, 1, 1, 2, 2, 2, 2, 2, 2, 1, 1, 1, 2, 1, 2, 2, 2, 1,
        1, 2, 2, 2, 2, 2])
        tensor([2, 2, 1, 1, 2, 2, 1, 1, 1, 1, 2, 2, 2, 2, 2, 1, 1, 1, 2, 1, 2, 2, 2, 1,
        1, 2, 2, 2, 2, 2])
        '''

        correct = (predictions == y_test).float().mean().item()
        accuracy_list[epoch] = correct

    # 打印每轮结果（百分比格式）
    print(f"Epoch {epoch+1:3d} | Loss: {loss.item():.4f} | Test Accuracy: {correct * 100:.2f}%")