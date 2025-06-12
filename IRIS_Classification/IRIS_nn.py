
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

# 设置随机种子保证结果可复现
# np.random.seed(0)
# torch.manual_seed(0)

# 加载数据集并标准化
iris = load_iris()
X = iris['data']
y = iris['target']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 划分训练和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=2)

# 转换为 PyTorch 张量
X_train = torch.from_numpy(X_train).float()
y_train = torch.from_numpy(y_train).long()
X_test = torch.from_numpy(X_test).float()
y_test = torch.from_numpy(y_test).long()

# 定义模型
class Model(nn.Module):
    def __init__(self, input_dim):
        super(Model, self).__init__()
        self.layer1 = nn.Linear(input_dim, 64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, 3)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)  # 注意：不加 softmax
        return x

# 实例化模型
model = Model(X_train.shape[1])
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

# 开始训练
EPOCHS = 100
loss_list = np.zeros(EPOCHS)
accuracy_list = np.zeros(EPOCHS)

for epoch in tqdm.trange(EPOCHS):
    model.train()
    y_pred = model(X_train)
    loss = loss_fn(y_pred, y_train)
    loss_list[epoch] = loss.item()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        y_pred_test = model(X_test)
        predictions = torch.argmax(y_pred_test, dim=1)
        correct = (predictions == y_test).float().mean().item()
        accuracy_list[epoch] = correct

    # 打印每轮结果（百分比格式）
    print(f"Epoch {epoch+1:3d} | Loss: {loss.item():.4f} | Test Accuracy: {correct * 100:.2f}%")

# 画出 Loss 和 Accuracy 曲线并保存
fig, (ax1, ax2) = plt.subplots(2, figsize=(12, 6), sharex=True)
ax1.plot(accuracy_list)
ax1.set_ylabel("Test Accuracy")
ax2.plot(loss_list)
ax2.set_ylabel("Loss")
ax2.set_xlabel("Epochs")
plt.savefig("iris_training_results.png")
plt.show()

