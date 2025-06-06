import torch # 1.导入库
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# 2. 选择设备，设置设备 检查当前 PyTorch 是否能使用 GPU（NVIDIA CUDA）。如果可以用 GPU，就设置为 "cuda"，否则就用 CPU。
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#3.图像预处理
transform = transforms.Compose([ # -->  把图片转为 Tensor
    transforms.ToTensor(), #PyTorch 的张量（Tensor），缩放到 0 ~ 1（除以 255）
    transforms.Normalize((0.5,), (0.5,)) #进一步变为 -1 ~ 1（常见做法）
])

#4.加载 USPS 数据集
train_dataset = torchvision.datasets.USPS(
    root='./data', train=True, transform=transform, download=True) #train=true 是训练集，False是测试集
test_dataset = torchvision.datasets.USPS(
    root='./data', train=False, transform=transform, download=True)# transform 是刚才定义好的那三行

# 5. 打包数据成 DataLoader，batch_size=64 --> 每次从数据集中取出 64 张图片作为一个 mini-batch 训练模型
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True) #打乱顺序训练更鲁棒（建议训练时设为shuffle=True）
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# 6. 模型定义-->简单的全连接神经网络 256-->128-->10(0-9) [batch_size, 10]
# batch：批量大小，一次送进几张图 ，channel 通道数，一般是 1（灰度图）或 3（彩色图）
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__() # 一个子类，SimpleNet 继承自 nn.Module
        self.flatten = nn.Flatten() #输入从形状 [batch, 1, 16, 16] 变成 [batch, 256]，送入全连接层做分类
# 添加了一个 “展平层”，它的作用是把多维的张量“拉平”成一维向量，[64, 1, 16, 16]  ← 这代表 64 张图片，每张是 1×16×16
# Flatten 后，[64, 256]，每张图像的 1×16×16 = 256 个像素值被变成一个一维向量

        self.fc = nn.Sequential( #将多个神经网络层组合成一个整体结构
            nn.Linear(16*16, 128), #全连接层1（nn.Linear）的输入是一维向量，输出维度是 128，可以理解为有 128 个神经元
            nn.ReLU(), #激活函数 ReLU（Rectified Linear Unit）把“负数砍掉”的简单激活函数，帮神经网络保留有用信号、丢弃无用信息
            nn.Linear(128, 10) #全连接层2，输入维度是 128，输出维度是 10 → 表示 10 类数字（0~9）
        )

    def forward(self, x): # PyTorch 模型中定义前向传播（forward）
        x = self.flatten(x) #把图像“摊平”
        x = self.fc(x) #送进神经网络
        return x #输出每张图像属于每个数字的分数

#7. 模型实例化并迁移到设备
# model = SimpleNet() → 实例化模型，此时模型还只存在于 CPU 的内存中 & model.to(device)→ 把模型搬到 GPU 或 CPU 上
model = SimpleNet().to(device) # 如果你电脑有 GPU，就把模型搬到 GPU 上，否则就在 CPU 上运行

#8. 损失函数 + 优化器
criterion = nn.CrossEntropyLoss() # 损失函数（Loss Function）衡量“预测值”和“真实值”之间差距的函数，也就是分类错误的“惩罚”
# 对 模型输出 logits（原始分数） 进行 自动 softmax（概率） →  自动计算交叉熵 →  输出 loss（损失值）
# logits 是通过神经网络的前向传播一步步计算出来的，最终由最后一层 线性层（Linear） 输出
#logits 中正确类别的分数越高，softmax 概率就越大，loss 越小，所以预测越“自信正确”
# output = torch.tensor([[5, 0.2, 2]])  # 模型输出的 logits，label = torch.tensor([0])  # 正确类别是第0类
# softmax为类别 0 的概率 ≈ 94.5%， loss=−log(softmax[label])，−log(0.945)≈0.056
optimizer = optim.Adam(model.parameters(), lr=0.001)
#给你的模型绑定一个“智能调参器”，它会在训练过程中根据每次的 loss 自动更新模型参数，让模型变得越来越聪明。
#就像学生做错题，loss 是老师打的分，optimizer.step() 就是学生改错的动作。
#学习率（learning rate）lr=0.001，控制“每次更新的步子多大”，0.001 是 Adam 的常用默认值

# 9.训练函数
def train(model, loader, epochs):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        print(f"Epoch {epoch+1}: Loss = {running_loss/len(loader):.4f}")

# 10.测试函数
def test(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"Test Accuracy: {100 * correct / total:.2f}%")

# 跑通训练和测试
train(model, train_loader, epochs=5)
test(model, test_loader)
