import torch # 1.导入库
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt #python自带画图软件
import random
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # macOS 中文支持
plt.rcParams['axes.unicode_minus'] = False

# 2. 选择设备，设置设备 检查当前 PyTorch 是否能使用 GPU（NVIDIA CUDA）。如果可以用 GPU，就设置为 "cuda"，否则就用 CPU。
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#3.图像预处理 数据！！！
transform = transforms.Compose([ # -->  把图片转为 Tensor  矢量/向量vector（二维) 标量scale(一维)
    transforms.ToTensor(), #PyTorch 的张量（Tensor），缩放到 0 ~ 1（除以 255） Tensor #属性shape dtype strides
    transforms.Normalize((0.5,), (0.5,)) #进一步变为 -1 ~ 1（常见做法）
])  #例如 百分制和五分制的衡量，拉到一个尺度去衡量 OR 80到100分，40-60分，由于卷子比较难，去比较也需要normalize

#4.加载 USPS 数据集
train_dataset = torchvision.datasets.USPS(
    root='./data', train=True, transform=transform, download=True) #train=true 是训练集，False是测试集
test_dataset = torchvision.datasets.USPS(
    root='./data', train=False, transform=transform, download=True)# transform 是刚才定义好的那三行

# 5. 打包数据成 DataLoader，batch_size=64 --> 每次从数据集中取出 64 张图片作为一个 mini-batch 训练模型,越小bitch size，迭代次数越多
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
        '''
        fc: fully connected ,nn --> neural network 的缩写，一个神经网络有一个layer fully connected
        神经网络-->模拟人的神经的机制，激活的机制，截取一段比较敏感的信号，然后将其放大激活.ReLU() 
        '''
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

#8. 损失函数 + 优化器. !.CrossEntropyLoss() !最常见
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

# 9.训练函数 model 是你要训练的神经网络/loader 是训练数据加载器（DataLoader），每次提供一批图像和标签
def train(model, loader, epochs): #epochs 是训练多少轮（遍历几次训练集
    model.train() #把模型设置为“训练模式”，这句告诉 PyTorch：“我要开始训练了”
    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in loader: # 从 DataLoader 中一批一批地取数据。
            #images：一个形状为 [batch_size, 1, 16, 16] 的张量，labels：对应的真实数字标签（如 [0, 2, 5, 7, ...]）
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad() #清空模型上次计算的梯度，如果不清除，上次的梯度会累加进来，导致模型参数更新错乱。
            outputs = model(images) #前向传播：把输入图像送进模型，得到预测输出（logits）
            # 比如返回的形状是 [64, 10]，表示每张图对 10 个类别的打分。例子中64代表64张图
            loss = criterion(outputs, labels) #损失函数:模型“预测结果”与“真实标签”之间的差距 一切我想优化的东西都写在loss，minimize这个loss
            loss.backward() #这些梯度告诉优化器该如何更新模型，使 loss 更小。
            optimizer.step() #优化器根据刚才的梯度，更新模型参数。

            running_loss += loss.item() #把这一批的 loss 累加起来，准备计算整轮的平均损失。.item() 是把张量变成普通数值。
        print(f"Epoch {epoch+1}: Loss = {running_loss/len(loader):.4f}")
# 打印这一轮训练的平均 loss，保留 4 位小数。len(loader) 是有多少批次，running_loss / len(loader) 就是每轮平均损失。
'''
def train(model, loader, epochs):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for i, (images, labels) in enumerate(loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # 准确率统计
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            # ✅ 每个 epoch 的第一个 batch 可视化一张图像 + 模型预测
            if i == 0:
                idx = random.randint(0, images.size(0) - 1)  # 随机选一张图
                img = images[idx].cpu().squeeze().numpy()
                true_label = labels[idx].item()
                pred_label = predicted[idx].item()

                plt.imshow(img, cmap='gray')
                plt.axis('off')
                plt.show()
                print(f"✅ 可视化样本：预测 = {pred_label}，真实 = {true_label}")

        avg_loss = running_loss / len(loader)
        acc = 100 * correct / total
        print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}, Accuracy = {acc:.2f}%")
'''
# 10.测试函数
def test(model, loader): #model 是你训练好的神经网络，loader 是测试数据集的 DataLoader（一次送入很多张图像）
    model.eval() #把模型设置为“评估模式”，和训练时的 .train() 相对
    correct = 0
    total = 0
    with torch.no_grad(): #在这个区域里不要计算梯度。
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images) #前向传播，把图片喂进神经网络，每张图片属于 10 个数字的“打分”（logits）
            #形状是 [batch_size, 10]，每一行表示一张图对每个数字的信心。
            _, predicted = torch.max(outputs, 1) #只关心第二个结果（预测类别），不需要最大值本身，就用 _ 表示“忽略这个值”。
            print(predicted)
            total += labels.size(0)
            correct += (predicted == labels).sum().item() # binary的TF变成0或者1 有一个T计一个1
    print(f"Test Accuracy: {100 * correct / total:.2f}%")

# 跑通训练和测试
train(model, train_loader, epochs= 5)
test(model, test_loader)

#demension = 0 的时候
# .to(device) 特殊用法
#建议打印139行看一看
#140行就是求概率最大的 最有信心的值是什么
#_, predicted == 前面那个是最大的值是多少，但是我不需要