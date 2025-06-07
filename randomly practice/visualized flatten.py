import torch
import torch.nn as nn
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # macOS 中文支持
plt.rcParams['axes.unicode_minus'] = False


# 随机生成一张图像，模拟 USPS 数据（1通道，16x16）
image = torch.rand(1, 16, 16)

# 展示原图
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.imshow(image.squeeze(), cmap='gray')
plt.title("原图像：16 x 16")
plt.axis("off")

# 展平：用 nn.Flatten()
flatten = nn.Flatten()
flat_image = flatten(image)  # 展平成 [1, 256]

# 展平后可视化为条形图
plt.subplot(1, 2, 2)
plt.plot(flat_image.squeeze().numpy())
plt.title("展平后：1 x 256 向量")
plt.xlabel("像素位置")
plt.ylabel("像素值")

plt.tight_layout()
plt.show()

# 打印形状对比
print("原始图像形状：", image.shape)       # torch.Size([1, 16, 16])
print("展平后的形状：", flat_image.shape)  # torch.Size([1, 256])
