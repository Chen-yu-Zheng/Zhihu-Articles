#! https://zhuanlan.zhihu.com/p/358371413
# AlexNet 论文阅读

## Abstract

网络特征: Large, deep, convolutional

网络结果：ImageNet数据集上的测试结果远超前人网络，大比分领先获得2012年冠军

网络结构：

* 非常大，6000w参数，650000神经元
* 采用了5个 conv-maxpooling层 + 全连接的分类网络

训练创新：采用了非饱和神经元，更有效的GPU

优化策略：dropout算法

## Introduction

* 现有的机器学习算法在几万张图片量级的数据集上表现非常出色，甚至在一些方面超越了人类。但是在最近出现了千万量级的大数据集让传统算法望而却步
* 要想完成真正意义上的目标检测任务，就连ImageNet的数据量也是不够的。要想进一步发展，我们需要先在ImageNet这种量级的数据集上取得突破
* 卷积神经网络的学习能力可以依靠调整其广度和深度来实现，在自然图像的提取与处理方面有极强的能力，而且相较于传统的全连接网络，它的参数量更少，更容易进行训练
* 尽管如此，普通的卷积神经网络来训练ImageNet仍然是有恐怖的消耗。幸运的是，在硬件上采用GPU，在算法方面高度优化卷积层的实现，使得现有资源能够支撑此次训练
* 第一次提出如此复杂的深度网络，使得过拟合现象较为严重，本次实验采用了多种防止过拟合的方法
* 深度似乎非常重要，去除5个卷积层中的一个(参数占比<1%)，都会导致表现结果大跌眼镜（如何确定网络结构的深度是否科学）
* 如果有更好的硬件，和更长的训练时间，此方法可以推广于更大的数据集

## 实现网络的一些新方法（按照重要性排序）

* ReLu饱和神经元训练往往伴随着梯度消失，使得训练过程缓慢，在大数据集上更会导致训练过久。而Relu非饱和，大大加快了训练进程；(ReLu有无局限性？之后我们在激活函数上又取得了什么样的进展？)
* GPU硬件加速，现在已经是无GPU不深度学习了
* Local Response Normalization，在一些层的ReLu之后使用（4个超参数是怎么选的？为什么能起作用？为什么要放到ReLu层之后？还有更好的中间过程处理方法吗？）
* Overlapping Pooling，大量实验表明，此方法能够略微提高准确率，并且防止过拟合（为什么？如何对于一个任务，一个网络，每个卷积层找到最好的stride和kernel size？或者说，这些超参数如何科学地确定？）

## 网络整体结构

* 输入图像大小(3,227,227)
* layer1：
  * 卷积层，96个卷积核，大小为(11,11,3)，步长为4，无填充 --> (96,55,55)
  * ReLu
  * Local Response Normalization
  * 池化层，大小为3，步长为2 --> (96,27,27)
* layer2：
  * 卷积层，256个卷积层，大小为(5,5,96)，步长为1，填充为2 --> (256,27,27)
  * ReLu
  * Local Response Normalization
  * 池化层，大小为3，步长为2--> (256,13,13)
* layer3：
  * 卷积层，384个卷积核，大小为(3,3,256)，步长为1，填充为1 --> (384,13,13)
  * ReLu
* layer4：
  * 卷积层，384个卷积核，大小为(3,3,384)，步长为1，填充为1 --> (384,13,13)
  * ReLu
* layer5:
  * 卷积层，256个卷积核，大小为(3,3,384)，步长为1，填充为1 --> (256,13,13)
  * ReLu
  * 池化层，大小为3，步长为2 --> (256,6,6) --> (256 * 6 * 6,1)
* Dense:
  * 256 * 6  * 6 --> 4096, ReLu, Dropout
  * 4096 --> 4096, ReLu, Dropout
  * 4096 --> 1000

## 具体的训练过程

* 优化器为带动量的随机梯度下降，momentum = 0.9，权重衰减指数为0.0005，实验表明，此处的权重衰减不仅起到了正则化的作用，而且能够提高在训练集上的得分
* 参数初始化
  * 每一层的权重 $w \sim N(0,0.01)$ 
  * 第2，4，5卷积层和所有连接层的偏置设为1
  * 其余层的偏置设为0
* 学习率初始化为0.01，在整个训练过程中手动下调，且每层的学习率保持一致。在现有的学习率条件下，每当练效果没有明显提升，那么将学习率除以10
* 大概在训练集上训练了90轮，用时5天多

## 防止过拟合的方法

* 在训练集上进行数据增强，对图像进行变换，包括变色，平移，翻转，切片，放缩等，得到更大的数据集
* 在训练集上利用PCA进行转化
* Dropout，在全连接层之间放置Dropout进行随机丢弃。从集成学习的角度来看，可以视作多个小模型的融合。从传递的角度来看，每个神经元不能依赖特定神经元的输出，从而强迫其学习到更为鲁棒性的特征

## 总结讨论

* 再一次声明，深度对于本模型至关重要，未来的趋势可能就是要考虑大而深的卷积神经网络
* 进一步的验证表明，更大更深的模型取得了更好的效果。但即使是这种大模型，相对于人类的神经网络系统，还是望尘莫及的
* 将使用更大的模型在video领域

## Python复现

```python
import torch
from torch import nn
from torch.nn import init

class AlexNet(nn.Module):
    def __init__(self):
        #img_size:(3,227,227)
        super(AlexNet, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels= 3, out_channels= 96, kernel_size= 11, stride= 4, padding= 0),
            nn.ReLU(inplace= True),
            nn.LocalResponseNorm(size= 5, alpha= 0.0001, beta= 0.75, k= 2),
            nn.MaxPool2d(kernel_size= 3, stride= 2)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels= 96, out_channels= 256, kernel_size= 5, stride= 1, padding= 2),
            nn.ReLU(inplace= True),
            nn.LocalResponseNorm(size= 5, alpha= 0.0001, beta= 0.75, k= 2),
            nn.MaxPool2d(kernel_size= 3, stride= 2)
        )
        
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels= 256, out_channels= 384, kernel_size= 3, stride= 1, padding= 1),
            nn.ReLU(inplace= True),
            nn.Conv2d(in_channels= 384, out_channels= 384, kernel_size= 3, stride= 1, padding= 1),
            nn.ReLU(inplace= True),
            nn.Conv2d(in_channels= 384, out_channels= 256, kernel_size= 3, stride= 1, padding= 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size= 3, stride= 2)
        )

        self.Dense = nn.Sequential(
            nn.Linear(in_features= 256*6*6, out_features= 4096),
            nn.ReLU(inplace= True),
            nn.Dropout(p= 0.5),
            nn.Linear(in_features= 4096, out_features= 4096),
            nn.ReLU(inplace= True),
            nn.Dropout(p= 0.5),
            nn.Linear(4096,21)
        )

    def init_params(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                init.normal_(param, mean= 0.0, std = 0.01)
            else:
                if ('layer2' in name) or ('layer3.2' in name) or ('layer3.4' in name) or ('Dense' in name):
                    init.constant_(param, 1)
                else:
                    init.constant_(param, 0)
                

    def forward(self, img):
        out = self.layer1(img)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.shape[0], -1)
        out = self.Dense(out)
        return out

if __name__ == '__main__':
    net = AlexNet()
    net.init_params()

    img = torch.rand((1,3,227,227))
    out = net(img)
    print(out)

    for name, param in net.named_parameters():
        print(name, param.data)
```





