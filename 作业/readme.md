# 数据增广

与物品识别不同，对人像处理时，尽量不要进行扭曲变形。

可以通过：

* 裁剪
* 旋转
* 调节灰度

来进行数据增强。

将增强的数据存入augment文件夹，完成数据增广

```python
import os, Augmentor
import shutil, glob

# 设置输出目录，控制不重复增强数据
if not os.path.exists(augment_path): 
    
    #os.walk()遍历指定目录下所有的文件夹
    for root, dirs, files in os.walk("/home/aistudio/data_new", topdown=False):
        print(root,dirs,files,'1')
        for name in dirs:
            path_ = os.path.join(root, name)

            # 过滤掉系统隐藏文件夹
            if '__MACOSX' in path_:continue
            print('数据增强：',os.path.join(root, name))
            print('image：',os.path.join(root, name))
            
            #创建 Augmentor.Pipeline 实例，并指定需要增强的图片所在目录，设置增强操作的参数，如旋转、缩放、扭曲等
            p = Augmentor.Pipeline(os.path.join(root, name),output_directory='output')
           
           #p.rotate()方法来设置旋转增强的概率（probability）
           #最大左旋角度（max_left_rotation）
           #最大右旋角度（max_right_rotation）
            p.rotate(probability=0.6, max_left_rotation=2, max_right_rotation=2)

            #p.zoom()方法用于设置缩放增强的概率（probability）
            #最小缩放因子（min_factor）
            #最大缩放因子（max_factor）
            p.zoom(probability=0.6, min_factor=0.9, max_factor=1.1)

            #p.random_distortion()方法用于设置扭曲增强的概率（probability）
            #网格高度（grid_height）、网格宽度（grid_width）和扭曲强度（magnitude）
            p.random_distortion(probability=0.4, grid_height=2, grid_width=2, magnitude=1)

            # 根据已有图片数量计算需要增强的数量
            count = 600 - len(glob.glob(pathname=path_+'/*.jpg'))

            #调用 sample() 方法进行样本扩增。
            p.sample(count, multi_threaded=False)
            p.process()

    print('将生成的图片拷贝到正确的目录')
    for root, dirs, files in os.walk("/home/aistudio/data_new", topdown=False):
        for name in files:
            path_ = os.path.join(root, name)
            if path_.rsplit('/',3)[2] == 'output':
                type_ = path_.rsplit('/',3)[1]
                dest_dir = os.path.join(augment_path ,type_) 
                if not os.path.exists(dest_dir):os.makedirs(dest_dir) 
                dest_path_ = os.path.join(augment_path ,type_, name) 
                shutil.move(path_, dest_path_)
    print('删除所有output目录')
    for root, dirs, files in os.walk("/home/aistudio/data_new", topdown=False):
        for name in dirs:
            if name == 'output':
                path_ = os.path.join(root, name)
                shutil.rmtree(path_)
    print('完成数据增强')
```



# 基于残差连接的CNN网络（用dropout防止过拟合)

一共分为5部分：

* 特征提取层

  * Conv2D
  * MaxPool2D

* **dropout层**

  是一种正则化技术，在神经网络中减少过拟合

  具体来说，`Dropout`层会随机选择一些神经元的输出，将其置为零。这些神经元在当前的前向传递中被“丢弃”，它们的权重不会更新。这样，每个神经元都有一定的概率被丢弃，从而减少神经元之间的依赖关系，使得模型更加鲁棒，并且更能泛化到新的数据上。

* **BatchNorm2D**

  是一种批量归一化技术，用于在神经网络中进行正则化，提高模型的训练速度和准确性。

  通过将数据进行归一化，可以确保激活函数的输入具有零均值和单位方差。这有助于减少内部协方差偏移问题，使得模型更容易收敛。此外，批量归一化还可以降低模型对初始化的敏感性，并且具有一定的正则化效果，有助于防止过拟合。

* **残差连接(Residual Connection)**

  残差连接通常用于深度卷积神经网络（CNN）和残差网络（ResNet）中，用于减轻梯度消失问题并加速模型的收敛速度。

* 全连接层

  进行拟合

```python
import paddle
from paddle.nn import Conv2D, MaxPool2D, Linear,BatchNorm2D
import paddle.nn.functional as F

class MyCNN(paddle.nn.Layer):  #@save
    def __init__(self):
        super(MyCNN, self).__init__()
        
        #特征提取层
        self.conv1=Conv2D(in_channels=3, out_channels=32, kernel_size=3,stride=1,padding=2)
        self.pool1=MaxPool2D(kernel_size=2,stride=2)
        self.conv2=Conv2D(in_channels=32,out_channels=32,kernel_size=3,stride=1,padding=2)
        self.pool2=MaxPool2D(kernel_size=2,stride=2)

        #dropout层，正则化技术，防止过拟合
        self.dropout = paddle.nn.Dropout(0.5)
        
        #BatchNorm2D--批量归一化技术
        self.bn1 = BatchNorm2D(32)#out_channels
        self.bn2 = BatchNorm2D(32)
        
        #全连接层
        self.fc1 = paddle.fluid.dygraph.Linear(input_dim=32*57*57, output_dim=1024, act='relu')
        self.fc2 = paddle.fluid.dygraph.Linear(input_dim=1024, output_dim=512, act='relu')
        self.fc3 = paddle.fluid.dygraph.Linear(input_dim=512, output_dim=128, act='relu')
        self.fc4 = paddle.fluid.dygraph.Linear(input_dim=128, output_dim=64, act='relu')
    def forward(self, input):

        Y = F.relu(self.bn1(self.conv1(input)))
        Y=self.pool1(Y)
        Y = self.bn2(self.conv2(Y))
        Y=self.pool2(Y)
 

        X=F.relu(self.bn1(self.conv1(input)))
        X=self.pool1(X)
        X = self.bn2(self.conv2(X))
        X=self.pool2(X)

		#残差连接
        Y+=X
        Y=paddle.reshape(Y,shape=[-1,32*57*57])
        Y=self.dropout(Y)
        Y=self.fc1(Y)
        Y=self.dropout(Y)
        Y=self.fc2(Y)
        Y=self.dropout(Y)
        Y=self.fc3(Y)
        Y=self.dropout(Y)
        Y=self.fc4(Y)
        return Y
```

