# 机器学习和深度学习的差别

## 机器学习

- **学习任务：**回归、分类、聚类
- **学习方式：**有监督学习、无监督学习（训练数据是否有标签）、强化学习（通过模拟大脑神经细胞中的奖励信号来改善行为的机器学习方法）

对比两者：

机器学习往往预先定义着可解释的特征，在训练之后，可以直观的分析数据的一些特征。

深度学习则通过大量的网络层自动提取输入的特征信息，且特征信息很难解释。

| 特性              | 机器学习                           | 深度学习                                 |
| ----------------- | ---------------------------------- | ---------------------------------------- |
| **特征提取**      | 人工提取，特征可解释               | 自动提取，特征难解释                     |
| **模型复杂性**    | 模型结构简单，可解释性强           | 模型复杂，非线性，难以解释               |
| **数据依赖性**    | 对中小规模数据表现好，依赖特征工程 | 需要大量数据，数据不足时表现不佳         |
| **计算资源需求**  | 资源需求较低，运行时间短           | 资源需求高，依赖GPU，训练时间长          |
| **可解释性**      | 高，可解释性好                     | 低，通常是“黑箱”                         |
| **适用场景**      | 适合结构化数据（如表格数据）       | 适合非结构化数据（如图像、语音、文本等） |
| **训练/推理时间** | 训练和推理时间较短                 | 训练时间长，推理时间较长                 |

## 深度学习

深度学习的核心在于深度神经网络，而神经网络其实是机器学习的一种模型，深度学习主要是采用了**多层的神经网络**

<img src="C:\Users\84301\AppData\Roaming\Typora\typora-user-images\image-20241018112234481.png" alt="image-20241018112234481" style="zoom: 50%;" />

损失函数：```L```，误差和损失函数是相关的关系而不对等

激活函数：```f```

**前向传播：**
$$
layer_1:\\
\text{线性输出：}z_1=W_1a_0+b_1\\
\text{激活函数：}a_1=f(z_1)\\
layer_2:\\
\text{线性输出：}z_2=W_1a_1+b_2\\
\text{激活函数：}a_2=f(z_2)\\
$$
**反向传播：**逐层计算损失函数关于模型参数的梯度$\delta(l)$
$$
\delta(2)=\frac{\partial L}{\partial z_2}=
\frac{\partial L}{\partial a_2}\frac{\partial a_2}{\partial z_2}\\
\delta(1)=\frac{\partial L}{\partial z_1}=
\frac{\partial L}{\partial a_2}\frac{\partial a_2}{\partial z_2}\frac{\partial z_2}{\partial a_1}\frac{\partial a_1}{\partial z_1}=
\delta(2)W_1f'(z_1)
$$
由此可见误差梯度可由最后一层往前传播，结合所有层的误差梯度即可更新各层参数

![image-20241018145711139](C:\Users\84301\AppData\Roaming\Typora\typora-user-images\image-20241018145711139.png)

## 线性到非线性神经网络

线性回归->softmax回归->多层感知机

预测->分类

<img src="C:\Users\84301\AppData\Roaming\Typora\typora-user-images\image-20241018154112529.png" alt="image-20241018154112529" style="zoom: 40%;" /><img src="C:\Users\84301\AppData\Roaming\Typora\typora-user-images\image-20241018154152139.png" alt="image-20241018154152139" style="zoom: 40%;" />

<img src="C:\Users\84301\AppData\Roaming\Typora\typora-user-images\image-20241022110953160.png" alt="image-20241022110953160" style="zoom: 40%;" /><img src="C:\Users\84301\AppData\Roaming\Typora\typora-user-images\image-20241022112100486.png" alt="image-20241022112100486" style="zoom:50%;" />

**线性回归到`softmax`回归主要是输出变多，多层感知机是在中间加入含有激活函数（非线性函数）的隐藏层，让特征提取更加泛化**

**为了防止模型的过拟合问题，又引入了`dropout`(暂退法)，通过一定概率，在训练的时候忽略掉一些神经元**

**判断网络是否是线性**的主要看是否引入非线性激活函数（ReLu，sigmoid，Tanh），除此之外，像dropout和maxpooling也会带来非线性操作。

### softmax+cross entropy

`softmax`是为了将输出统一到和为1，是一种规范化<img src="C:\Users\84301\AppData\Roaming\Typora\typora-user-images\image-20241018154536230.png" alt="image-20241018154536230" style="zoom:50%;" />

`cross entropy`是交叉熵损失<img src="C:\Users\84301\AppData\Roaming\Typora\typora-user-images\image-20241018155355444.png" alt="image-20241018155355444" style="zoom:50%;" />多用于分类问题的损失函数

`softmax+cross entrop`：

<img src="C:\Users\84301\AppData\Roaming\Typora\typora-user-images\image-20241018155157837.png" alt="image-20241018155157837" style="zoom:50%;" />

基于损失函数为交叉熵的`softmax`的梯度为观测值y和估计值y^之间的差异

### 非线性激活函数

`ReLu:`<img src="C:\Users\84301\AppData\Roaming\Typora\typora-user-images\image-20241022170159865.png" alt="image-20241022170159865" style="zoom:50%;" />

**优点：**解决了 Sigmoid 和 Tanh 函数的梯度消失问题，当 x>0x > 0x>0 时，梯度为常数，不会缩小。

**缺点：** **Dying ReLU** 问题：当输入小于等于 0 时，梯度为 0，可能导致神经元永远不更新，从而"死亡"。

![../_images/output_mlp_76f463_18_1.svg](https://zh.d2l.ai/_images/output_mlp_76f463_18_1.svg)

`sigmoid:`<img src="C:\Users\84301\AppData\Roaming\Typora\typora-user-images\image-20241022170345224.png" alt="image-20241022170345224" style="zoom:50%;" />

**优点：**常用于二分类问题，可以输出概率值。

![../_images/output_mlp_76f463_48_0.svg](https://zh.d2l.ai/_images/output_mlp_76f463_48_0.svg)

`Tanh:`<img src="C:\Users\84301\AppData\Roaming\Typora\typora-user-images\image-20241022170432862.png" alt="image-20241022170432862" style="zoom:50%;" />

**优点： ** **零中心化**：输出在 −1 到 1 之间，且零点对称。

![../_images/output_mlp_76f463_78_0.svg](https://zh.d2l.ai/_images/output_mlp_76f463_78_0.svg)

## 卷积神经网络

### 全连接层到卷积神经网络的过渡

全连接层多考虑一维输入输出，涉及到图像这种二维甚至三维（RGB通道）的输入时，全连接层会出现无法全面提取特征的弊端。

卷积神经网络里提到的卷积其实是互相关运算，数学意义上的卷积还需要将卷积核先进行上下左右反转之后运算。

<img src="C:\Users\84301\AppData\Roaming\Typora\typora-user-images\image-20241023093744055.png" alt="image-20241023093744055" style="zoom:50%;" />**卷积**

<img src="C:\Users\84301\AppData\Roaming\Typora\typora-user-images\image-20241023093753016.png" alt="image-20241023093753016" style="zoom:50%;" />**互相关**



- **平移不变性：**不管检测对象出现在图像中的哪个位置，神经网络的前面几层应该对相同的图像区域具有相似的反应，即为“平移不变性”。

  **在卷积神经网络中，对一个二维输入一个输出时，只采取一个卷积核对整幅图像进行操作，而不是在图像的不同区域有不同的卷积核。**

- **局部性：**神经网络的前面几层应该只探索输入图像中的局部区域，而不过度在意图像中相隔较远区域的关系，这就是“局部性”原则。最终，可以聚合这些局部特征，以在整个图像级别进行预测。

  **卷积就是局部提取图像的特征，全连接层相当于一个w\*h的卷积核，一次考虑了整幅图像的信息，而卷积核则是通过卷积运算，比如3\*3的卷积核就是考虑了目标像素点附近八邻近点的信息，因此比起一个w\*h的权重矩阵，3\*3的卷积核大大降低了权重参数量**

### 多输入到多输出的演示：

<img src="C:\Users\84301\AppData\Roaming\Typora\typora-user-images\image-20241023094519131.png" alt="image-20241023094519131" style="zoom: 67%;" />

一个卷积核对应一个二维输入得到一个二维输出

三通道输入则对应三个卷积核，分别卷积后相加

两个输出则对应两组三个卷积核

故卷积核的数量应该等于输入乘输出

### 批量规范化（Batch Normalization）

对每一个小批量输入先做规范化

- 统一输入参数量级
- 不会造成不同层之间过大的差距，简化模型的收敛
- 深层神经网络容易过拟合，规范化作为一种正则化手段，降低了模型过拟合的概率

全连接层是在仿射变换和激活函数之间加入BN

卷积层是在卷积和激活函数之间加入BN

实际的BN不是简单的将小批量数据规范化到N(0, 1)，而是还引入了**拉伸参数**$\gamma$、**偏移参数**$\beta$，在训练过程中这两个参数也会学习更新

<img src="C:\Users\84301\AppData\Roaming\Typora\typora-user-images\image-20241023155823765.png" alt="image-20241023155823765" style="zoom:50%;" />

**解决的问题：**

内部协变量偏移（internal covariate shift）：经过多层的神经网络，每一层的输入不再属于同一概率分布

**注：**这里解释下此处的协变量，假设我们要拟合方程 $y=wx$，对于一个数据对$（x，y）$:$y$为因变量，$w$为自变量，$x$为协变量。

### 残差网络（Res Net）

#### DNN（深度神经网络）存在的问题

理论上，网络越深，参数的效果应该越好，但实际却并非如此，可能的问题有：过拟合（overfitting）、梯度消失/梯度爆炸

- **overfitting：**表现出来的应该是“高方差，低偏差”，即测试误差大而训练误差小。但实际上，深层CNN的训练误差和测试误差都很大。
- **梯度消失/爆炸：**由于参数更新是基于反向传播，而反向传播是基于链式法则求导，所以越靠近输入层的梯度所包含的乘积项就越多，所以如果多个小/大量相乘也许就会导致梯度消失/爆炸。但是通过引入BN已经基本上解决了这个问题

排除这两个问题，真正导致深层网络效果不佳的原因是模型随着网络的加深出现了**模型退化**现象。

#### 为何引入Res Net

<img src="C:\Users\84301\AppData\Roaming\Typora\typora-user-images\image-20241025160716322.png" alt="image-20241025160716322" style="zoom:50%;" />

从网络的角度来看，模型通过不断的加深网络来接近目标映射$f^*$。

深度学习通过引入非线性的处理使得模型能更充分的提取特征，但相对的也给予了过多的自由度。网络层数的加深给了这些自由度不可逆、不可控的发展，使得并没有更靠近理想映射。从图中可以看出，理想的情况在于，随着网络加深，后一层网络应该包含前一层网络的信息，这样才能保证模型是在向理想映射靠近。

通俗点讲，当浅层网络就已经训练出还不错的结果时，理论上，深层的网络**什么也不做**，也不会导致模型效果变差，但问题就在于非线性带来的灵活度很难使得模型**什么也不做**

形容这种什么也不做的能力叫做**恒等映射（identity mapping）**$f(x)=x$，我们希望模型至少要保持这样的特性。

#### 残差块的设计思路

要使模型学习到恒等映射的特性有两种办法：直接学习恒等映射、学习残差映射再加上原始输入

<img src="C:\Users\84301\AppData\Roaming\Typora\typora-user-images\image-20241025165537390.png" alt="image-20241025165537390" style="zoom:50%;" />

对比来看，两种拟合方式分别是拟合**输入**和拟合**输入输出的差，也就是残差**，而深层网络要准确的学习输入信息是很困难的，而学习残差其实省去了很多输入细节信息的学习，因此这样更简单。学习到残差之后，再加上原本的输入来拟合恒等映射。

