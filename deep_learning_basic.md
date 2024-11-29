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

**线性回归到`softmax`回归主要是输出变多**

<img src="C:\Users\84301\AppData\Roaming\Typora\typora-user-images\image-20241022110953160.png" alt="image-20241022110953160" style="zoom: 50%;" /><img src="C:\Users\84301\AppData\Roaming\Typora\typora-user-images\image-20241022112100486.png" alt="image-20241022112100486" style="zoom:50%;" />

**多层感知机是在中间加入含有激活函数（非线性函数）的隐藏层，让特征提取更加泛化。**

**为了防止模型的过拟合问题，又引入了`dropout`(暂退法)，通过一定概率，在训练的时候忽略掉一些神经元。**

**判断网络是否是线性**的主要看是否引入非线性激活函数（ReLu，sigmoid，Tanh），除此之外，像dropout和maxpooling也会带来非线性操作。**softmax回归**的输出本质上还是由输入进行仿射变换得到，所以也属于线性回归的一种

### softmax+cross entropy

`softmax`是为了将输出统一到和为1，是一种规范化<img src="C:\Users\84301\AppData\Roaming\Typora\typora-user-images\image-20241018154536230.png" alt="image-20241018154536230" style="zoom:50%;" />

加上softmax之后可以进行概率意义上的比较

`cross entropy`是交叉熵损失<img src="C:\Users\84301\AppData\Roaming\Typora\typora-user-images\image-20241018155355444.png" alt="image-20241018155355444" style="zoom:50%;" />多用于分类问题的损失函数

`softmax+cross entrop`（下面的章节中详细介绍）：

<img src="C:\Users\84301\AppData\Roaming\Typora\typora-user-images\image-20241018155157837.png" alt="image-20241018155157837" style="zoom:50%;" />

基于损失函数为交叉熵的`softmax`的梯度为观测值y和估计值y^之间的差异

### 为何引入非线性激活函数

光是构建更多的隐藏层，最后将每一层的运算汇总到一起的时候，多层的线性运算其实等价于一层线性运算。神经网络拟合的权重终究是为了找到一个输入到输出的理想映射，而光是线性的模型是无法拟合非线性的数据的。

### 非线性激活函数

`ReLu:`<img src="C:\Users\84301\AppData\Roaming\Typora\typora-user-images\image-20241022170159865.png" alt="image-20241022170159865" style="zoom:50%;" />

**优点：**解决了 Sigmoid 和 Tanh 函数的梯度消失问题，当 x>0 时，梯度为常数，不会缩小。

**缺点：** **Dying ReLU** 问题：当输入小于等于 0 时，梯度为 0，可能导致神经元永远不更新，从而"死亡"。

![../_images/output_mlp_76f463_18_1.svg](https://zh.d2l.ai/_images/output_mlp_76f463_18_1.svg)

`sigmoid:`<img src="C:\Users\84301\AppData\Roaming\Typora\typora-user-images\image-20241022170345224.png" alt="image-20241022170345224" style="zoom:50%;" />

**优点：**常用于二分类问题，可以输出概率值。

![../_images/output_mlp_76f463_48_0.svg](https://zh.d2l.ai/_images/output_mlp_76f463_48_0.svg)

`Tanh:`<img src="C:\Users\84301\AppData\Roaming\Typora\typora-user-images\image-20241022170432862.png" alt="image-20241022170432862" style="zoom:50%;" />

**优点： ** **零中心化**：输出在 −1 到 1 之间，且零点对称。

![../_images/output_mlp_76f463_78_0.svg](https://zh.d2l.ai/_images/output_mlp_76f463_78_0.svg)

## 卷积神经网络（convolution neural network）

### 全连接层到卷积神经网络的过渡

全连接层多考虑一维输入输出，涉及到图像这种二维甚至三维（RGB通道）的输入时，全连接层会出现无法全面提取特征的弊端。

> 卷积神经网络里提到的卷积其实是互相关运算，数学意义上的卷积还需要将卷积核先进行上下左右反转之后运算。

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

三通道输入两个输出则对应两组三个卷积核

故卷积核的数量应该等于输入乘输出

### 批量规范化（Batch Normalization）

神经网络的本质是在学习数据的分布，而很重要的一点就是需要训练数据和测试数据是独立同分布的，而深层的网络会慢慢改变深层输入的分布，引入BN对数据做一个规范化可以有效的解决这一问题。

对每一个小批量输入先做规范化

- 统一输入参数量级
- 不会造成不同层之间过大的差距，简化模型的收敛
- 深层神经网络容易过拟合，规范化作为一种正则化手段，降低了模型过拟合的概率

**全连接层是在仿射变换和激活函数之间加入BN**

**卷积层是在卷积和激活函数之间加入BN**

如果只是简单的将数据规范化到N(0, 1)会削减模型的泛化能力，所以在此基础上，再引入拉伸和偏移

实际的BN不是简单的将小批量数据规范化到N(0, 1)，而是还引入了**拉伸参数**$\gamma$、**偏移参数**$\beta$，在训练过程中这两个参数也会学习更新

<img src="C:\Users\84301\AppData\Roaming\Typora\typora-user-images\image-20241023155823765.png" alt="image-20241023155823765" style="zoom:50%;" />

**解决的问题：**

内部协变量偏移（internal covariate shift）：经过多层的神经网络，每一层的输入不再属于同一概率分布

> 这里解释下此处的协变量，假设我们要拟合方程 $y=wx$，对于一个数据对$（x，y）$:$y$为因变量，$w$为自变量，$x$为协变量。
>

### 残差网络（ResNet）

#### 深度神经网络（DNN）存在的问题

理论上，网络越深，参数的效果应该越好，但实际却并非如此，可能的问题有：过拟合（overfitting）、梯度消失/梯度爆炸

- **overfitting：**表现出来的应该是“高方差，低偏差”，即测试误差大而训练误差小。但实际上，深层CNN的训练误差和测试误差都很大。
- **梯度消失/爆炸：**由于参数更新是基于反向传播，而反向传播是基于链式法则求导，所以越靠近输入层的梯度所包含的乘积项就越多，所以如果多个小/大量相乘也许就会导致梯度消失/爆炸。但是通过引入BN已经基本上解决了这个问题

排除这两个问题，真正导致深层网络效果不佳的原因是模型随着网络的加深出现了**模型退化**现象（深层网络的效果还不如浅层网络）。

#### 为何引入ResNet

<img src="C:\Users\84301\AppData\Roaming\Typora\typora-user-images\image-20241025160716322.png" alt="image-20241025160716322" style="zoom:50%;" />

​																	（左图是实际网络加深的效果，右图是理想网络加深的效果）

从网络的角度来看，模型通过不断的加深网络来接近目标映射$f^*$

深度学习通过引入非线性的处理使得模型能更充分的提取特征，但相对的也给予了过多的自由度。网络层数的加深给了这些自由度不可逆、不可控的发展，使得并没有更靠近理想映射。从图中可以看出，理想的情况在于，随着网络加深，后一层网络应该包含前一层网络的信息，这样才能保证模型是在向理想映射靠近。

通俗点讲，当浅层网络就已经训练出还不错的结果时，理论上，深层的网络**什么也不做**，也不会导致模型效果变差，但问题就在于非线性带来的灵活度很难使得模型**什么也不做**

形容这种什么也不做的能力叫做**恒等映射（identity mapping）**$f(x)=x$，我们希望模型至少要保持这样的特性。

#### 残差块的设计思路

要使模型学习到恒等映射的特性有两种办法：直接学习恒等映射、学习残差映射再加上原始输入

<img src="C:\Users\84301\AppData\Roaming\Typora\typora-user-images\image-20241025165537390.png" alt="image-20241025165537390" style="zoom:50%;" />

对比来看，两种拟合方式分别是拟合**输入**和拟合**输入输出的差，也就是残差**，而深层网络要准确的学习输入信息是很困难的，而学习残差其实省去了很多输入细节信息的学习，因此这样更简单。学习到残差之后，再加上原本的输入来拟合恒等映射。

### 稠密连接网络（DenseNet）

基于ResNet带来的启发，DenseNet做了逻辑上的进一步拓展。简单来看ResNet是将上一层的输入信息包含进来生成输出，而DenseNet则是将改层及以前所有层的输入都考虑进来了，而主要区别在于，ResNet是直接将输入加上输出，而DenseNet是将输入输出连接起来

ResNet：$f(x)=x+g(x)$

DenseNet：$x\rightarrow[x,f_1(x),f_2[x,f_1(x)],f_3(x,f_1(x),[x,f_1(x),f_2([x,f_1(x)]),\dots]$

稠密网络主要由两部分构成：*稠密块*（dense block）和*过渡层*（transition layer）。 前者定义如何连接输入和输出，而后者则控制通道数量，使其不会太复杂。

## 循环神经网络（recurrent neural network，RNN）

到目前为止，我们处理过两种数据：表格数据和图像数据。 这些数据有一些特点：都来自于某种分布， 并且所有样本都是独立同分布的。然而，大多数的数据并非如此。例如，文章中的单词是按顺序写的，如果顺序被随机地重排，就很难理解文章原始的意思。 同样，视频中的图像帧、对话中的音频信号以及网站上的浏览行为都是有顺序的。

简言之，如果说卷积神经网络可以有效地处理空间信息， 那么循环神经网络则可以更好地处理序列信息。 循环神经网络通过引入状态变量存储过去的信息和当前的输入，从而可以确定当前的输出。

### 词元（token）

对于输入序列为文本的情况，我们通常先将文本行数据词元化，对每一个文本行作分割得到token，再将token映射到数字作为索引，得到一个词表（vocabulary）（按照token出现频率分配索引）

> 词元就是组成序列的最小单位

### 语言模型（language model）

假设长度为$T$的文本序列中的token依次为$x_1,x_2,…,x_T$。 于是，$x_t(1≤t≤T)$ 可以被认为是文本序列在**时间步**$t$处的观测或标签。 在给定这样的文本序列时，**语言模型（language model）**的目标是估计序列的联合概率
$$
P(x_1,x_2,...,x_T)
$$
则抽到某个token，$x_t \sim P(x_t|x_1,...,x_t-1)$。

而包含了四个单词的一个文本序列的概率是：
$$
P(deep,learning,is,fun)=P(deep)P(learning|deep)P(is|deep,learning)P(fun|deep,learning,is)
$$

数据是基于时间序列的，其实存在一定的因果关系，而为了得到时间t时刻的概率并且想利用到时刻$t$前$t-1$步的信息，如果直接求上述的条件概率模型需要的参数量会巨大，不如使用隐变量模型：
$$
P(x_t|x_{t-1},\dots,x_1)\approx P(x_t|h_{t-1})
$$
其中$h_{t-1}$为隐状态，该变量抽象的概括了时间步$1$到$t-1$的信息
$$
h_t=f(x_t,h_{t-1})
$$
引入隐状态的神经网络就比单纯的全连接和卷积不一样了，多加入了一个隐状态权重$W_{hh}$
$$
H_t=\phi(X_tW_{xh}+H_{t-1}W_{hh}+b_h)
$$
从相邻时间步的隐藏变量$H_t$和 $H_{t−1}$之间的关系可知， 这些变量捕获并保留了序列直到其当前时间步的历史信息， 就如当前时间步下神经网络的状态或记忆， 因此这样的隐藏变量被称为**隐状态（hidden state）**。 由于在当前时间步中， 隐状态使用的定义与前一个时间步中使用的定义相同， 因此上式的计算是**循环的（recurrent）**。 

于是基于循环计算的隐状态神经网络被命名为 **循环神经网络**。 在循环神经网络中执行计上述计算的层 称为**循环层（recurrent layer）**。

<img src="C:\Users\84301\AppData\Roaming\Typora\typora-user-images\image-20241114165837067.png" alt="image-20241114165837067" style="zoom:50%;" />

​																						（可以看到输出是顺序生成的）

### 损失函数

为了评判语言模型的质量，引入了困惑度的概念。为了充分理解困惑度，先引入信息论中**熵**和**交叉熵**的概念

#### 熵

信息论中熵的概念首次被香农提出，目的是寻找一种高效/无损地编码信息的方法：以编码后数据的平均长度来衡量高效性，平均长度越小越高效；同时还需满足“无损”的条件，即编码后不能有原始信息的丢失。这样，香农提出了熵的定义：**无损编码事件信息的最小平均编码长度**。

我们首先定义每个**事件**的**信息量**。假设有一个事件 $x_i$ 发生的概率为 $P(x_i)$，根据信息量的定义，一个事件发生时所携带的信息量应该与该事件的概率相关。我们直觉上认为，事件发生得越不可能（即 $P(x_i)$ 越小），它所包含的信息量越大；反之，越可能发生的事件所包含的信息量越小。

因此，信息量可以定义为概率的倒数的对数值，常见的定义是：
$$
I(x_i) = -\log_b P(x_i)
$$
> $b$是进行编码的位数，例如二进制$b=2$
>

如果一个信息源是随机的，即从 $n$ 个可能事件中选择一个事件发生，那么我们想计算在这种情况下，平均每个事件携带的信息量。假设信息源选择事件 $x_i$ 的概率为 $P(x_i)$，那么事件 $x_i$ 对平均信息量的贡献为 $P(x_i) \cdot I(x_i)$，即：
$$
P(x_i) \cdot I(x_i) = - P(x_i) \cdot \log_b P(x_i)
$$
为了得到整个信息源的平均信息量，我们需要对所有可能的事件进行加权求和，得到信息源的平均信息量（即熵）：
$$
H(P) = \sum_{i=1}^{n} P(x_i) \cdot I(x_i) = -\sum_{i=1}^{n} P(x_i) \log_b P(x_i) = E_{x \sim P}[-logP(x)]
$$
这个公式就是**香农熵的定义公式**。

至于**信息量**与**编码长度**之间的联系可以举一个不太恰当的例子，假设某一个随机事件包含$N$种情况，每一种情况发生的概率都相等为$P(x_i) = \frac{1}{N}$，我们采用二进制编码，假设编码长度为$n$。在二进制的情况下，产生的编码情况为$2^n$（$n=3$时有：000，001，010，011，100，101，110，111，8种情况），这应该包含（等于）$N$种情况，即：
$$
2^n=N \\
n = log_2N = -log_2\frac{1}{N} = -log_2P(X_i)
$$
这与信息量的定义是统一的。

举个案例：

假设我们采用二进制编码东京的天气信息，并传输至纽约，其中东京的天气状态有4种可能，对应的概率如下图，每个可能性需要1个编码，东京的天气共需要4个编码。让我们采用3种编码方式，并对比下编码长度。

| 编码方式 | Fine(50%) | Cloudy(25%) | Rainy(12.5%) | Snow(12.5%) |                           编码长度                           |
| :------: | :-------: | :---------: | :----------: | :---------: | :----------------------------------------------------------: |
|    1     |    10     |     110     |      0       |     111     | 2$\times$50%+3$\times$25%+1$\times$12.5%+3$\times$12.5%=2.25 |
|    2     |     0     |     110     |      10      |     111     | 1$\times$50%+3$\times$25%+1$\times$12.5%+3$\times$12.5%=1.875 |
|    3     |     0     |     10      |     110      |     111     | 1$\times$50%+2$\times$25%+1$\times$12.5%+3$\times$12.5%=1.75 |

上表是枚举出来的结果，而直接用熵的计算公式可直接得到：
$$
Entropy=-(0.5 \log_2 0.5 + 0.25 \log_2 (0.25) + 0.125 \log_2 (0.125) + 0.125 \log_2 (0.125)) = 1.75
$$

#### 交叉熵

从上述熵的公式中可以看出，我们是知道一个随机变量的概率分布$P$的，但如果我们不知道数据的概率分布又想知道熵就只能对熵进行估计，**而熵的估计的过程自然而然的引出了交叉熵**。为了后续的讨论，我们要做出一个假设：对数据进行一定的观察之后，可以得到真实的概率分布$P$，而我们做出的估计为$Q$。

熵是无损最小平均编码长度，为了比较我们估计的概率分布和真实概率分布，引入交叉熵的定义：
$$
CrossEntropy=E_{x \sim P}[-log(Q(x))]=H(P,Q)
$$
上式代表用$P$计算期望，用$Q$计算编码长度，而熵是最小的情况，所以$H(P,Q) \ge H(P)$，下面我们讨论一个以交叉熵为损失函数的例子：

| Animal |      Dog      |      Fox      |     Horse     |     Eagle     |   squirrel    |
| :----: | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: |
| Label  | $[1,0,0,0,0]$ | $[0,1,0,0,0]$ | $[0,0,1,0,0]$ | $[0,0,0,1,0]$ | $[0,0,0,0,1]$ |

其中Label采用了one-hot编码（简单看就是用一个向量来表示各类概率，分类的情况有$N$种，该向量就有$N$个维度，并且分量的和为1，就是各类的概率和为1，最大分量即代表当前估计的种类，softmax就是对一些数据进行操作，得到这种和为1的向量），理想情况的Label就是对应位置为1，其余为0，$[1,0,0,0,0]$就说明当前情况百分百为Dog。

假设两个模型$Q_1,Q_2$对第一张图片进行预测，得到:

| Model |        Prediction         |
| :---: | :-----------------------: |
| $Q_1$ | $[0.4,0.3,0.05,0.05,0.2]$ |
| $Q_2$ |  $[0.98,0.01,0,0,0.01]$   |

真实标签为$[1,0,0,0,0]$
$$
H(P,Q_1)=-(1log(0.4)+0log(0.3)+0log(0.05)+0log(0.05)+0log(0.2)) \approx 0.916 \\
H(P,Q_1)=-(1log(0.98)+0log(0.01)+0log(0)+0log(0)+0log(0.01)) \approx 0.02 
$$
可以发现，预测的情况越好，交叉熵就越小，如果预测完全正确，交叉熵的值就为0，因此分类问题多用此作为损失函数。

#### 困惑度（perplexity）

困惑度的公式是直接对交叉熵取指数（根据交叉熵的底数变化，假设底数为$e$），定义为：
$$
exp(-\frac{1}{n}\Sigma^n_{t=1}logP(x_t|x_{t-1},\dots x_1))
$$
或者：
$$
\sqrt[N]{\frac{1}{P(x_1,\dots x_t)}}
$$

> 困惑度的最好的理解是“下一个token的实际选择数的调和平均数”。

从第二个式子来看，产生这个语句的概率（预测下一个词元的概率）越大，说明句子越可能常见、合理，所以困惑度越小，完全正确的时候，概率为1，困惑度也为1。而概率越小，说明句子可能越罕见，不合理，所以困惑度越大，概率趋于0的时候，困惑度趋于无穷。回到上面的一个特殊情况的时候，$\frac{1}{P}=N$，困惑度就能看作是token $x_t$实际可选的所有情况了。

### 梯度截断

对于长度为$T$的序列，我们在迭代中计算这$T$个时间步上的梯度， 将会在反向传播过程中产生长度为$O(T)$的矩阵乘法链。当$T$比较大的时候，就容易产生梯度爆炸的现象，为此我们约束一下梯度：
$$
g \leftarrow \min(1,\frac{\theta}{||g||})g=\min(||g||,\theta)\frac{g}{||g||}
$$
这样可以保证梯度的范数不会超过$\theta$

### 通过时间反向传播

**通过时间反向传播（backpropagation through time，BPTT）**实际上是循环神经网络中反向传播技术的一个特定应用。 它要求我们将循环神经网络的计算图一次展开一个时间步， 以获得模型变量和参数之间的依赖关系。 然后，基于链式法则，应用反向传播来计算和存储梯度。 由于序列可能相当长，因此依赖关系也可能相当长。 

为了保持简单，我们考虑一个没有偏置参数的循环神经网络， 其在隐藏层中的激活函数使用恒等映射$ϕ(x)=x$。 对于时间步$t$，设单个样本的输入及其对应的标签分别为 $x_t∈R_d$和$yt$。 计算隐状态$h_t∈R_h$和 输出$o_t∈R_q$，以及损失函数$L$的方式为：
$$
h_t=W_{hx}x_t+W_{hh}h_{t-1}\\
o_t=W{qh}h_t\\
L=\frac{1}{T}\Sigma^T_{t=1}l(o_t,y_t)
$$
<img src="C:\Users\84301\AppData\Roaming\Typora\typora-user-images\image-20241118170723475.png" alt="image-20241118170723475" style="zoom:50%;" />

​																											 (依赖图)

计算损失函数关于各参数的梯度
$$
\begin{split}\begin{aligned}
\frac{\partial L}{\partial \mathbf{W}_{hx}}
&= \sum_{t=1}^T \text{prod}\left(\frac{\partial L}{\partial \mathbf{h}_t}, \frac{\partial \mathbf{h}_t}{\partial \mathbf{W}_{hx}}\right)
= \sum_{t=1}^T \frac{\partial L}{\partial \mathbf{h}_t} \mathbf{x}_t^\top\\
\frac{\partial L}{\partial \mathbf{W}_{hh}}
&= \sum_{t=1}^T \text{prod}\left(\frac{\partial L}{\partial \mathbf{h}_t}, \frac{\partial \mathbf{h}_t}{\partial \mathbf{W}_{hh}}\right)
= \sum_{t=1}^T \frac{\partial L}{\partial \mathbf{h}_t} \mathbf{h}_{t-1}^\top
\end{aligned}\end{split}\\
\frac{\partial L}{\partial \mathbf{h}_t}= \sum_{i=t}^T {\left(\mathbf{W}_{hh}^\top\right)}^{T-i} \mathbf{W}_{qh}^\top \frac{\partial L}{\partial \mathbf{o}_{T+t-i}}
$$
可以发现$W_{hx}、W_{hh}$梯度的计算都依赖于$\frac{\partial L}{\partial \mathbf{h}_T}$，而它则取决于$W^T_{hh}$的幂，在这个幂中，小于1的特征值将会消失，大于1的特征值将会发散。 这在数值上是不稳定的，表现形式为梯度消失或梯度爆炸。

解决此问题的一种方法是按照计算方便的需要截断时间步长的尺寸，只需将求和终止为$\partial h_{t-\tau}/\partial w_h$。 在实践中，这种方式工作得很好。 它通常被称为**截断的通过时间反向传播**。 这样做导致该模型主要侧重于短期影响，而不是长期影响。 这在现实中是可取的，因为它会将估计值偏向更简单和更稳定的模型。

### 门控循环单元（GRU）

在RNN中出现的梯度异常的现象在实际中的体现：

- 如果在较早的观测对未来观测具有十分重要的影响，在计算中就会给予它较大的梯度
- 如果一些token对预测没有帮助，我们需要跳过这样的token
- 序列的各个部分之间存在逻辑中断，我们需要有类似过渡的机制

#### 重置门和更新门

**重置门（reset gate）**和**更新门（update gate）**。 我们把它们设计成$(0,1)$区间中的向量， 这样我们就可以进行凸组合。 重置门允许我们控制“可能还想记住”的过去状态的数量； 更新门将允许我们控制新状态中有多少个是旧状态的副本。

<img src="C:\Users\84301\AppData\Roaming\Typora\typora-user-images\image-20241118191832838.png" alt="image-20241118191832838" style="zoom:50%;" />

重置门$\mathbf{R}_t \in \mathbb{R}^{n \times h}$和 更新门$\mathbf{R}_t \in \mathbb{R}^{n \times h}$的计算如下所示：
$$
\begin{split}\begin{aligned}
\mathbf{R}_t = \sigma(\mathbf{X}_t \mathbf{W}_{xr} + \mathbf{H}_{t-1} \mathbf{W}_{hr} + \mathbf{b}_r)\\
\mathbf{Z}_t = \sigma(\mathbf{X}_t \mathbf{W}_{xz} + \mathbf{H}_{t-1} \mathbf{W}_{hz} + \mathbf{b}_z)
\end{aligned}\end{split}
$$

#### 候选隐状态

将重置门$R_t$与中的常规隐状态更新机制集成， 得到在时间步$t$的**候选隐状态（candidate hidden state）**$\tilde{\mathbf{H}}_t \in \mathbb{R}^{n \times h}$。
$$
\tilde{\mathbf{H}}_t = \tanh(\mathbf{X}_t \mathbf{W}_{xh} + \left(\mathbf{R}_t \odot \mathbf{H}_{t-1}\right) \mathbf{W}_{hh} + \mathbf{b}_h)
$$

> $\odot$是Hadamard积（按元素乘积）运算符

每当重置门$R_t$中的项接近1时， 我们恢复一个普通的循环神经网络。 对于重置门$R_t$中所有接近0的项， 候选隐状态是以$X_t$作为输入的多层感知机的结果。

<img src="C:\Users\84301\AppData\Roaming\Typora\typora-user-images\image-20241118192210608.png" alt="image-20241118192210608" style="zoom:50%;" />

#### 隐状态

这一步确定新的隐状态$\mathbf{H}_t \in \mathbb{R}^{n \times h}$在多大程度上来自旧的状态$\mathbf{H}_{t-1}$和 新的候选状态$\tilde{\mathbf{H}}_t$。
$$
\mathbf{H}_t = \mathbf{Z}_t \odot \mathbf{H}_{t-1}  + (1 - \mathbf{Z}_t) \odot \tilde{\mathbf{H}}_t
$$
每当更新门$Z_t$接近1时，模型就倾向只保留旧状态。 此时，来自$X_t$的信息基本上被忽略， 从而有效地跳过了依赖链条中的时间步$t$。 相反，当$Z_t$接近0时， 新的隐状态$H_t$就会接近候选隐状态$\tilde{\mathbf{H}}_t$。

<img src="C:\Users\84301\AppData\Roaming\Typora\typora-user-images\image-20241118192501942.png" alt="image-20241118192501942" style="zoom:50%;" />

总之，门控循环单元具有以下两个显著特征：

- 重置门有助于捕获序列中的短期依赖关系；
- 更新门有助于捕获序列中的长期依赖关系。

### 长短期记忆网络（LSTM）

隐变量模型存在着长期信息保存和短期输入缺失的问题。 解决这一问题的最早方法之一是**长短期存储器（long short-term memory，LSTM）**

为了控制记忆元，我们需要许多门。 其中一个门用来从单元中输出条目，我们将其称为**输出门（output gate）**。 另外一个门用来决定何时将数据读入单元，我们将其称为**输入门（input gate）**。 我们还需要一种机制来重置单元的内容，由**遗忘门（forget gate）**来管理， 这种设计的动机与**GRU**相同， 能够通过专用机制决定什么时候记忆或忽略隐状态中的输入。

#### 输入门、忘记门、输出门

<img src="C:\Users\84301\AppData\Roaming\Typora\typora-user-images\image-20241118193254854.png" alt="image-20241118193254854" style="zoom:50%;" />
$$
\begin{split}\begin{aligned}
\mathbf{I}_t &= \sigma(\mathbf{X}_t \mathbf{W}_{xi} + \mathbf{H}_{t-1} \mathbf{W}_{hi} + \mathbf{b}_i)\\
\mathbf{F}_t &= \sigma(\mathbf{X}_t \mathbf{W}_{xf} + \mathbf{H}_{t-1} \mathbf{W}_{hf} + \mathbf{b}_f)\\
\mathbf{O}_t &= \sigma(\mathbf{X}_t \mathbf{W}_{xo} + \mathbf{H}_{t-1} \mathbf{W}_{ho} + \mathbf{b}_o)
\end{aligned}\end{split}
$$
它们由三个具有$sigmoid$激活函数的全连接层处理， 以计算输入门、遗忘门和输出门的值。 因此，这三个门的值都在$(0,1)$的范围内。

#### 候选记忆元

它的计算与上面描述的三个门的计算类似， 但是使用$\tanh$函数作为激活函数，函数的值范围为$(−1,1)$。 下面导出在时间步$t$处的方程：
$$
\tilde{\mathbf{C}}_t = \text{tanh}(\mathbf{X}_t \mathbf{W}_{xc} + \mathbf{H}_{t-1} \mathbf{W}_{hc} + \mathbf{b}_c)
$$
<img src="C:\Users\84301\AppData\Roaming\Typora\typora-user-images\image-20241118193645010.png" alt="image-20241118193645010" style="zoom:50%;" />

#### 记忆元

在门控循环单元中，有一种机制来控制输入和遗忘（或跳过）。 类似地，在长短期记忆网络中，也有两个门用于这样的目的： 输入门$I_t$控制采用多少来自$\tilde{\mathbf{C}}_t$的新数据， 而遗忘门$F_t$控制保留多少过去的 记忆元$\mathbf{C}_{t-1} \in \mathbb{R}^{n \times h}$的内容。 使用按元素乘法，得出：

<img src="C:\Users\84301\AppData\Roaming\Typora\typora-user-images\image-20241118193846578.png" alt="image-20241118193846578" style="zoom:50%;" />
$$
\mathbf{C}_t = \mathbf{F}_t \odot \mathbf{C}_{t-1} + \mathbf{I}_t \odot \tilde{\mathbf{C}}_t
$$
如果遗忘门始终为$1$且输入门始终为$0$， 则过去的记忆元$\mathbf{C}_{t-1}$ 将随时间被保存并传递到当前时间步。 引入这种设计是为了缓解梯度消失问题， 并更好地捕获序列中的长距离依赖关系。

#### 隐状态

我们需要定义如何计算隐状态$\mathbf{H}_t \in \mathbb{R}^{n \times h}$， 这就是输出门发挥作用的地方。 在长短期记忆网络中，它仅仅是记忆元的$\tanh$的门控版本。 这就确保了$\mathbf{H}_t$的值始终在区间$(−1,1)$内：
$$
\mathbf{H}_t = \mathbf{O}_t \odot \tanh(\mathbf{C}_t)
$$
只要输出门接近$1$，我们就能够有效地将所有记忆信息传递给预测部分， 而对于输出门接近$0$，我们只保留记忆元内的所有信息，相当于当前时间步对预测帮助不大，而且输出门接近$0$本身也是当前信息不重要的一种表现。

<img src="C:\Users\84301\AppData\Roaming\Typora\typora-user-images\image-20241118194837459.png" alt="image-20241118194837459" style="zoom:50%;" />

### 深度循环神经网络

<img src="C:\Users\84301\AppData\Roaming\Typora\typora-user-images\image-20241118200101365.png" alt="image-20241118200101365" style="zoom:50%;" />

​																							 （深度循环神经网络结构）

与多层感知机一样，隐藏层数目$L$和隐藏单元数目$h$都是超参数。 也就是说，它们可以由我们调整的。 另外，用**GRU**或**LSTM**的隐状态来代替隐状态进行计算， 可以很容易地得到深度门控循环神经网络或深度长短期记忆神经网络。

### 双向循环神经网络

除了预测任务，还会遇到类似填空这种需要考虑上下文的情况，**双向循环神经网络（bidirectional RNNs）** 添加了反向传递信息的隐藏层，以便更灵活地处理此类信息。

<img src="C:\Users\84301\AppData\Roaming\Typora\typora-user-images\image-20241119104138284.png" alt="image-20241119104138284" style="zoom:50%;" />
$$
\begin{split}\begin{aligned}
\overrightarrow{\mathbf{H}}_t &= \phi(\mathbf{X}_t \mathbf{W}_{xh}^{(f)} + \overrightarrow{\mathbf{H}}_{t-1} \mathbf{W}_{hh}^{(f)}  + \mathbf{b}_h^{(f)})\\
\overleftarrow{\mathbf{H}}_t &= \phi(\mathbf{X}_t \mathbf{W}_{xh}^{(b)} + \overleftarrow{\mathbf{H}}_{t+1} \mathbf{W}_{hh}^{(b)}  + \mathbf{b}_h^{(b)})
\end{aligned}\end{split}
$$

> $\overrightarrow{\mathbf{H}}_t \in \mathbb{R}^{n \times h}、\overleftarrow{\mathbf{H}}_t \in \mathbb{R}^{n \times h}$，将前向隐状态$\overrightarrow{\mathbf{H}}_t$和反向隐状态$\overleftarrow{\mathbf{H}}_t$连接起来，获得需要送入输出层的隐状态$\mathbf{H}_t \in \mathbb{R}^{n \times 2h}$

$$
\mathbf{O}_t = \mathbf{H}_t \mathbf{W}_{hq} + \mathbf{b}_q
$$

### 编码器-解码器架构

语句的补充、预测大多是单语言的生成任务，可以直接通过自回归语言模型，这种方式输入的上下文直接作为生成的条件。

而在机器翻译中，源语言和目标语言的表征方式不同，直接用语言模型生成目标语言容易丢失源语言信息。

机器翻译是序列转换模型的一个核心问题， 其输入和输出都是**长度可变**的序列。 为了处理这种类型的输入和输出， 我们可以设计一个包含两个主要组件的架构： 第一个组件是一个**编码器（Encoder）**： 它接受一个长度可变的序列作为输入， 并将其转换为具有**固定形状的编码状态**。 第二个组件是**解码器（Decoder）**： 它将固定形状的编码状态映射到长度可变的序列。

<img src="C:\Users\84301\AppData\Roaming\Typora\typora-user-images\image-20241119110402436.png" alt="image-20241119110402436" style="zoom:50%;" />

> 在编码器接口中，我们只指定长度可变的序列作为编码器的输入`X`。
>
> 在解码器接口中，我们新增一个`init_state`函数， 用于将编码器的输出转换为编码后的状态。 **注意**，此步骤可能需要额外的输入，例如：输入序列的有效长度。 为了逐个地生成长度可变的词元序列， 解码器在每个时间步都会将输入 （例如：在前一时间步生成的词元）和编码后的状态 映射成当前时间步的输出词元。

### seq2seq

在机器翻译任务中，长度可变的输入输出实际上是一种**序列到序列（sequence to sequence）**的学习任务，可以使用两个RNN来构建Encoder和Decoder。

Encoder主要是将输入的seq信息转换到Encoder的**上下文变量$\mathbf{c}$（集成了Encoder每一时间步的隐状态）**作为Encoder的输出，然后作为Decoder的输入
$$
\mathbf{c} =  q(\mathbf{h}_1, \ldots, \mathbf{h}_T)
$$
![../_images/seq2seq.svg](https://zh.d2l.ai/_images/seq2seq.svg)

> 英语输入序列：("They", "are",  "watching", ".")
>
> 法语输出序列：("Ils", "regardent", ".")
>
> 序列的开始词元（“<bos>”）和结束词元（“<eos>”）

特定的“<bos>”表示序列开始词元，它是解码器的输入序列的第一个词元。

特定的“<eos>”表示序列结束词元。 一旦输出序列生成此词元，模型就会停止预测。

允许标签成为原始的输出序列， 从源序列词元“<bos>”“Ils”“regardent”“.” 到新序列词元 “Ils”“regardent”“.”“<eos>”来移动预测的位置。

![../_images/seq2seq-details.svg](https://zh.d2l.ai/_images/seq2seq-details.svg)

> 嵌入层用来提取序列的特征向量

### 预测序列的评估

原则上说，对于预测序列中的任意$n$元语法（n-grams）， **BLEU（bilingual evaluation understudy）**的评估都是这个$n$元语法是否出现在标签序列中。

我们将BLEU定义为：
$$
\exp\left(\min\left(0, 1 - \frac{\mathrm{len}_{\text{label}}}{\mathrm{len}_{\text{pred}}}\right)\right) \prod_{n=1}^k p_n^{1/2^n}
$$
其中$\mathrm{len}_{\text{label}}$表示标签序列中的词元数和 $\mathrm{len}_{\text{pred}}$表示预测序列中的词元数， $k$是用于匹配的最长的$n$元语法。

$p_n$表示$n$元语法的精确度，它是两个数量的比值： 第一个是预测序列与标签序列中匹配的$n$元语法的数量， 第二个是预测序列中$n$元语法的数量的比率。

## 注意力机制

注意力分为两种：自主性的与非自主性

自主性提示是主体主观意愿的推动，非自主性提示是基于环境中物体的突出性和易见性。

<img src="C:\Users\84301\AppData\Roaming\Typora\typora-user-images\image-20241119154755808.png" alt="image-20241119154755808" style="zoom:50%;" />

所有纸制品都是黑白印刷的，但咖啡杯是红色的。 换句话说，这个咖啡杯在这种视觉环境中是突出和显眼的， 不由自主地引起人们的注意**——非自主提示**

<img src="C:\Users\84301\AppData\Roaming\Typora\typora-user-images\image-20241119154929173.png" alt="image-20241119154929173" style="zoom:50%;" />

喝咖啡后，我们会变得兴奋并想读书， 所以转过头，重新聚焦眼睛，然后看看书。与上图中由于突出性导致的选择不同， 此时选择书是受到了认知和意识的控制， 因此注意力在基于自主性提示去辅助选择时将更为谨慎**——自主提示**

应用到神经网络的设计中，如果仅包含非自主性提示，则可以简单地使用参数化的全连接层， 甚至是非参数化的最大汇聚层或平均汇聚层。

> 上述的神经网络都是基于数据本身的特征

因此，“是否包含**自主性提示**”将注意力机制与全连接层或汇聚层区别开来。

在注意力机制的背景下，自主性提示被称为**查询（query）**。 给定任何查询，注意力机制通过**注意力汇聚（attention pooling）** 将选择引导至**感官输入（sensory inputs）**。 在注意力机制中，这些感官输入被称为**值（value）**。 更通俗的解释，每个值都与一个**键（key）**配对， 这可以想象为感官输入的非自主提示。 如图所示，可以通过设计注意力汇聚的方式， 便于给定的查询（自主性提示）与键（非自主性提示）进行匹配， 这将引导得出最匹配的值（感官输入）。

<img src="C:\Users\84301\AppData\Roaming\Typora\typora-user-images\image-20241119155412086.png" alt="image-20241119155412086" style="zoom:50%;" />

> 自主——query
>
> 非自助——key
>
> 注意力集中的地方——value

注意力汇聚则结合query和key**倾向选择**value，这就是注意力机制与全连接层，CNN等模型的差别

### 注意力评分函数

简单起见，考虑下面这个回归问题： 给定的成对的“输入－输出”数据集$\{(x_1, y_1), \ldots, (x_n, y_n)\}$， 如何学习$f$来预测任意新输入$x$的输出$\hat{y} = f(x)$？

**Nadaraya和 Watson提出了Nadaraya-Watson核回归（Nadaraya-Watson kernel regression）**，该模型特点在于根据输入的位置对输出$y_i$进行了加权：
$$
f(x) = \sum_{i=1}^n \frac{K(x - x_i)}{\sum_{j=1}^n K(x - x_j)} y_i
$$
$K$是核函数，具体实现不重要，受此启发， 从注意力机制框架的角度重写， 成为一个更加通用的**注意力汇聚（attention pooling）**公式：
$$
f(x) = \sum_{i=1}^n \alpha(x, x_i) y_i
$$

> 其中$x$是查询，$(x_i,y_i)$是键值对
>
> 注意力汇聚是$y_i$的加权平均
>
> query和key之间的关系为**注意力权重（attention weight）**$\alpha(x, x_i)$

对于任何查询，模型在所有键值对注意力权重都是一个有效的概率分布： 它们是非负的，并且总和为1

而计算注意力权重的函数就是注意力评分函数简称评分函数，注意力汇聚的输出就是基于这些注意力权重的值的加权和。

![../_images/attention-output.svg](https://zh.d2l.ai/_images/attention-output.svg)

用数学语言描述，假设有一个查询$\mathbf{q} \in \mathbb{R}^q$和 $m$个“键－值”对 $(\mathbf{k}_1, \mathbf{v}_1), \ldots, (\mathbf{k}_m, \mathbf{v}_m)$， 其中$\mathbf{k}_i \in \mathbb{R}^k$，$\mathbf{v}_i \in \mathbb{R}^v$，$a$是评分函数。 注意力汇聚函数$f$就被表示成值的加权和：
$$
f(\mathbf{q}, (\mathbf{k}_1, \mathbf{v}_1), \ldots, (\mathbf{k}_m, \mathbf{v}_m)) = \sum_{i=1}^m \alpha(\mathbf{q}, \mathbf{k}_i) \mathbf{v}_i \in \mathbb{R}^v\\
\alpha(\mathbf{q}, \mathbf{k}_i) = \mathrm{softmax}(a(\mathbf{q}, \mathbf{k}_i)) = \frac{\exp(a(\mathbf{q}, \mathbf{k}_i))}{\sum_{j=1}^m \exp(a(\mathbf{q}, \mathbf{k}_j))} \in \mathbb{R}
$$
softmax只是将计算出来的权重们映射到$(0,1)$区间并且和为$1$。

当然考虑到之前的问题中出现的`<pad>`等无效字符，还要为softmax加上一个掩蔽操作，也就是将有效值以外的输入清零

#### 加性注意力

当查询和键是不同长度的矢量时，可以使用加性注意力作为评分函数。给定查询$q∈R_q$和 键$k∈R_k$， **加性注意力（additive attention）**的评分函数为：
$$
a(\mathbf q, \mathbf k) = \mathbf w_v^\top \text{tanh}(\mathbf W_q\mathbf q + \mathbf W_k \mathbf k) \in \mathbb{R},
$$

> 其中可学习的参数是$W_q∈R_h×q、 W_k∈R_h×k$和 $w_v∈R_h$

#### 缩放点积注意力

使用点积可以得到计算效率更高的评分函数， 但是点积操作要求查询和键具有相同的长度$d$。 假设查询和键的所有元素都是独立的随机变量， 并且都满足零均值和单位方差， 那么两个向量的点积的均值为$0$，方差为$d$。 为确保无论向量长度如何， 点积的方差在不考虑向量长度的情况下仍然是$1$， 我们再将点积除以$d$， 则*缩放点积注意力*（scaled dot-product attention）评分函数为：
$$
a(\mathbf q, \mathbf k) = \mathbf{q}^\top \mathbf{k}  /\sqrt{d}.
$$
在实践中，我们通常从小批量的角度来考虑提高效率， 例如基于$n$个查询和$m$个键－值对计算注意力， 其中查询和键的长度为$d$，值的长度为$v$。 查询$Q∈R_n×d$、 键$K∈R_m×d$和 值$V∈R_m×v$的缩放点积注意力是：
$$
\mathrm{softmax}\left(\frac{\mathbf Q \mathbf K^\top }{\sqrt{d}}\right) \mathbf V \in \mathbb{R}^{n\times v}.
$$

### Bahdanau注意力

在考虑seq2seq问题的时候采用的是Encoder—Decoder架构，Encoder会产生一个上下文变量$\mathbf{c}$作为Decoder的输入，$\mathbf{c}$是通过一定手段集成了Encoder每一个时间步的隐状态的：

- 直接取最后一步的隐状态作为$\mathbf{c}$
- 拼接、平均每一个时间步的隐状态

然而考虑到一个问题，并不是所有的输入（源）词元都对解码某个词元都有用，那么应该设计出会变化的上下文变量。

Bahdanau注意力模型则是使用注意力机制，将上下文变量视为注意力集中的输出，为Decoder计算每一个时间步$t'$的上下文变量$\mathbf{c_{t'}}$。假设输入序列中有$T$个词元， 解码时间步$t'$下文变量是注意力集中的输出：
$$
\mathbf{c}_{t'} = \sum_{t=1}^T \alpha(\mathbf{s}_{t' - 1}, \mathbf{h}_t) \mathbf{h}_t
$$

> 时间步$t' - 1$时的解码器隐状态$\mathbf{s}_{t' - 1}$是query， 编码器隐状态$h_t$既是key，也是value

![../_images/seq2seq-attention-details.svg](https://zh.d2l.ai/_images/seq2seq-attention-details.svg)

### 多头注意力（Multi-Head Attention）

在实践中，当给定相同的查询、键和值的集合时， 我们希望模型可以基于相同的注意力机制学习到不同的行为， 然后将不同的行为作为知识组合起来， 捕获序列内各种范围的依赖关系 （例如，短距离依赖和长距离依赖关系）。 因此，允许注意力机制组合使用query、key和value的不同**子空间表示（representation subspaces）**可能是有益的。

为此，与其只使用单独一个注意力汇聚， 我们可以用独立学习得到的$h$组不同的**线性投影（linear projections）**来变换query、key和value。然后，这$h$组变换后的查询、键和值将并行地送到注意力汇聚中。 最后，将这$h$个注意力汇聚的输出拼接在一起， 并且通过另一个可以学习的线性投影进行变换， 以产生最终输出。

> 其实就是通过线性变换将一组(query, key, value)先变换到不同的(query, key, value)，再分别求注意力输出然后汇总

![../_images/multi-head-attention.svg](https://zh.d2l.ai/_images/multi-head-attention.svg)

multi-head attention的数学模型为：给定query $q∈R^{d_q}$、 key $k∈R^{d_k}$和value $v∈R^{d_v}$， 每个注意力头$h_i（i=1,…,h）$的计算方法为：
$$
\mathbf{h}_i = f(\mathbf W_i^{(q)}\mathbf q, \mathbf W_i^{(k)}\mathbf k,\mathbf W_i^{(v)}\mathbf v) \in \mathbb R^{p_v}\\
\begin{split}\mathbf W_o \begin{bmatrix}\mathbf h_1\\\vdots\\\mathbf h_h\end{bmatrix} \in \mathbb{R}^{p_o}\end{split}
$$
其中，可学习的参数包括$\mathbf W_i^{(q)}\in\mathbb R^{p_q\times d_q}$、$\mathbf W_i^{(k)}\in\mathbb R^{p_k\times d_k}$和$\mathbf W_i^{(v)}\in\mathbb R^{p_v\times d_v}$， 以及代表注意力汇聚的函数$f$。$f$可以是加性注意力和缩放点积注意力。 多头注意力的输出需要经过另一个线性转换， 它对应着$h$个头连结后的结果，因此其可学习参数是$\mathbf W_o\in\mathbb R^{p_o\times h p_v}$

基于这种设计，每个头都可能会关注输入的不同部分， 可以表示比简单加权平均值更复杂的函数。

### 自注意力（Self-Attention）

在深度学习中，经常使用卷积神经网络（CNN）或循环神经网络（RNN）对序列进行编码。 想象一下，有了注意力机制之后，我们将词元序列输入注意力池化中， 以便同一组词元同时充当查询、键和值。 具体来说，每个查询都会关注所有的键－值对并生成一个注意力输出。 由于查询、键和值来自同一组输入，因此被称为**自注意力（self-attention）**

给定一个由词元组成的输入序列$x_1,…,x_n$， 其中任意$x_i∈R_d（1≤i≤n）$。 该序列的自注意力输出为一个长度相同的序列$y_1,…,y_n$，其中：
$$
\mathbf{y}_i = f(\mathbf{x}_i, (\mathbf{x}_1, \mathbf{x}_1), \ldots, (\mathbf{x}_n, \mathbf{x}_n)) \in \mathbb{R}^d
$$
由于每一个输出都包含了所有输入的信息，所以self-attention允许模型捕获**长距离依赖关系**

#### 位置编码（Positional Encoding）

在处理词元序列时，循环神经网络是逐个的重复地处理词元的， 而自注意力则因为并行计算而放弃了顺序操作。 为了使用序列的顺序信息，通过在输入表示中添加**位置编码**来注入绝对的或相对的位置信息。 位置编码可以通过学习得到也可以直接固定得到。

假设输入表示$X∈R_{n×d}$ 包含一个序列中$n$个词元的$d$维嵌入表示。 位置编码使用相同形状的位置嵌入矩阵$P∈R_{n×d}$输出$X+P$， 矩阵第$i$行、第$2j$列和$2j+1$列上的元素为：
$$
\begin{split}\begin{aligned} p_{i, 2j} &= \sin\left(\frac{i}{10000^{2j/d}}\right)\\
p_{i, 2j+1} &= \cos\left(\frac{i}{10000^{2j/d}}\right)\end{aligned}\end{split}
$$

**绝对位置信息：**

编码维度上升则交替频率单调降低

```tex
0的二进制是：000
1的二进制是：001
2的二进制是：010
3的二进制是：011
4的二进制是：100
5的二进制是：101
6的二进制是：110
7的二进制是：111
```

**相对位置信息：**

除了捕获绝对位置信息之外，上述的位置编码还允许模型学习得到输入序列中相对位置信息。 这是因为对于任何确定的位置偏移$δ$，位置$i+δ$处 的位置编码可以线性投影位置$i$处的位置编码来表示。任何一对 $(p_{i,2j},p_{i,2j+1})$都可以线性投影到 $(p_{i+δ,2j},p_{i+δ,2j+1})$：
$$
\begin{split}\begin{aligned}
&\begin{bmatrix} \cos(\delta \omega_j) & \sin(\delta \omega_j) \\  -\sin(\delta \omega_j) & \cos(\delta \omega_j) \\ \end{bmatrix}
\begin{bmatrix} p_{i, 2j} \\  p_{i, 2j+1} \\ \end{bmatrix}\\
=&\begin{bmatrix} \cos(\delta \omega_j) \sin(i \omega_j) + \sin(\delta \omega_j) \cos(i \omega_j) \\  -\sin(\delta \omega_j) \sin(i \omega_j) + \cos(\delta \omega_j) \cos(i \omega_j) \\ \end{bmatrix}\\
=&\begin{bmatrix} \sin\left((i+\delta) \omega_j\right) \\  \cos\left((i+\delta) \omega_j\right) \\ \end{bmatrix}\\
=&
\begin{bmatrix} p_{i+\delta, 2j} \\  p_{i+\delta, 2j+1} \\ \end{bmatrix}
\end{aligned}\end{split}
$$

### Transformer

