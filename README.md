# motion_capture

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

深度学习的核心在于深度神经网络，而神经网络其实是机器学习的一种模型，深度学习主要是采用了多层的神经网络

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

线性回归->softmax回归

预测->分类

<img src="C:\Users\84301\AppData\Roaming\Typora\typora-user-images\image-20241018154112529.png" alt="image-20241018154112529" style="zoom: 50%;" /><img src="C:\Users\84301\AppData\Roaming\Typora\typora-user-images\image-20241018154152139.png" alt="image-20241018154152139" style="zoom:50%;" />

softmax是为了将输出统一到和为1，是一种规范化<img src="C:\Users\84301\AppData\Roaming\Typora\typora-user-images\image-20241018154536230.png" alt="image-20241018154536230" style="zoom:50%;" />

cross entropy是交叉熵损失<img src="C:\Users\84301\AppData\Roaming\Typora\typora-user-images\image-20241018155355444.png" alt="image-20241018155355444" style="zoom:50%;" />多用于分类问题的损失函数

softmax+cross entrop：

<img src="C:\Users\84301\AppData\Roaming\Typora\typora-user-images\image-20241018155157837.png" alt="image-20241018155157837" style="zoom:50%;" />

基于损失函数为交叉熵的softmax的梯度为观测值y和估计值y^之间的差异

