# 第六,七周 服务端&&pytorch中文关系提取数据预处理

### 本周进度

1. 将github上opennre项目封装为服务端, 进行关系提取任务
2. 继承并重写了pytorch的dataset类,实现在pytorch完成关系分类任务读取自定义中文数据集
3. 学习半监督学习算法:高斯混合模型

#### 服务端

1. 使用flask框架,输入样例为json字符串,分别代表进行关系抽取的句子和两实体在句子中的位置,样例中句子选自某英文小说:

   ```
   1: {"text": "He was the son of Máel Dúin mac Máele Fithrich, and grandson of the high king Áed Uaridnach(died 612).", "h": {"pos": [18, 46]}, "t": {"pos": [78, 91]}}
   2. {"text":"In the coffee-room, a tea-kettle was already surmounting the fire which Milka the ostler, as red in the face as a crab, was blowing with a pair of bellows." ,"h":{"pos":[7,18]},"t":{"pos":[22,32]}}
   ```

2. 根据每个输入,给出句子中两实体名,关系和预测的准确度

#### 中文数据预处理及重写dataset

**思路:**

1. 通过jieba包将训练集的所有句子分成词,将所有词统计至一字典中,并给每一个词设定一个id
2. 读取关系集中所有关系,为每个关系赋予id
3. 将数据集每一行的两实体,句子和关系转换为id序列或id并加入dataset中

#### 模型学习

相关学习笔记:https://github.com/QIANSUIMINGMINGMING/learningNote

### 下周预期:

1. 解决中文数据集读取的一些遗留问题
2. 仿照git项目opennre的代码实现简易的embedding+CNN/RNN+softmax关系分类

