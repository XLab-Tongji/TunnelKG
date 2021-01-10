## NER(Named Entity Recognition)

### Overview

Named-entity recognition (NER) (also known as (named) entity identification, entity chunking, and entity extraction) is a subtask of information extraction that seeks to locate and classify named entities mentioned in unstructured text into pre-defined categories such as person names, organizations, locations, medical codes, time expressions, quantities, monetary values, percentages, etc.

### Dataset

The dataset should be arranged in sentence level, and be separated into two files.

+ English dataset:

  + seq.in

  ```
  i want to fly from baltimore to dallas round trip
  round trip fares from baltimore to philadelphia less than 1000 dollars round trip fares from denver to philadelphia less than 1000 dollars round trip fares from pittsburgh to philadelphia less than 1000 dollars
  show me the flights arriving on baltimore on june fourteenth
  what are the flights which depart from san francisco fly to washington via indianapolis and arrive by 9 pm
  which airlines fly from boston to washington dc via other cities
  ```

  + seq.out

  ```
  O O O O O B-fromloc.city_name O B-toloc.city_name B-round_trip I-round_trip
  B-round_trip I-round_trip O O B-fromloc.city_name O B-toloc.city_name B-cost_relative O B-fare_amount I-fare_amount B-round_trip I-round_trip O O B-fromloc.city_name O B-toloc.city_name B-cost_relative O B-fare_amount I-fare_amount B-round_trip I-round_trip O O B-fromloc.city_name O B-toloc.city_name B-cost_relative O B-fare_amount I-fare_amount
  O O O O O O B-toloc.city_name O B-arrive_date.month_name B-arrive_date.day_number
  O O O O O O O B-fromloc.city_name I-fromloc.city_name O O B-toloc.city_name O B-stoploc.city_name O O B-arrive_time.time_relative B-arrive_time.time I-arrive_time.time
  O O O O B-fromloc.city_name O B-toloc.city_name B-toloc.state_code O O O
  ```

  

+ Chinese dataset: universal dataset is mainly from People's Daily which is already tagged. Tunnel-related dataset is collected and processed by our team.

  + universal.in

  ```
  人 民 网 1 月 1 日 讯 据 《 纽 约 时 报 》 报 道 , 美 国 华 尔 街 股 市 在 2 0 1 3 年 的 最 后 一 天 继 续 上 涨 , 和 全 球 股 市 一 样 , 都 以 最 高 纪 录 或 接 近 最 高 纪 录 结 束 本 年 的 交 易 。
  《 纽 约 时 报 》 报 道 说 , 标 普 5 0 0 指 数 今 年 上 升 2 9 . 6 % , 为 1 9 9 7 年 以 来 的 最 大 涨 幅 ; 道 琼 斯 工 业 平 均 指 数 上 升 2 6 . 5 % , 为 1 9 9 6 年 以 来 的 最 大 涨 幅 ; 纳 斯 达 克 上 涨 3 8 . 3 % 。
就 1 2 月 3 1 日 来 说 , 由 于 就 业 前 景 看 好 和 经 济 增 长 明 年 可 能 加 速 , 消 费 者 信 心 上 升 。 工 商 协 进 会 ( C o n f e r e n c e B o a r d ) 报 告 , 1 2 月 消 费 者 信 心 上 升 到 7 8 . 1 , 明 显 高 于 1 1 月 的 7 2 。
  另 据 《 华 尔 街 日 报 》 报 道 , 2 0 1 3 年 是 1 9 9 5 年 以 来 美 国 股 市 表 现 最 好 的 一 年 。 这 一 年 里 , 投 资 美 国 股 市 的 明 智 做 法 是 追 着 “ 傻 钱 ” 跑 。 所 谓 的 “ 傻 钱 ” 策 略 , 其 实 就 是 买 入 并 持 有 美 国 股 票 这 样 的 普 通 组 合 。 这 个 策 略 要 比 对 冲 基 金 和 其 它 专 业 投 资 者 使 用 的 更 为 复 杂 的 投 资 方 法 效 果 好 得 多 。 ( 老 任 )
人 民 网 平 壤 1 月 1 日 电 ( 记 者 王 莉 、 程 维 丹 ) 朝 鲜 最 高 领 导 人 金 正 恩 1 日 通 过 朝 鲜 各 大 媒 体 发 表 2 0 1 4 年 新 年 贺 词 , 向 朝 鲜 全 体 军 民 致 以 新 年 问 候 , 强 调 在 新 的 一 年 里 将 继 续 加 强 经 济 强 国 建 设 , 提 高 人 民 生 活 水 平 , 加 强 国 防 力 量 , 在 祖 国 统 一 斗 争 中 取 得 新 的 进 展 。 这 一 新 年 贺 词 被 视 为 朝 鲜 未 来 一 年 工 作 的 指 导 方 针 。
  ```
  
  + universal.out
  
  ```json
  O O O B_T I_T I_T I_T O O O B_LOC I_LOC O O O O O O B_LOC I_LOC I_LOC I_LOC I_LOC O O O B_T I_T I_T I_T I_T O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O
  O B_LOC I_LOC O O O O O O O O O O O O O O B_T I_T O O O O O O O O O B_T I_T I_T I_T I_T O O O O O O O O B_ORG I_ORG I_ORG O O O O O O O O O O O O O O O B_T I_T I_T I_T I_T O O O O O O O O B_PER I_PER O O O O O O O O O O
  O B_T I_T I_T I_T I_T I_T O O O O O O O O O O O O O O O O B_T I_T O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O B_T I_T I_T O O O O O O O O O O O O O O O O O B_T I_T I_T O O O O
  O O O B_LOC I_LOC I_LOC O O O O O O B_T I_T I_T I_T I_T O B_T I_T I_T I_T I_T O O B_LOC I_LOC O O O O O O O O O O O O O O O O O B_LOC I_LOC O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O B_LOC I_LOC O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O
  O O O B_LOC I_LOC I_T I_T I_T I_T O O O O B_PER I_PER O B_PER I_PER I_PER O B_LOC I_LOC O O O O O B_PER I_PER I_PER I_T I_T O O B_LOC I_LOC O O O O O O B_T I_T I_T I_T I_T I_T I_T O O O O B_LOC I_LOC O O O O O O B_T I_T O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O B_T I_T O O O O O B_LOC I_LOC I_T I_T O O O O O O O O O O
  ```
  
  + tunnel.in
  
  ```
  管 节 纵 断 面 迎 流 横 江 浮 运 红 谷 隧 道 管 节 纵 断 面 迎 流 横 江 浮 运 作 为 创 新 点 有 个 过 程 ： 一 是 管 节 绞 拉 出 坞 ， 干 坞 内 的 水 流 速 很 小 ， 而 管 节 出 坞 后 纵 断 面 受 横 向 水 流 的 影 响 ， 存 在 往 下 游 偏 移 搁 浅 的 重 大 风 险 ； 二 是 过 第 座 大 桥 前 ， 管 节 需 转 体 至 平 行 于 水 流 方 向 ， 转 体 过 程 中 ， 迎 流 面 积 从 最 大 变 到 最 小 ， 该 过 程 受 力 极 其 复 杂 ； 三 是 管 节 在 调 头 回 旋 区 打 横 调 头 及 进 隧 址 ， 受 江 心 洲 及 东 岸 围 堰 影 响 ， 河 道 宽 度 大 幅 度 缩 窄 ， 导 致 赣 江 东 汊 航 道 水 流 速 激 增 及 水 流 向 变 得 复 杂 ， 风 险 极 大 。 
  施 工 中 ， 首 次 创 新 采 用 了 “ 挂 拖 绑 拖 牵 拖 吊 拖 地 锚 ” 的 混 合 拖 航 浮 运 管 节 关 键 技 术 ， 解 决 了 上 述 风 险 。 
  窄 航 道 长 距 离 浮 运 管 节 红 谷 隧 道 为 国 内 首 座 江 河 中 游 沉 管 法 隧 道 ， 浮 运 航 道 自 干 坞 起 ， 沿 途 穿 越 生 米 大 桥 、 朝 阳 大 桥 和 南 昌 大 桥 ， 最 后 到 达 隧 址 ， 全 长 ， 受 季 节 性 降 水 影 响 ， 水 位 和 流 速 变 化 幅 度 大 ， 且 浮 运 航 道 距 离 长 ， 水 位 标 高 控 制 严 ， 航 道 窄 且 多 次 蜿 蜒 转 向 ， 施 工 风 险 大 ， 浮 运 窗 口 期 较 少 。 
  红 谷 隧 道 浮 运 航 道 平 面 示 意 图 如 图 所 示 。
  ```
  
  + tunnel.out
  
  ```
  O O B_x I_x I_x O O O O O O O O B_x I_x O O B_x I_x I_x O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O B_x I_x I_x O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O B_x I_x O O O O O O O O O B_x I_x O O O O O O O O O O O O O O O O O O O B_x I_x O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O B_x I_x O O O B_x I_x B_x I_x O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O 
  O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O 
  O O O O O O O O O O O O B_x I_x O O O O O O O O O O O O B_x I_x O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O B_x I_x O O O O O O O O O O O O O O O O O O B_x I_x B_x I_x O O O O O O O O O O O O O O O O O O O O O O O O O O O O O 
  O O B_x I_x O O O O B_x I_x I_x I_x I_x O O O O O 
  ```

### Evalution

| dataset                   | f1 value | precision |
| ------------------------- | -------- | --------- |
| English dataset           | 0.98     | 0.99      |
| Chinese universal dataset | 0.92     | 0.9       |
| Chinese tunnel dataset    | 0.5~0.7  | 0.6       |

### Key Operations

In this part, we will just focus on Chinese NER part.

#### data prepare: 

Use some scripts to transform the original dataset to meet the needs of NER.

#### Tokenization:

1. split sentence into word tokens
2. map tokens to numbers
3. add masks
4. wrap with [CLS] and [SEP] as an input to ALBERT
5. Use ALBERT output as embedding layer of LSTM

```python
def read_corpus(train_file_data, train_file_tag, max_length, label_dic):
    """
    :param train_file_data:训练数据
    :param train_file_tag: 训练数据对应的标签
    :param max_length: 训练数据每行的最大长度
    :param label_dic: 标签对应的索引
    :return:
    """

    VOCAB = config['albert_vocab_path']  # your path for model and vocab
    tokenizer = BertTokenizer.from_pretrained(VOCAB)
    result = []
    with open(train_file_data, 'r', encoding='utf-8') as file_train:
        with open(train_file_tag, 'r', encoding='utf-8') as file_tag:
            train_data = file_train.readlines()
            tag_data = file_tag.readlines()
            for text, label in zip(train_data, tag_data):
                tokens = text.split()
                label = label.split()
                if len(tokens) > max_length-2: #大于最大长度进行截断
                    tokens = tokens[0:(max_length-2)]
                    label = label[0:(max_length-2)]
                tokens_cs ='[CLS] ' + ' '.join(tokens) + ' [SEP]'
                label_cs = "[CLS] " + ' '.join(label) + ' [SEP]'
                # token -> index
                tokenized_text = tokenizer.tokenize(tokens_cs)  # 用tokenizer对句子分词
                input_ids  = tokenizer.convert_tokens_to_ids(tokenized_text)  # 索引列表

                # tag -> index
                label_ids = [label_dic[i] for i in label_cs.split()]
                input_mask = [1] * len(input_ids)

                while len(input_ids) < max_length:
                    input_ids.append(0)
                    input_mask.append(0)
                    label_ids.append(0)
                assert len(input_ids) == max_length
                assert len(input_mask) == max_length
                assert len(label_ids) == max_length
                feature = InputFeatures(input_id=input_ids, input_mask=input_mask, label_id=label_ids)
                result.append(feature)
    return result
```

#### Model

The model is based on ALBERT. We mainly introduce `BiLSTMCRF` part here.

![CRF-LAYER-1-v2](.\image\CRF-LAYER-1-v2.png)

+ **BiLSTM layer:** 

  The picture above illustrates that the outputs of BiLSTM layer are the scores of each label. For example, for w0w0，the outputs of BiLSTM node are 1.5 (B-Person), 0.9 (I-Person), 0.1 (B-Organization), 0.08 (I-Organization) and 0.05 (O). These scores will be the inputs of the CRF layer.

  ```python
  import numpy as np
  
  import torch
  from torch import nn
  
  from albert.model.modeling_albert import BertConfig, BertModel
  from albert.configs.base import config
  from lstm_crf.crf import CRF
  from sklearn.metrics import f1_score, classification_report
  class BiLSTMCRF(nn.Module):
  
      def __init__(
              self,
              tag_map={ 
                     'B_x': 0,
                     'I_x': 1,
                          'O': 2},
              batch_size=20,
              hidden_dim=128,
              dropout=1.0,
              embedding_dim=100
      ):
          super(BiLSTMCRF, self).__init__()
          self.batch_size = batch_size
          self.hidden_dim = hidden_dim
          self.embedding_dim = embedding_dim
          self.dropout = dropout
  
          self.tag_size = len(tag_map)  # 标签个数
          self.tag_map = tag_map
  
          bert_config = BertConfig.from_pretrained(str(config['albert_config_path']), share_type='all')
          self.word_embeddings = BertModel.from_pretrained(config['bert_dir'], config=bert_config)
          self.word_embeddings.to(DEVICE)
          self.word_embeddings.eval()
  
          self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim // 2,
                              num_layers=1, bidirectional=True, batch_first=True, dropout=self.dropout)
          self.hidden2tag = nn.Linear(self.hidden_dim, self.tag_size)
          self.crf = CRF(self.tag_size)
  ```

+ **CRF layer:** 

  Then, all the scores predicted by the BiLSTM blocks are fed into the CRF layer. In the CRF layer, the label sequence which has the highest prediction score would be selected as the best answer.

  ```python
  class CRF(nn.Module):
      """线性条件随机场"""
      def __init__(self, num_tag):
          if num_tag <= 0:
              raise ValueError("Invalid value of num_tag: %d" % num_tag)
          super(CRF, self).__init__()
          self.num_tag = num_tag
          self.start_tag = num_tag
          self.end_tag = num_tag + 1
          # 转移矩阵transitions：P_jk 表示从tag_j到tag_k的概率
          # P_j* 表示所有从tag_j出发的边
          # P_*k 表示所有到tag_k的边
          self.transitions = nn.Parameter(torch.Tensor(num_tag + 2, num_tag + 2))
          nn.init.uniform_(self.transitions, -0.1, 0.1)
          self.transitions.data[self.end_tag, :] = -10000   # 表示从EOS->其他标签为不可能事件, 如果发生，则产生一个极大的损失
          self.transitions.data[:, self.start_tag] = -10000   # 表示从其他标签->SOS为不可能事件, 同上
  
      def real_path_score(self, features, tags):
          """
          features: (seq_len, num_tag)
          tags:real tags
          real_path_score表示真实路径分数
          它由Emission score和Transition score两部分相加组成
          Emission score由LSTM输出结合真实的tag决定，表示我们希望由输出得到真实的标签
          Transition score则是crf层需要进行训练的参数，它是随机初始化的，表示标签序列前后间的约束关系（转移概率）
          Transition矩阵存储的是标签序列相互间的约束关系
          在训练的过程中，希望real_path_score最高，因为这是所有路径中最可能的路径
          """
          r = torch.LongTensor(range(features.size(0)))
          r = r.to(DEVICE)
          pad_start_tags = torch.cat([torch.LongTensor([self.start_tag]).to(DEVICE), tags])
          pad_stop_tags = torch.cat([tags, torch.LongTensor([self.end_tag]).to(DEVICE)])
          # Transition score + Emission score
          score = torch.sum(self.transitions[pad_start_tags, pad_stop_tags]) + torch.sum(features[r, tags])
          return score
  ```

#### Train

Set configuration and train the model.

```python
    def train(self):
        self.model.to(DEVICE)
        #weight decay是放在正则项（regularization）前面的一个系数，正则项一般指示模型的复杂度，所以weight decay的作用是调节模型复杂度对损失函数的影响，若weight decay很大，则复杂的模型损失函数的值也就大。
        optimizer = optim.Adam(self.model.parameters(), lr = 0.001, weight_decay=0.0005)
        '''
        当网络的评价指标不在提升的时候，可以通过降低网络的学习率来提高网络性能:
        optimer指的是网络的优化器
        mode (str) ，可选择‘min’或者‘max’，min表示当监控量停止下降的时候，学习率将减小，max表示当监控量停止上升的时候，学习率将减小。默认值为‘min’
        factor 学习率每次降低多少，new_lr = old_lr * factor
        patience=10，容忍网路的性能不提升的次数，高于这个次数就降低学习率
        verbose（bool） - 如果为True，则为每次更新向stdout输出一条消息。 默认值：False
        threshold（float） - 测量新最佳值的阈值，仅关注重大变化。 默认值：1e-4
        cooldown(int)： 冷却时间“，当调整学习率之后，让学习率调整策略冷静一下，让模型再训练一段时间，再重启监测模式。
        min_lr(float or list):学习率下限，可为 float，或者 list，当有多个参数组时，可用 list 进行设置。
        eps(float):学习率衰减的最小值，当学习率变化小于 eps 时，则不调整学习率。
        '''
        # schedule = ReduceLROnPlateau(optimizer=optimizer, mode='min',factor=0.1,patience=100,verbose=False)
        total_size = self.train_data.train_dataloader.__len__()
        for epoch in range(10):
            index = 0
            print(sys.getdefaultencoding())
            for batch in self.train_data.train_dataloader:
                self.model.train()
                index += 1
                self.model.zero_grad()  # 与optimizer.zero_grad()作用一样
                batch = tuple(t.to(DEVICE) for t in batch)
                b_input_ids, b_input_mask, b_labels, b_out_masks = batch

                bert_encode = self.model(b_input_ids, b_input_mask)
                loss = self.model.loss_fn(bert_encode=bert_encode, tags=b_labels, output_mask=b_out_masks)
                progress = ("#" * int(index * 25 / total_size)).ljust(25)
                print("""epoch [{}] |{}| {}/{}\n\tloss {:.2f}""".format(
                    epoch, progress, index, total_size, loss.item()))
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(),1) #梯度裁剪
                optimizer.step()
                # schedule.step(loss)
                if index % 10 == 0:
                    self.eval_2()
                    print("-" * 50)
        torch.save(self.model.state_dict(), self.model_path + 'params.pkl')
        self.eval_2()
        print("-" * 50)
```

#### Predict

Get a input string having a sentence, then return the entity.

```python
    def predict(self, input_str=""):
        self.model.eval()  # 取消batchnorm和dropout,用于评估阶段
        self.model.to(DEVICE)
        VOCAB = config['albert_vocab_path']  # your path for model and vocab
        tokenizer = BertTokenizer.from_pretrained(VOCAB)
        with torch.no_grad():
            #input_str = input("请输入文本: ")
            input_ids = tokenizer.encode(input_str,add_special_tokens=True)  # add_spicial_tokens=True，为自动为sentence加上[CLS]和[SEP]
            input_mask = [1] * len(input_ids)
            output_mask = [0] + [1] * (len(input_ids) - 2) + [0]  # 用于屏蔽特殊token

            input_ids_tensor = torch.LongTensor(input_ids).reshape(1, -1)
            input_mask_tensor = torch.LongTensor(input_mask).reshape(1, -1)
            output_mask_tensor = torch.LongTensor(output_mask).reshape(1, -1)
            input_ids_tensor = input_ids_tensor.to(DEVICE)
            input_mask_tensor = input_mask_tensor.to(DEVICE)
            output_mask_tensor = output_mask_tensor.to(DEVICE)

            bert_encode = self.model(input_ids_tensor, input_mask_tensor)
            predicts = self.model.predict(bert_encode, output_mask_tensor)

            #print('paths:{}'.format(predicts))
            entities = []
            for tag in self.tags:
                tags = get_tags(predicts[0], tag, self.model.tag_map)
                entities += format_result(tags, input_str, tag)
            print(entities)
            return entities
```

### Class Design

#### Overview

![](.\image\NER.png)

#### torch.nn.Module

A Python library's class, for building layers in neural networks. 

#### NER

The main class in the project, implement NER methods.

**constructed function**

```python
    def __init__(self, exec_type="train"):
        self.load_config()
        self.__init_model(exec_type)

    def __init_model(self, exec_type):
        if exec_type == "train":
            self.train_data = DataFormat(batch_size=self.batch_size, max_length=self.max_legnth, data_type='train')
            self.dev_data = DataFormat(batch_size=16, max_length=self.max_legnth, data_type="dev")

            self.model = BiLSTMCRF(
                tag_map=self.train_data.tag_map,
                batch_size=self.batch_size,
                dropout=self.dropout,
                embedding_dim=self.embedding_size,
                hidden_dim=self.hidden_size,
            )
            self.restore_model()

        elif exec_type == "eval":
            self.train_data = DataFormat(batch_size=self.batch_size, max_length=self.max_legnth, data_type='train')
            self.dev_data = DataFormat(batch_size=16, max_length=self.max_legnth, data_type="dev")

            self.model = BiLSTMCRF(
                tag_map=self.train_data.tag_map,
                batch_size=self.batch_size,
                dropout=self.dropout,
                embedding_dim=self.embedding_size,
                hidden_dim=self.hidden_size,
            )
            self.restore_model()


        elif exec_type == "predict":
            self.model = BiLSTMCRF(
                dropout=self.dropout,
                embedding_dim=self.embedding_size,
                hidden_dim=self.hidden_size
            )
            self.restore_model()
```

**methods**

1. ```python
       def train(self):
   ```

   + parameters:
     + None
   + return:
     + None
   
2. ```python
      def predict(self, input_str=""):
   ```

   - parameters:
     - input_str:  the sentence to be used to predict entities
   - return:
     - entities: predicted entities

#### DataFormat

A class implementing data formatting

##### constructed function

```python
    def __init__(self, max_length=100, batch_size=20, data_type='train'):
        self.index = 0
        self.input_size = 0
        self.batch_size = batch_size
        self.max_length = max_length
        self.data_type = data_type
        self.train_data = []
        self.tag_map = {
                   'B_x': 0,
                   'I_x': 1,
                        'O': 2}
        base_path = os.path.abspath(os.path.join(os.getcwd(), "../.."))
        if data_type == "train":
            self.data_path = base_path + '/data/ner_data/train/'
        elif data_type == "dev":
            self.data_path = base_path + "/data/ner_data/dev/"
        elif data_type == "test":
            self.data_path = base_path + "/data/ner_data/test/"

        self.read_corpus(self.data_path + 'source.txt', self.data_path + 'target.txt', self.max_length, self.tag_map)
        self.train_dataloader= self.prepare_batch(self.train_data, self.batch_size)
```

##### methods

1. ```python
   def read_corpus(self, train_file_data, train_file_tag, max_length, label_dic):
   ```

   + parameter:
     + train_file_data: data used for training
     + train_file_tag: tag for data
     + max_length:  max length for each line
     + label_dic: dictionary for labels
   + return
     + None

2. ```python
       def prepare_batch(self, train_data, batch_size):
   ```

   + parameter:
     + train_data: data used for training
     + batch_size: size of each batch
   + return:
     + DataLoader: class for training

#### BiLSTMCRF

A class implement NER models.

##### constructed function

```python
    def __init__(
            self,
            tag_map={ 
                   'B_x': 0,
                   'I_x': 1,
                        'O': 2},
            batch_size=20,
            hidden_dim=128,
            dropout=1.0,
            embedding_dim=100
    ):
        super(BiLSTMCRF, self).__init__()
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.dropout = dropout

        self.tag_size = len(tag_map)  # 标签个数
        self.tag_map = tag_map

        bert_config = BertConfig.from_pretrained(str(config['albert_config_path']), share_type='all')
        self.word_embeddings = BertModel.from_pretrained(config['bert_dir'], config=bert_config)
        self.word_embeddings.to(DEVICE)
        self.word_embeddings.eval()

        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim // 2,
                            num_layers=1, bidirectional=True, batch_first=True, dropout=self.dropout)
        self.hidden2tag = nn.Linear(self.hidden_dim, self.tag_size)
        self.crf = CRF(self.tag_size)
```

##### methods

1. ```python
       def predict(self, bert_encode, output_mask):
   ```

   + parameter:
     + bert_encode: output of ALBERT
     + output_mask: output of LSTM
   + return:
  + predicts: prediction

2. ```python
       def forward(self, input_ids, attention_mask):
   ```

   + parameter
     + input_ids: input id for word_embeddings
     + attention_mask: mask for word_embeddings
   + return
  + output: output of lstm

#### CRF

A class implements CRF layer

##### constructed function

``` python
    def __init__(self, num_tag):
        if num_tag <= 0:
            raise ValueError("Invalid value of num_tag: %d" % num_tag)
        super(CRF, self).__init__()
        self.num_tag = num_tag
        self.start_tag = num_tag
        self.end_tag = num_tag + 1
        # 转移矩阵transitions：P_jk 表示从tag_j到tag_k的概率
        # P_j* 表示所有从tag_j出发的边
        # P_*k 表示所有到tag_k的边
        self.transitions = nn.Parameter(torch.Tensor(num_tag + 2, num_tag + 2))
        nn.init.uniform_(self.transitions, -0.1, 0.1)
        self.transitions.data[self.end_tag, :] = -10000   # 表示从EOS->其他标签为不可能事件, 如果发生，则产生一个极大的损失
        self.transitions.data[:, self.start_tag] = -10000   # 表示从其他标签->SOS为不可能事件, 同上
```

##### methods 

1. ``` python
       def negative_log_loss(self, inputs, output_mask, tags):
   ```

   + parameter
     + inputs: source data inputs
     + output_mask: output mask of last layer
     + tags: tags without CLS&SEP
+ return
     + loss: log loss of given input

2. ``` PYTHON
       def get_batch_best_path(self, inputs, output_mask):
   ```

   + parameter:
     + inputs: input of length and name tags
     + output_mask: data used for getting best path
   + return
     + batch_best_path: best path of batch

