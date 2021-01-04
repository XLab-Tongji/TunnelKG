[TOC]

# 隧道知识图谱-本学期进度小结

> 1852652 孙文麒



## 1 完成的工作
### 1.1 文献阅读

- BERT模型论文 
  - BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
  - Well-Read Students Learn Better: On the Importance of Pre-training Compact Models
- Attention 相关论文
  - Attention is all you need
- Few-shot Slot Tagging 论文
  - Few-shot Slot Tagging with Collapsed Dependency Transfer and Label-enhanced Task-adaptive Projection Network

### 1.2 英文模型训练
- [bert_slot_tagging](https://github.com/Nntraveler/bert_slot_tagging)
  - 用预训练BERT模型实现英文序列标注模型
  - 使用allennlp, pytorch, BERT
  - 参数、数据集内容见仓库内容
  - 对仓库进行了一定修改，修复了无法预测的问题
  - 对环境配置需求严格
  - 训练了20个epoch，精度已经达到0.99

### 1.3 中文模型训练

- [albert_lstm_crf_ner](https://github.com/Nntraveler/albert_lstm_crf_ner)
  - 使用ALBERT模型+lstm_crf进行中文命令实体识别
  - 使用pytorch实现
  - 训练了20个epoch，在通用数据集精确度达到0.99（识别`PER` `LOC` `ORG` `T`）
  - 在土木领域数据集训练了20个epoch，精确度在0.6左右，仍未完全收敛，猜测与数据问题有关(识别`x`)
  - 对仓库进行了一定修改以支持识别土木词汇(还未推到github仓库中)

### 1.4 土木领域命名实体识别数据集

- 土木数据集
  - 沿用爬取论文得到的数据
  - 只识别JIEBA分词下的`n`与`x`，且将`n`与`x`视作同一实体类
  - 选取了3000句作为训练集 200句作为测试集

## 2 遗留问题

- 土木数据集存在较多噪声
  - 由于是pdf转txt，部分内容存在错位、断层问题，例如作者信息插入到正文中
  - 标签并非人工标注，准确度方面并不理想
  - 使用标点符号分句，效果较差，大量句子并不成句
  - 存在大量纯`O`句子
- 未对土木标签进行进一步细分，目前只归类为`x`，可能影响到模型效果 
- 尚未对albert进行fine-tuning
