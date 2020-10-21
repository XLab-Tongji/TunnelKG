# 孙文麒-第二周-try_slot_tagging

* 在github上寻找相关代码仓库并尝试run通代码。

## 1 代码仓库

### 1.1 [bert_slot_tagging](https://github.com/yym6472/bert_slot_tagging)

* 用预训练BERT实现英文序列标注模型
* 使用allennlp, torch, bert
* 自带训练数据集
* 参数、数据集内容见仓库内容

#### 存在的问题

* allennlp的版本要求严格，向后兼容较差 1.0以上版本无法运行1.0以下版本的代码。

* 文档存在一些问题 测试代码跑不通 提示

  * ```sh
    allennlp.common.checks.ConfigurationError: 'bert_st is not a registered name for Predictor. You probably need to use the --include-package flag to load your custom code. Alternatively, you can specify your choices using fully-qualified paths, e.g. {"model": "my_module.models.MyModel"} in which case they will be automatically imported correctly.'
    ```

  * 预期在下周内解决

* allennlp仅支持linux mac，常用工作环境为windows，手头服务器上没有gpu，目前只能使用cpu进行测试，效率较低

#### 成果

* 目前保存了一个跑了20个epoch的model 打包成了model.tar.gz储存在服务器之中



### 1.2 [FewShotTagging](https://github.com/AtmaHou/FewShotTagging)

* 基于最新的ACL 2020论文 [Few-shot Slot Tagging with Collapsed Dependency Transfer and Label-enhanced Task-adaptive Projection Network](https://atmahou.github.io/attachments/atma's_acl2020_FewShot.pdf).
* 基于allennlp, pytorch, bert, pytorch-nlp, bert
* 作者提供了数据集

#### 存在的问题

* 代码需要linux环境下使用gpu执行，目前没有条件执行，因此暂时无法尝试该仓库的代码

### 1.3 [BERT-BiLSTM-CRF-NER](https://github.com/macanv/BERT-BiLSTM-CRF-NER)

* 使用谷歌的BERT模型在BLSTM-CRF模型上进行预训练用于中文命名实体识别的Tensorflow代码'

* 暂时还未对该仓库进行尝试



## 2 TO-DO

1. 将上文所提仓库1的测试代码跑通
2. 寻找一些公开可用的数据集 中英，中文优先
3. 建立带gpu的linux环境，尝试run通仓库2

