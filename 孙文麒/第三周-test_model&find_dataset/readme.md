# 孙文麒-第三周-test_model&find_dataset

* 对上周的bert_slot_tagging仓库代码进行修改，使得模型能够正常predict
* 寻找公开数据集

## 1 代码修改

### 1.1 上周遇到的错误

* 代码库：https://github.com/Nntraveler/bert_slot_tagging

* ```sh
  allennlp.common.checks.ConfigurationError: 'bert_st is not a registered name for Predictor. You probably need to use the --include-package flag to load your custom code. Alternatively, you can specify your choices using fully-qualified paths, e.g. {"model": "my_module.models.MyModel"} in which case they will be automatically imported correctly.'
  ```

* 通过阅读allennlp官方教程并对该错误进行了一定的搜索查询，判断原因主要出在predictor.py之中

### 1.2 问题剖析

* 该问题是由于bert_st为被注册为Predictor造成的，该问题的可能原因有两个

  * predictor未被包含在项目之中，此时需要--include-package或者声明路径
  * predictor的名字与config中设置不一致

* 在本项目中，问题是由于第二个造成的，在将predictor更改为bert_st后可正常运行。

  ```sh
  - @Predictor.register("slot_filling_predictor")
  + @Predictor.register("bert_st")
  ```

### 1.3 结果验证

```sh
(allennlp-0.9) root@iZwz9j3jcn4gcki3zkofq9Z:/usr/local/courses/software_engineering_project/bert_slot_tagging# python3 test.py --output_dir ./output/bert-atis/
{'tokens': ['show', 'me', 'the', 'first', 'class', 'and', 'coach', 'flights', 'between', 'jfk', 'and', 'orlando'], 'predict_labels': ['O', 'O', 'O', 'B-class_type', 'I-class_type', 'O', 'B-class_type', 'O', 'O', 'B-fromloc.city_name', 'O', 'B-toloc.city_name']}
```

| tokens  | results             |
| ------- | ------------------- |
| show    | 0                   |
| me      | 0                   |
| the     | 0                   |
| first   | B-class_type        |
| class   | I-class_type        |
| and     | 0                   |
| coach   | B-class_type        |
| flights | 0                   |
| between | 0                   |
| jfk     | B-fromloc.city_name |
| and     | 0                   |
| orlando | B-fromloc.city_name |



## 2 公开数据集

1. [ChineseNLPCorpus](https://github.com/InsaneLife/ChineseNLPCorpus)
   * 包含大量不仅限于命令实体识别的中文标注数据集
2. [ChineseNER](https://github.com/zjy-ucas/ChineseNER)
   * 数据位于./data中，主要用于命令实体识别的中文数据集

## 3 TODO

* 寻找可用linux环境 用于跑上周剩余两个项目
* 对[bert_slot_tagging](https://github.com/Nntraveler/bert_slot_tagging)仓库进行修改，进行一系列探索性试验

