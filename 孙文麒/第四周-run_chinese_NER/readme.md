# 孙文麒-第四周-run_chinese_NER

## 1 代码仓库

### 1.1 [albert_chinese_NER](https://github.com/ProHiryu/albert-chinese-ner)

- 使用albert模型进行中文命令实体识别
- 自带数据集
- 使用tensorflow
- 存在问题
  - 由于CUDA版本问题，目前还没有训练完毕
  - 考虑寻找pytorch替代仓库

### 1.2 [BERT-NER](https://github.com/weizhepei/BERT-NER)

- 使用BERT模型进行中文/英文命令实体识别
- 中文使用MSRA数据集
- 使用pytorch
- 存在问题
  - BERT模型较大，训练难度较大

## TODO
- 在GPU上完成训练，并顺利预测
- 配置docker，暴露端口 为展示做准备

