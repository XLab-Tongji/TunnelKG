# 闫沐西-第三周 opennre&linux

## 本周进度

### 1. 配置linux双系统以及linux下的GPU模式pytorch环境

+ 基本配置完成，仍存在问题：cuda版本与系统显卡驱动版本不匹配，导致pytorch训练时使用CPU进行计算，预计下周解决。

### 2. clone并配置运行github中的opennre等关系抽取相关项目

1. 数据集

   + 组成：由训练数据，校验数据，测试数据，以及关系-to-id数据

   + 样例：

     + 训练数据等：txt

     ```
     {"token": ["the", "most", "common", "audits", "were", "about", "waste", "and", "recycling", "."], "h": {"name": "audits", "pos": [3, 4]}, "t": {"name": "waste", "pos": [6, 7]}, "relation": "Message-Topic(e1,e2)"}
     ```

     + 关系-to-id：json
     
     ```
     "Instrument-Agency(e2,e1)": 2
     ```

2. 训练关系抽取模型的结构

   data->tokenization->encoder->selection model->classification model

## 下周目标

1. 处理好显卡驱动和cuda版本问题
2. 尝试使用中文数据集训练关系识别模型