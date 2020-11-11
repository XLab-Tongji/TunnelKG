# 武信庭-第五周-albert

## 本周进度

配置albert环境并运行fine-tuning的demo

## 1. albert

#### 实验成果

1. 环境配置：对应repo中的readme安装pytorch, cuda, cudnn, sklearn, sentencepiece等依赖库，实验环境搭建在本机（win10+2080+python3.7）

2.  下载预训练pytorch模型[albert_base_v2.zip](https://drive.google.com/open?id=1byZQmWDgyhrLpj8oXtxBG6AA52c8IHE-)并解压放入prev_trained_model文件夹

3. 转换tensorflow checkpoint为pytorch checkpoint

4. 通过[python脚本](https://gist.github.com/W4ngatang/60c2bdb54d156a41194446737ce03e2e)下载[General Language Understanding Evaluation (GLUE) benchmark](https://gluebenchmark.com/) 语句理解测试任务并解压至相应dataset文件夹

5. 运行shell脚本对模型进行微调并测试模型效果

   ```shell
   sh scripts/run_classifier_sst2.sh
   ```
   
6. 在output文件夹中通过对应日志与结果文档得到测试结果：

   acc_ = 0.9197247706422018

#### 遗留实验：

1. 研究代码细节与微调接口
2. 利用新语料词典重新训练word2vec模型

