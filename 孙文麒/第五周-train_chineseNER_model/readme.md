# 孙文麒-第五周-train_chineseNER_model

## 1 代码仓库

### 1.1 [albert_lstm_crf_ner](https://github.com/jiangnanboy/albert_lstm_crf_ner)

- 使用albert模型+lstm_crf进行中文命令实体识别
- 自带数据集
- 使用pytorch

**代码修改**

- 该仓库下src路径配置有些问题，执行python main.py会产生`No module`的问题，因此需要加入一些语句
- 该仓库需要将albert模型从tensorflow转至pytorch版本，但仓库本身没有提供转换的指导，因此需要依据src/albert下py文件 输入命令

**成果**

- 成功进行了训练、预测

## 2 一些坑

### 2.1 print 中文问题

[参考博客](https://blog.csdn.net/j___t/article/details/97705231)

**问题阐述**
当使用print打印unicode字符时，报错
```sh
UnicodeEncodeError: 'ascii' codec can't encode characters in position 1279-1280: ordinal not in range(128)
```

**解决方案**
- 安装中文utf-8

```sh
sudo apt-get -y install language-pack-zh-hans
```

- 修改`~/.bashrc`环境变量

```sh
export PYTHONIOENCODING=utf-8
```

## 3 TODO

- 模型嵌入