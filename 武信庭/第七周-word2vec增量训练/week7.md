# 武信庭-第六周-word2vec模型改进

## 本周进度

在原先语料库基础上加入维基百科语料库wiki_zh合并训练提升效果

## 1. word2vec改进

#### 实验成果

1. 下载wiki数据，约1.9G：https://dumps.wikimedia.org/zhwiki/latest/zhwiki-latest-pages-articles.xml.bz2

2. 提取压缩文件并转为单一txt：https://github.com/AwsomeName/ChineseWordEmbWithWiki/blob/master/DATA/WikiExtractor.py

3. 训练模型，步骤与之前训练过程相同，将上一步中转化完成的txt语料加入语料库中一并训练，读取日志如下

   PROGRESS: at sentence #4940000, processed 231314667 words, keeping 2978939 word types

4. 保存模型为word2vec模型与KeyedVectors模型，前者用以重新训练与增量训练，后者用以快速输出预测结果

