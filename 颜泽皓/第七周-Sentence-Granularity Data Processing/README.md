# 颜泽皓-第七周-Sentence-Granularity Data Processing

这一周的主要任务是将数据处理为以句子为粒度的单元，以及将源数据和BIO标记分开处理。

## 文档树

```
asci-utf8.py
bio_tagger.py
README.md
txt_process.py
```

## txt_process.py

将以下的标点符号纳入了筛选范围：

。？！【】，、；：「」『』’“”‘（）〔〕…–．—《》〈〉

并且以下的符号作为句分隔符：

。！．？

