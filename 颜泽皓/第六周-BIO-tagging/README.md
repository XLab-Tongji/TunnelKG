# 颜泽皓-第六周-BIO-tagging

这一周的工作内容是使用BIO标注法标注文章实体，并写好解析器代码。

## 文档树

git:

```
bio_parser.py
bio_tagger.py
pos.py
README.md
```

server:

```
./data_source:
bio_parser.py
biotxt.tar.gz
entities.csv
```

（原先的`data.csv`已修改为`entities.csv`，交换了后两列并去除了一些误插入的BOM头。）

## biotxt.tar.gz

带有BIO标记的文本。示例[^ bio_eg]如下：

> 黄\_O 衢\_O 南\_O 高\_B-n 速\_I-n 公\_I-n 路\_I-n 隧\_B-x 道\_I-x 工\_I-x 程\_I-x 中\_O 采\_O 用\_O 了\_O 大\_O 量\_O 的\_O 锚\_B-n 杆\_I-n 锚\_B-j 固\_I-j 围\_B-n 岩\_I-n 治\_O 理\_O 方\_B-n 案\_I-n 利\_O 用\_O 弹\_B-n 性\_I-n 波\_O 在\_O 不\_O 同\_O 波\_O 阻\_O 抗\_O 面\_O 反\_B-v 射\_I-v 的\_O 量\_O 和\_O 相\_O 位\_O 的\_O 变\_O 化\_O 原\_O 理\_O 采\_O 用\_O 特\_O 殊\_O 的\_O 声\_O 波\_O 发\_O 射\_O 和\_O 接\_O 收\_O 装\_B-n 置\_I-n 对\_O 该\_O 工\_B-n 程\_I-n 的\_O 锚\_B-n 杆\_I-n 锚\_B-j 固\_I-j 质\_B-n 量\_I-n 进\_O 行\_O 了\_O 检\_B-vn 测\_I-vn 结\_O 果\_O 表\_O 明\_O 锚\_B-n 杆\_I-n 锚\_B-j 固\_I-j 质\_B-n 量\_I-n 的\_O 无\_B-x 损\_I-x 检\_I-x 测\_I-x 不\_O 仅\_O 可\_O 行\_O 有\_O 效\_O 而\_O 且\_O 取\_O 得\_O 了\_O 较\_O 好\_O 的\_O 经\_O 济\_O 效\_O 益\_O 和\_O 社\_O 会\_O 效\_O 益\_O 隧\_B-n 道\_I-n 围\_B-n 岩\_I-n 工\_B-n 程\_I-n 锚\_B-n 杆\_I-n 锚\_B-j 固\_I-j 无\_B-x 损\_I-x 检\_I-x 测\_I-x 弹\_B-n 性\_I-n 波\_O 靠\_O 性\_O 评\_O 估\_O 是\_O 可\_O 信\_O 的\_O 这\_O 样\_O 就\_O 可\_O 以\_O 方\_O 便\_O 对\_O 任\_O 意\_O 失\_B-a 效\_I-a 概\_O 表\_O 用\_O 修\_O 正\_O 过\_O 的\_O 概\_O 率\_O 模\_B-n 型\_I-n 计\_O 算\_O 得\_O 到\_O 的\_O 在\_O 几\_O 种\_O 失\_B-a 效\_I-a 概\_O 率\_O 下\_O 桥\_B-n 梁\_I-n 结\_B-n 构\_I-n 的\_O 疲\_O 劳\_O 寿\_O 命\_O 结\_O 语\_O 本\_O 文\_O 主\_O 要\_O 是\_O 根\_O 据\_O 推\_O 荐\_O 的\_O 公\_B-n 式\_I-n 对\_O 桥\_B-n 梁\_I-n 构\_B-n 件\_I-n 的\_O 疲\_O 劳\_O 寿\_O 命\_O 进\_O 行\_O 初\_O 步\_O 预\_O 测\_O 其\_O 次\_O 是\_O 根\_O 据\_O 率\_O 下\_O 的\_O 疲\_O 劳\_O 寿\_O 命\_O 进\_O 行\_O 评\_O 估\_O 

[^ bio_eg]: \biotxt\CDP\黄衢南高速公路隧道锚杆锚固质量无损检测技术及应用_陈武.txt

## bio\_parser.py

使用[pyahocorasick](https://github.com/WojciechMula/pyahocorasick)库来实现多串匹配的Aho-Corasick算法。