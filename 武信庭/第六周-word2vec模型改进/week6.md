# 武信庭-第六周-word2vec模型改进

## 本周进度

使用土木词汇表作为分词词典改进分词效果并重新训练模型

## 1. word2vec

#### 实验成果

1. 将csv词典文件转为txt词典

```python
data = pd.read_csv('entities.csv', encoding='utf-8')
df = pd.DataFrame(data)
word = df["word"]
pos = df['pos']
length = df.shape[0]

with open('dict.txt', 'w+', encoding='utf-8') as f:
    for i in range(length):
        f.write(word[i] + ' 3 ' + pos[i] + '\n')
```

2. 导入词典并重新分词

```python
# 导入词典
jieba.load_userdict('dict.txt')

# 分词处理
files = os.listdir(path)
for file in files:
    position = path + '\\' + file
    position_out = path_out + '/' + file
    print(position)
    with open(position, "rb") as f:
        document = f.read()
        document_cut = jieba.cut(document)
        result = ' '.join(document_cut)
        result = result.encode('utf-8')
        with open(position_out, 'wb') as f2:
            f2.write(result)
    f.close()
    f2.close()
```

3. 训练模型，步骤与之前训练过程相同，增加保存模型步骤，支持本地快速加载并使用模型



## 2. 下周目标

1. 配置并训练elmo模型

