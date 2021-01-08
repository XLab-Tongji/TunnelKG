# Detailed Design

## Background

As a key concept in natural language processing, knowledge graph (KG) presents an approach to organize knowledge points in certain fields and visualize their relations. Several successful instances of KG, such as *BioBERT* in biological medicine, has been published. But up till now, no instance of KG in civil engineering has been published yet. Thus, this project has an objective of constructing a knowledge graph in the field of civil engineering.

Following steps are taken in the construction of our KG: named entity recognition (NER), word embedding and relation extraction. NER recognize named entities (mostly field-related terms) and extracts them from sentence. Word embeddings maps entities to vectors, which is later used in finding the most unrelated entity in the sentence and giving related words of entity specified. Relation extraction classifies the relations between every two entities in the sentence given.

The KG constructed by us was initially aimed at tunnel engineering. With the generalization ability of model, we has extended the scope of KG into civil engineering currently.

## Process Flow Design

- Sentence processing process flow

![](./image/SentenceProcessSequenceDiagram.svg)

- Entity processing process flow

![](./image/EntityProcessSequenceDiagram.svg)



## Algorithm Design



### language modelling algorithm design

#### Background

From the machine learning perspective, language modelling is the modelling of the probability distribution of sentences with the aim of establishing a probability distribution that describes the occurrence of a given sequence of words in a language. This means that it is possible to determine whether a sequence of words is a normal utterance.

Word2vec characterizes the semantic information of words by learning the text and then using word vectors, i.e. mapping the original word space to a new space so that words that are semantically similar are close to each other in that space. The word is used for almost everything based on depth. 

Word2vec is used upstream in almost all deep learning-based NLP/knowledge mapping tasks, i.e. transforming text into vectors.

Also, Python has good packages for word vector training, namely Gensim, an open source third-party Python toolkit for unsupervised learning of topic vector representations from raw, unstructured text to the hidden layers of text. There are also good packages for Python to perform word separation on Chinese text, such as jieba.



#### Training process

1. familiar with the model: search word2vec related materials, understand the model principle, and build your own English word corpus and build the corresponding word2vec model, familiar with the model training API, such as the output word vector and word similarity, etc.
2. corpus collection: organize the corpus of papers crawled by the crawler and the Chinese wiki corpus as the base corpus in a folder
3. Corpus pre-processing: use jieba to sort the 6,000 pieces of corpus in turn and store them in the sorted corpus folder, keeping the file encoding format uniform (UTF-8/GBK)
4. model training: the processed corpus was used as input, and multiple files were read as a list of sentences, which were trained using the algorithm in the gensim library to obtain the word2vec model
5. model using: judge the similarity and mismatch of the input words
6. model comparison: the model was trained using the Chinese wiki corpus alone, and compared with the model with the pre-collection of civil engineering, to compare the prediction effect of similar words and word vector size of the same words



#### Training Improvements

1. Insufficient word separation accuracy during pre-processing.

   Introduced the civil engineering lexicon from the input method lexicon as a lexicon, improved the lexical separation and retrained the model to successfully improve the lexical separation accuracy

2. Poor prediction performance of the trained model.

   The model was improved by adding the Wikipedia corpus wiki_zh to the original corpus, mainly to improve the prediction of words that are general in nature.

3. Long training time, inconvenient to adjust parameters and recall

   Using gensim to provide an interface to save the model files locally, the model files are loaded directly when the interface is called or incremental training is required, eliminating the need for retraining.

   

#### Algorithm Characteristics and usage

1. transforming text into vectors
2. output the predicted results like similar words, irrelevant words and output similarity
3. adding Civil Engineering field corpus to fit the Civil Engineering word predict task



#### Key Operation

1. **pre-processing**

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

2. **model training**

```python
# 导入已处理语料
sentences = word2vec.LineSentence(wiki_file)
model = Word2Vec(sentences, min_count=5, window=5, size=256,
                 workers=multiprocessing.cpu_count(), iter=10,
                 sg=1, )
# 保存模型
model.save("./models/wiki_w2v_model.model")
```

3. **model using**

```python
ce_file = "./models/added_w2v_model.bin"
wiki_file = "./models/wiki_w2v_model.bin"

# 词向量规模
model_ce = KeyedVectors.load_word2vec_format(ce_file)
model_wk = KeyedVectors.load_word2vec_format(wiki_file)
var_ce = len(model_ce.vocab)
var_wk = len(model_wk.vocab)
var_sub = var_ce - var_wk
var_vs = model_ce.vector_size
var_vs2 = model_wk.vector_size
print("wiki语料对应模型词向量个数为：", var_wk)
print("添加了土木语料对应模型词向量个数为：", var_ce)
print("共添加词向量：", var_sub)
print("词向量维度为：", var_vs, " 和", var_vs2)

//预测相似词与无关词
similarity_1 = model.most_similar('隧道')
similarity_2 = model.most_similar('金属')
li = ["金属", "合金", "隧道"]
print("与隧道最相似的词为：", similarity_1)
print("与金属最相似的词为：", similarity_2)
print("金属, 合金, 隧道中差别最大的词为：", model.doesnt_match(li))
```



#### Algorithm results

The number of model word vectors for the wiki corpus is: 683473
The number of word vectors added to the civil engineering corpus is: 706148
Total number of word vectors added: 22675
Word vector dimensions: 256 and 256

Comparison of the test results before and after the addition of the corpus：

![w2v_wiki-1607436950006](image/w2v_wiki.png)



## Class Design

![language model class diagram](image/language model class diagram.png)

#### Corpus

A class for pre-process the corpus and dictionary

##### constructed function

``` python
def __init__(self, path, path_out, user_dict)
```

+ parameter:
  +  path: the input root path of raw corpus
  +  path_out: the output root path of processed corpus
  +  user_dict: the dictionary user defined

##### methods 

1. ``` python
   def dict(self, dict_entity)
   ```

   + parameter
     + dict_entity: the csv type of dictionary
   + return
     + None

2. ``` PYTHON
   def segment(self)
   ```

   + parameter:
     + None
   + return
     + None



#### Model

A class for training the pre-processed corpus

##### constructed function

``` python
def __init__(self, sg, min_count, window, size, iter):
```

+ parameter:
  +  sg: used to set the training algorithm, the default is 0, which corresponds to the CBOW algorithm; if sg is 1, the skip-gram algorithm is used.
  +  size: the vector dimension of the output words, default is 100. larger size requires more training data, but the result will be better. 
  +  window: the training window size, 8 means that each word is considered for the first 8 words and the last 8 words (there is a random window selection process in the actual code, window size <= 5), the default value is 5.
  +  min_count: allows the dictionary to be truncated. Words with a frequency of less than min_count are discarded, the default value is 5. 
  +  iter: the number of iterations, default is 5.  

##### methods 

1. ``` python
   def train(self, wiki_file, model_save_path):
   ```

   + parameter
     + wiki_file: the corpus extracted from wiki Chinese pages
     + model_save_path: the path where saved the model
   + return
     + model: the trained word2vec model


