# -*- coding: utf-8 -*-
import multiprocessing
import jieba
import logging
import os
import sys
from gensim.models import word2vec

path = "./txt"
path_out = "./txt_out"

# 分词处理
# files = os.listdir(path)
# for file in files:
#     position = path + '\\' + file
#     position_out = path_out + '/' + file
#     print(position)
#     with open(position, "rb") as f:
#         document = f.read()
#         document_cut = jieba.cut(document)
#         result = ' '.join(document_cut)
#         result = result.encode('utf-8')
#         with open(position_out, 'wb') as f2:
#             f2.write(result)
#     f.close()
#     f2.close()

# 训练模型
program = os.path.basename(sys.argv[0])
logger = logging.getLogger(program)
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logging.root.setLevel(level=logging.INFO)
logger.info("running %s" % ' '.join(sys.argv))

sentences = word2vec.PathLineSentences(path_out)

model = word2vec.Word2Vec(sentences, min_count=5, window=5, size=256,
                          workers=multiprocessing.cpu_count(), iter=10,
                          sg=1, )

similarity = model.wv.most_similar('土木工程')
print("与土木工程最相似的词为：", similarity)
