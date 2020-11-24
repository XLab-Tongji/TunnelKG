# -*- coding: utf-8 -*-
import multiprocessing
import logging
import os
import sys
from gensim.models import word2vec

path_out = "./txt_out"

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
model.save("w2v_model.model")

similarity_n = model.wv.most_similar('隧道')
print("与隧道最相似的词为：", similarity_n)
