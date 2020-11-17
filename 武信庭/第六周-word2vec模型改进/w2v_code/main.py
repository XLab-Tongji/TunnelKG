from gensim.models import KeyedVectors
import numpy

model = KeyedVectors.load("w2v_model.model")
similarity_n = model.most_similar('隧道')
li = ["隧道", "桥梁", "周边"]
print("与隧道最相似的词为：", similarity_n)
print("隧道, 桥梁, 周边中差别最大的词为：", model.doesnt_match(li))
