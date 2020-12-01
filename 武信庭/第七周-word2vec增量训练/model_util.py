from gensim.models import KeyedVectors, Word2Vec

file = "./models/added_w2v_model.bin"

# model = KeyedVectors.load_word2vec_format(file, encoding="utf-8", limit=500000)
# model.init_sims(replace=True)
# model.save('./models/pre_trained_word.bin')
model = KeyedVectors.load_word2vec_format(file, limit=500000)
similarity_1 = model.most_similar('隧道')
similarity_2 = model.most_similar('金属')
li = ["金属", "合金", "隧道"]
print("与隧道最相似的词为：", similarity_1)
print("与金属最相似的词为：", similarity_2)
print("金属, 合金, 隧道中差别最大的词为：", model.doesnt_match(li))
