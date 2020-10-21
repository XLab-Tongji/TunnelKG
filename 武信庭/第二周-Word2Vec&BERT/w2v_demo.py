from gensim.models import Word2Vec

sentences = [["cat", "say", "meow"], ["dog", "say", "woof"], ["wolf", "say", "woof"]]
model = Word2Vec(sentences, min_count=1, workers=2)

cat = model.wv['cat']
print(cat)

similarity = model.wv.most_similar('cat')
# similarity2 = model.wv.similar_by_word('dog')
print(similarity)

match = model.wv.doesnt_match("dog wolf cat".split())
print(match)
# for i in model.wv.vocab.keys():
#     print(type(i))
#     print(i)

