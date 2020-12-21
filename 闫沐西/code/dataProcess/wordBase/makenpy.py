import numpy as np
import json
import os

words = {}
word_list = []

root_path = "./sanwen"


def makeDic(infile, dictionary):
    num = len(dictionary)
    with open(infile, "r", encoding="utf-8") as infile:
        for lines in infile.readlines():
            line = json.loads(lines)
            tokens = line["tokens"]
            for word in tokens:
                if word in dictionary:
                    continue
                else:
                    dictionary[word] = num
                    num += 1


def createVec(dictionary, aList):
    for word in dictionary:
        w2v = np.random.rand(50)
        aList.append(w2v)

    ndArray = np.array(aList)
    return ndArray


makeDic(os.path.join(root_path, "test.txt"), words)
makeDic(os.path.join(root_path, "train.txt"), words)
makeDic(os.path.join(root_path, "valid.txt"), words)

word_vec = createVec(words, word_list)
np.save(os.path.join(root_path, "myvec.npy"), word_vec)

word2id = json.dumps(words, ensure_ascii=False)
file = open(os.path.join(root_path, "word2id.json"), 'w', encoding="utf-8")
file.write(word2id)
file.close()
