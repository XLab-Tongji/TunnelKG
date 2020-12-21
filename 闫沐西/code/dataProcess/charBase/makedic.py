import numpy as np
import json
import os

words = {}
word_list = []

root_path = "../sanwenc"


def makeDic(infile, dictionary):
    num = 0
    ndArray = []
    with open(infile, "r", encoding="utf-8") as infile:
        for lines in infile.readlines():
            word = lines.split()[0]

            dictionary[word] = num
            num += 1

            w2v = []
            # print(lines.split()[1])
            for i in range(1, 101):
                value = float(lines.split()[i])
                w2v.append(value)
            ndArray.append(w2v)

    return ndArray


word_vec = np.array(makeDic(os.path.join(root_path, "vec.txt"), words))

np.save(os.path.join(root_path, "myvec.npy"), word_vec)

word2id = json.dumps(words, ensure_ascii=False)
file = open(os.path.join(root_path, "word2id.json"), 'w', encoding="utf-8")
file.write(word2id)
file.close()
