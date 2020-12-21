import opennre
import torch
import os
import sys
import json
import numpy as np
import logging
import jieba

root_path = "../sanwenc"


def getModel(path):
    ckpt = './ckpt/sanwenchar_cnn_soft.pth.tar'
    rel2id = json.load(open(os.path.join(path, 'rel2id.json')))

    word2id = json.load(open(os.path.join(path, 'word2id.json')))
    word2vec = np.load(os.path.join(path, 'myvec.npy'))

    # Define the sentence encoder
    sentence_encoder = opennre.encoder.CNNEncoder(
        token2id=word2id,
        max_length=120,
        word_size=100,
        position_size=10,
        hidden_size=230,
        blank_padding=True,
        kernel_size=3,
        padding_size=1,
        word2vec=word2vec,
        dropout=0.5
    )

    # Define the model
    m = opennre.model.SoftmaxNN(sentence_encoder, len(rel2id), rel2id)
    m.load_state_dict(torch.load(ckpt, map_location='cpu')['state_dict'])

    return m


def tokenize(sentence, entity1, entity2):
    tokens = []
    for words in sentence:
        if words == "<" or words == ">":
            continue
        tokens.append(words)
    data = {"tokens": tokens}
    i = 0
    p1 = sentence.find(entity2)
    p2 = sentence.find(entity1)

    h = {"name": entity1, "pos": [p1, p1 + len(entity1)]}
    t = {"name": entity2, "pos": [p2, p2 + len(entity2)]}
    data['h'] = h
    data['t'] = t
    h = {"name": entity1, "pos": [p1, p1 + 1]}
    t = {"name": entity2, "pos": [p2, p2 + 1]}
    data['h'] = h
    data['t'] = t
    return data


def jsonProcess(js, model):
    jsText = json.loads(js)
    tokens = []
    for words in jsText['sentence']:
        if words == "<" or words == ">":
            continue
        tokens.append(words)

    i = 0
    h = {}
    tails = []
    relations = []
    for entity in jsText['entities']:
        i += 1
        if i == jsText["chosen_entity"]:
            h["pos"] = entity
        else:
            tails.append({"pos": entity})

    data = {"tokens": tokens, "h": h}
    for entity in tails:
        data["t"] = entity
        relation = model.infer(data)
        relations.append(relation)
    return relations


dic = {
    "sentence": "通过盾构外壳和管片支承四周围岩防止发生往隧道内的坍塌",
    "entities": [
        [2, 5],
        [7, 8],
        [13, 14],
        [20, 21]
    ],
    "chosen_entity": 1
}
text = json.dumps(dic)

model = getModel(root_path)
print(jsonProcess(text, model))
# print(tokenize("北京在上海北边", "北京", "上海"))
# print(model.infer(tokenize("北京在上海北边", "北京", "上海")))
