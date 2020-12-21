import jieba
import json
import os

root_path = "./predata"
go_path = "./sanwen"

with open(os.path.join(root_path, "valid.txt"), 'r', encoding='utf-8') as data_file:
    for lines in data_file.readlines():
        if lines == "\n":
            continue
        head = lines.split()[0]
        tail = lines.split()[1]
        jieba.add_word(head)
        jieba.add_word(tail)

with open(os.path.join(root_path, "valid.txt"), 'r', encoding='utf-8') as data_file:
    file = open(os.path.join(go_path, "valid.txt"), 'w', encoding='utf-8')
    for lines in data_file.readlines():
        if lines == "\n":
            continue
        head = lines.split()[0]
        tail = lines.split()[1]
        rel = lines.split()[2]
        sentence = lines.split()[3]
        sentence = jieba.cut(sentence)
        tokens = []
        for words in sentence:
            if words == "<" or words == ">":
                continue
            tokens.append(words)
        data = {"tokens": tokens}
        i = 0;
        p1 = p2 = -1
        for words in tokens:
            if words == head:
                p1 = i
            elif words == tail:
                p2 = i
            i += 1
        h = {"name": head, "pos": [p1, p1 + 1]}
        t = {"name": tail, "pos": [p2, p2 + 1]}
        data['h'] = h
        data['t'] = t
        data["relation"] = rel
        d = json.dumps(data, ensure_ascii=False)
        if p1 >= 0 and p2 >= 0:
            file.write(d)
            file.write("\n")
    file.close()
