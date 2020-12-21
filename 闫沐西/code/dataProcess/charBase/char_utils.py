import jieba
import json
import os

root_path = "../predata"
go_path = "../sanwenc"

with open(os.path.join(root_path, "test.txt"), 'r', encoding='utf-8') as data_file:
    file = open(os.path.join(go_path, "test.txt"), 'w', encoding='utf-8')
    for lines in data_file.readlines():
        if lines == "\n":
            continue
        head = lines.split()[0]
        tail = lines.split()[1]
        rel = lines.split()[2]
        sentence = lines.split()[3]
        tokens = []
        for words in sentence:
            if words == "<" or words == ">":
                continue
            tokens.append(words)
        data = {"tokens": tokens}
        i = 0
        p1 = sentence.find(head)
        p2 = sentence.find(tail)

        h = {"name": head, "pos": [p1, p1 + len(head)]}
        t = {"name": tail, "pos": [p2, p2 + len(tail)]}
        data['h'] = h
        data['t'] = t
        data["relation"] = rel
        d = json.dumps(data, ensure_ascii=False)
        if p1 >= 0 and p2 >= 0:
            file.write(d)
            file.write("\n")
    file.close()
