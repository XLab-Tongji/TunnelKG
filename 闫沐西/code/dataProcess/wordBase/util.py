import torch
import torch.nn.functional as F
import torch.utils.data.dataset
import json
import jieba
import os

root_path = "./predata"
go_path = "./sanwen"

all_words = {}
rel2id = {}
with open(os.path.join(root_path, "relation2id.txt"), "r", encoding="utf-8") as rel_file:
    for line in rel_file.readlines():
        rel2id[line.split()[0]] = int(line.split()[1])
    rel = json.dumps(rel2id, ensure_ascii=False)
    f = open(os.path.join(go_path, "rel2id.json"), 'w', encoding="utf-8")
    f.write(rel)
    f.close()
    rel_file.close()
