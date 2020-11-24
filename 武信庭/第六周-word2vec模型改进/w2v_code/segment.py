import os
import jieba

path = "./txt"
path_out = "./txt_out"

# 导入词典
jieba.load_userdict('dict.txt')

# 分词处理
files = os.listdir(path)
for file in files:
    position = path + '\\' + file
    position_out = path_out + '/' + file
    print(position)
    with open(position, "rb") as f:
        document = f.read()
        document_cut = jieba.cut(document)
        result = ' '.join(document_cut)
        result = result.encode('utf-8')
        with open(position_out, 'wb') as f2:
            f2.write(result)
    f.close()
    f2.close()