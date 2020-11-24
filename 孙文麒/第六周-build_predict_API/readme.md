# 孙文麒-第六周-build_predict_API

## 1 代码仓库

### 1.1 [albert_lstm_crf_ner](https://github.com/Nntraveler/albert_lstm_crf_ner)

**代码修改**

- 对predict函数进行了修改，现在predict函数会返回 格式为`{start: end: name: type:}`的list
- 将源仓库中main.py改名为function.py 并加入了新的main.py用于FASTAPI

**成果**

- 可以顺利通过12001端口进行predict调用

## 2 一些坑

- 需要对docker命令行进行配置，否则可能无法正常输入中文。

## 3 TODO

- 进一步修改function.py, main.py中的路径配置，考虑加入os.chdir等函数，使得后续可以将FASTAPI的调用移动到父级目录，提高可扩展性。