import pandas as pd

# csv转txt词典
data = pd.read_csv('entities.csv', encoding='utf-8')
df = pd.DataFrame(data)
word = df["word"]
pos = df['pos']
length = df.shape[0]

with open('dict.txt', 'w+', encoding='utf-8') as f:
    for i in range(length):
        f.write(word[i] + ' 3 ' + pos[i] + '\n')
