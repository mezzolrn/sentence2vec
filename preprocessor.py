import json
import string
from utils import Dictionary
import jieba
import jieba.posseg #需要另外加载一个词性标注模块

import pandas as pd
# from nltk.tokenize import word_tokenize

P2P = 'total'
# ********************************************************************* #
datapath = '../../data/'+P2P+'/'+P2P+'.csv'
outpath = '../../data/'+P2P+'/data(%s).json'
dictpath = '../../data/'+P2P+'/mydict(%s).json'
debug_flag = False
stop = False
# ********************************************************************* #

mydict = Dictionary()
mydict.add_word('<pad>')
# mydict.add_word('<unk>')
stopping_word = open('../../data/stopping_word', 'r', encoding='utf-8').readlines()
for i in range(len(stopping_word)):
    stopping_word[i] = stopping_word[i].strip()
    
reviews = pd.read_csv(datapath, index_col=0, header=0, encoding='utf-8')
labels = list(reviews['reviewEvaluation'])
reviews = list(reviews['reviewContent'])
# reviews = open(datapath).readlines()
n_reviews = len(reviews)
print('%d条评论将被载入...' % n_reviews)

if debug_flag:
    size = '5'
else:
    size = 'all'

with open(outpath % size, 'a') as f:
    for i, line in enumerate(reviews):
        print(line)
        if debug_flag:
            if i == 5:
                break
#         json_data = json.loads(line)
#         words = word_tokenize(json_data['text'].lower())
        s = jieba.posseg.cut(line)
        words = [x.word for x in s]
        only_words = list()
        for word in words:
            # 去除标点和数字
#             if word in stopping_word or word.isdigit():
#                 continue
#             else:
            only_words.append(word)

        if stop is False:
            only_words = words
        data = {
            'label': labels[i],
            'text': only_words,
        }
        print(words)
        f.write(json.dumps(data,ensure_ascii=False) + '\n')

        for word in words:
            mydict.add_word(word)
        if i % 100 == 99:
            print('%.2f%% done, dictionary size: %d' % ((i + 1) * 100 / n_reviews, len(mydict)))

# 保存字典，下次可以直接载入
with open(dictpath % size, 'a') as f:
    f.write(json.dumps(mydict.idx2word,ensure_ascii=False) + '\n')
    f.close()
