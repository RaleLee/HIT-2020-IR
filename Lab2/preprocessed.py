# encoding = utf-8
# 每一行是一个json格式的字符串 文件很大  读一行处理一行 结果索引最后也保存起来

import json
import os
from pyltp import Segmentor

LTP_DATA_DIR = 'D:/Course/IR/ltp_data_v3.4.0'
STOP_WORDS_PATH = "data/stopwords.txt"
DATA_PATH = "data/passages_multi_sentences.json"
INDEX_PATH = "data/index.txt"
cws_model_path = os.path.join(LTP_DATA_DIR, 'cws.model')
stop_words = []
word_dict = {}


def get_stop_words():
    """
    从指定的文件中获取stopwords
    :return: 文件不存在则报错，存在则返回stopwords列表
    """
    path = STOP_WORDS_PATH
    if not os.path.exists(path):
        print("No stop words file!")
        return
    for line in open(path, "r", encoding="utf-8"):
        stop_words.append(line.strip())
    print(len(stop_words))
    return


def remove_stop_words(text_words: list):
    """
    对分词结果进行去停用词处理
    :param text_words: 分词列表
    :return: 去掉停用词后的分词结果
    """
    ret = []
    for text_word in text_words:
        if text_word not in stop_words:
            ret.append(text_word)
    return ret


def preprocess():
    print("Finding existing index!")
    if os.path.exists(INDEX_PATH):
        print("Already existing! Loading...")
        for line in open(INDEX_PATH, 'r', encoding='utf-8'):
            li = line.split(": ")
            assert len(li) == 2, "Something wrong!"
            word = li[0]
            pid_list = li[1].strip()[:-1].split(",")
            word_dict[word] = set(pid_list)
        print("Finish loading.")
        return
    print("Initializing Segmentor!")
    segmentor = Segmentor()
    segmentor.load(cws_model_path)
    print(len(stop_words))
    for line in open(DATA_PATH, 'r', encoding='utf-8'):
        passage = json.loads(line)
        pid = passage['pid']
        if pid % 1000 == 0:
            print(pid)
        doc = passage['document']
        for sen in doc:
            words = list(segmentor.segment(sen))
            pred_words = remove_stop_words(words)
            # make inverted index
            for word in pred_words:
                if word not in word_dict:
                    word_dict[word] = set()
                    word_dict[word].add(pid)
                else:
                    word_dict[word].add(pid)
    print("Finish preprocess!")

    # write word_dict
    with open(INDEX_PATH, 'w', encoding='utf-8') as f:
        for k, v in word_dict.items():
            f.write(str(k) + ": ")
            for pi in v:
                f.write(str(pi) + ",")
            f.write("\n")
    print("Finish write Index!")
    return


def main():
    get_stop_words()
    preprocess()
    search()


def search():
    """
    A naive search system.
    Use BM25 search for a better accuracy.
    :return:
    """
    print("检索系统说明：")
    print("可以使用&&和||来连接不同的词，&&表示与，||表示或，只支持全&&或全||，不支持混合式")
    print("如果不使用连接，将会默认进行分词和采用全AND模式进行查询")
    print("输入exit退出")
    print("Initializing Segmentor!")
    segmentor = Segmentor()
    segmentor.load(cws_model_path)
    while True:
        print("输入要搜索的词")
        question = input()
        if question == 'exit':
            print("Exit!")
            break
        word_list = []
        and_mode = True
        if '&&' in question:
            word_list = question.split('&&')
        elif '||' in question:
            word_list = question.split('||')
            and_mode = False
        else:
            word_list = remove_stop_words(list(segmentor.segment(question)))
        count = []
        for word in word_list:
            if word in word_dict:
                count.append(word_dict[word])
        if and_mode:
            length = len(count)
            if length == 1:
                result = count[0]
            else:
                result = count[0]
                for i in range(1, length):
                    result = result.intersection(count[i])
        else:
            length = len(count)
            if length == 1:
                result = count[0]
            else:
                result = count[0]
                for i in range(1, length):
                    result = result.union(count[i])
        result = list(result)
        print("Find pid: ")
        for res in result[:-1]:
            print(res + ", ", end='')
        print(result[-1])
    return


def BM25_search():
    pass


if __name__ == "__main__":
    main()
