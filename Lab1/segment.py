import os
import json
import re
from pyltp import Segmentor

LTP_DATA_DIR = 'D:/Course/IR/ltp_data_v3.4.0'
STOP_WORDS_PATH = "./stopwords.txt"
DATA_PATH = "results/full_data.json"
cws_model_path = os.path.join(LTP_DATA_DIR, 'cws.model')


def get_stop_words():
    """
    从指定的文件中获取stopwords
    :return: 文件不存在则报错，存在则返回stopwords列表
    """
    stopwords = []
    path = STOP_WORDS_PATH
    if not os.path.exists(path):
        print("No stop words file!")
        return
    for line in open(path, "r", encoding="utf-8"):
        stopwords.append(line.strip())
    return stopwords


def get_need_seg_file():
    """
    从指定的文件中获取前10个需要进行分词的行
    :return: 文件不存在则报错，存在则返回待分词的10行
    """
    need_seg_file = []
    path = DATA_PATH
    if not os.path.exists(path):
        print("No data!")
        return
    with open(path, "r", encoding="utf-8") as f:
        for _ in range(10):
            need_seg_file.append(f.readline().strip())
    return need_seg_file


def seg(stopwords, needsegs, segor: Segmentor):
    """
    分词执行程序，将会进行分词和去停用词
    :param stopwords: 停用词表
    :param needsegs: 需要分词的json列表
    :param segor: 分词程序
    :return: 列表，每个元素是一个json格式的数据
    """
    ret = []
    for data in needsegs:
        dic = json.loads(data)
        title = dic.pop("title")
        title_words = list(segor.segment(title))
        dic["segmented_title"] = remove_stop_words(stopwords, title_words)
        para = dic.pop("paragraphs")
        pattern = re.compile(r"([\n\t])")
        para = re.sub(pattern, "", para)
        parawords = list(segor.segment(para))
        dic["segmented_paragraphs"] = remove_stop_words(stopwords, parawords)
        # 保持要求的文件格式
        tmp_file_name = dic.pop("file_name")
        dic["file_name"] = tmp_file_name
        ret.append(dic)
    return ret


def remove_stop_words(stopwords: list,
                      text_words: list):
    """
    对分词结果进行去停用词处理
    :param stopwords: 停用词列表
    :param text_words: 分词列表
    :return: 去掉停用词后的分词结果
    """
    ret = []
    for text_word in text_words:
        if text_word not in stopwords:
            ret.append(text_word)
    return ret


def write_result(data: list):
    """
    将结果写入文件
    :param data: 结果
    :return: None
    """
    with open("results/preprocessed.json", "w", encoding="utf-8") as f:
        for line in data:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    print("Loading stop words and data!")
    stop_words = get_stop_words()
    need_segs = get_need_seg_file()
    print("Initializing Segmentor!")
    segmentor = Segmentor()
    segmentor.load(cws_model_path)
    print("Segmenting!")
    results = seg(stop_words, need_segs, segmentor)
    segmentor.release()
    write_result(results)
    print("Finish!")
