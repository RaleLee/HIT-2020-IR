import os
import json
import re
import random
import docx
from pyltp import Segmentor
from utils import STOP_WORDS, DATA_DIR, cws_model_path


def get_stop_words():
    """
    从指定的文件中获取stopwords
    :return: 文件不存在则报错，存在则返回stopwords列表
    """
    stopwords = []
    path = STOP_WORDS
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
    path = DATA_DIR
    if not os.path.exists(path):
        print("No data!")
        return
    with open(path, "r", encoding="utf-8") as f:
        return f.readlines()


def seg(stopwords, needsegs, segor: Segmentor):
    """
    分词执行程序，将会进行分词和去停用词
    :param stopwords: 停用词表
    :param needsegs: 需要分词的json列表
    :param segor: 分词程序
    :return: 列表，每个元素是一个json格式的数据
    """
    ret = []
    index = 0
    for data in needsegs:
        index += 1
        print(index)
        dic = json.loads(data.strip())
        title = dic["title"]
        title_words = list(segor.segment(title))
        dic["segmented_title"] = remove_stop_words(stopwords, title_words)
        para = dic["paragraphs"]
        pattern = re.compile(r"([\n\t])")
        para = re.sub(pattern, "", para)
        parawords = list(segor.segment(para))
        dic["segmented_paragraphs"] = remove_stop_words(stopwords, parawords)
        # 保持要求的文件格式
        tmp_file_name = dic["file_name"]
        segmented_file_name = []
        seg_file_contents = []
        for file_name in tmp_file_name:
            tmp1 = file_name.split('files/doc/')[1]
            if '.doc' not in tmp1:
                continue
            seg_file_name = list(segor.segment(tmp1.split('.')[0]))
            segmented_file_name.append(remove_stop_words(stopwords, seg_file_name))

            file_name = tmp1 if '.docx' in tmp1 else tmp1 + 'x'
            try:
                file = docx.Document('D:/Course/IR/HIT-2020-IR/Lab3/files/doc/' + file_name)
                t = ' '.join([p.text for p in file.paragraphs])
                seg_t = list(segor.segment(t))
                seg_file_contents.append(remove_stop_words(stopwords, seg_t))
            except docx.opc.exceptions.PackageNotFoundError:
                continue

        dic["segmented_file_name"] = segmented_file_name
        dic["segmented_file_contents"] = seg_file_contents
        dic["authority"] = random.randint(1, 4)
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
    with open("data/preprocessed.json", "w", encoding="utf-8") as f:
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
