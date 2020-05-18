# -*- coding: utf-8 -*-

import os
import numpy as np
from preprocessed import STOP_WORDS_PATH, cws_model_path
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm, metrics
from pyltp import Segmentor

TRAIN_DATA_PATH = 'question_classification/train_questions.txt'
TEST_DATA_PATH = 'question_classification/test_questions.txt'
QUESTION_CLASSIFICATION_PATH = 'question_classification'
FINE_RESULT_PATH = 'question_classification/test_questions_fine_class.txt'
ROUGH_RESULT_PATH = 'question_classification/test_questions_rough_class.txt'
stop_words = []


def train_predict_naive_bayes():
    train_text, train_y = preprecess_data(True)
    count_vector = CountVectorizer()
    vector_matrix = count_vector.fit_transform(train_text)

    train_tfidf = TfidfTransformer().fit_transform(vector_matrix)
    clf = MultinomialNB(alpha=0.1).fit(train_tfidf, train_y)

    test_text, test_y = preprecess_data(False)
    predict_result = []
    for text in test_text:
        test_vector_martix = count_vector.transform([text])
        test_tfidf = TfidfTransformer().fit_transform(test_vector_martix)
        predict_result.append(clf.predict(test_tfidf))

    # calculate acc
    n, right = 0, 0

    for pred, real, text in zip(predict_result, test_y, test_text):
        n += 1
        if pred[0] == real:
            right += 1
        else:
            print(str(pred[0]) + " " + str(real) + " " + str(text))

    return right * 1.0 / n


def train_predict_svm(fine=True):
    train_text, train_y = preprecess_data(True, fine)
    # count_vector = CountVectorizer()
    # 发现单字的问题是token_pattern这个参数问题。它的默认值只匹配长度≥2的单词
    # token_pattern这个参数使用正则表达式来分词，
    # 其默认参数为r"(?u)\b\w\w+\b"，其中的两个\w决定了其匹配长度至少为2的单词，
    # 所以这边减到1个。
    count_vector = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
    train_data = count_vector.fit_transform(train_text)

    if fine:
        clf = svm.SVC(C=100.0, gamma=0.05)
    else:
        clf = svm.SVC(C=100.0, gamma=0.05)
    clf.fit(train_data, np.asarray(train_y))

    test_text, test_y = preprecess_data(False, fine)
    test_data = count_vector.transform(test_text)
    result = clf.predict(test_data)
    score = clf.score(test_data, test_y)
    print(score)
    path = ROUGH_RESULT_PATH
    if fine:
        path = FINE_RESULT_PATH

    with open(path, 'w', encoding='utf-8') as f:
        for pred, real, text in zip(result, test_y, test_text):
            f.write("{}\t{}\t{}\n".format(pred, real, text))
    return


def evaluate(actual, pred):
    m_precision = metrics.precision_score(actual, pred, average='micro')
    m_recall = metrics.recall_score(actual, pred, average='micro')
    m_f1 = metrics.f1_score(actual, pred, average='micro')
    print('precision:{0:.3f}'.format(m_precision))
    print('recall:{0:0.3f}'.format(m_recall))
    print('f1:{0:0.3f}'.format(m_f1))


def preprecess_data(train_mode=True, fine=True, remove_stopwords=False):
    print("Initializing Segmentor!")
    segmentor = Segmentor()
    segmentor.load(cws_model_path)
    print(len(stop_words))
    text, y = [], []
    if train_mode:
        path = TRAIN_DATA_PATH
    else:
        path = TEST_DATA_PATH
    for line in open(path, 'r', encoding='utf-8'):
        tmp = line.split('\t')
        assert len(tmp) == 2, "Something wrong with the data!"
        if fine:
            tag, question = tmp[0], tmp[1]
        else:
            tag, question = tmp[0].split('_')[0], tmp[1]
        if remove_stopwords:
            pred_words = remove_stop_words(list(segmentor.segment(question)))
        else:
            pred_words = list(segmentor.segment(question))
        seg_text = ''
        for word in pred_words:
            seg_text += word + ' '
        text.append(seg_text)
        y.append(tag)
    segmentor.release()
    return text, y


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


def main():
    # get_stop_words()
    # rate = train_predict_naive_bayes()
    # print("acc: ", rate * 100, '%')
    train_predict_svm(True)
    train_predict_svm(False)
    # print("acc: ", rate * 100, '%')


if __name__ == '__main__':
    main()
