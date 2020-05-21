# -*- coding: utf-8 -*-
import numpy as np
from preprocessed import cws_model_path, get_stop_words, remove_stop_words
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib
from sklearn import svm
from pyltp import Segmentor

TRAIN_DATA_PATH = 'question_classification/train_questions.txt'
TEST_DATA_PATH = 'question_classification/test_questions.txt'
QUESTION_CLASSIFICATION_PATH = 'question_classification'
FINE_RESULT_PATH = 'question_classification/test_questions_fine_class.txt'
ROUGH_RESULT_PATH = 'question_classification/test_questions_rough_class.txt'
FINE_MODEL_PATH = 'question_classification/fine_model'
ROUGH_MODEL_PATH = 'question_classification/rough_model'
TF_IDF_MODEL_PATH = 'question_classification/tf_idf_model'
stop_words = []


def train_predict_naive_bayes():
    train_text, train_y = preprocess_data(True)
    count_vector = CountVectorizer()
    vector_matrix = count_vector.fit_transform(train_text)

    train_tfidf = TfidfTransformer().fit_transform(vector_matrix)
    clf = MultinomialNB(alpha=0.1).fit(train_tfidf, train_y)

    test_text, test_y = preprocess_data(False)
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
    train_text, train_y = preprocess_data(True, fine)
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

    # save model
    if fine:
        joblib.dump(clf, FINE_MODEL_PATH)
    else:
        joblib.dump(clf, ROUGH_MODEL_PATH)
        joblib.dump(count_vector, TF_IDF_MODEL_PATH)

    test_text, test_y = preprocess_data(False, fine)
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


def preprocess_data(train_mode=True, fine=True, remove_stopwords=False):
    print("Initializing Segmentor!")
    segmentor = Segmentor()
    segmentor.load(cws_model_path)
    if remove_stopwords:
        get_stop_words()
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


def main():
    # get_stop_words()
    # rate = train_predict_naive_bayes()
    # print("acc: ", rate * 100, '%')
    train_predict_svm(True)
    train_predict_svm(False)
    # print("acc: ", rate * 100, '%')


if __name__ == '__main__':
    main()
