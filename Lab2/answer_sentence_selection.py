from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from gensim.summarization import bm25
from scipy.linalg import norm
import json
from pyltp import Segmentor
from preprocessed import cws_model_path
import numpy as np

RAW_PASSAGE_DATA = 'data/passages_multi_sentences.json'
SEG_PASSAGE_DATA = 'data/seg_passages.json'
TRAIN_DATA = 'data/train.json'
RAW_FEATURE = 'data/feature.txt'
RAW_SENTENCE = 'data/all_sentence.txt'
TRAIN_BM25 = 'data/train_bm25.txt'
SEARCH_RESULT = 'data/test_search_res.json'
TEST_FEATURE = 'data/test_feature.txt'
TEST_SENTENCE = 'data/test_sentence.txt'
SVM_RANK_TRAIN_DATA = 'data/svm_train.txt'
SVM_RANK_DEV_DATA = 'data/svm_dev.txt'
SVM_RANK_TEST_DATA = 'data/svm_test.txt'
RANK_RESULT = 'data/svm_result.txt'
SVM_RANK_TRAIN_RESULT = 'data/dev_predictions'
SVM_RANK_TEST_RESULT = 'data/test_predictions'
TEST_RESULT = 'data/test_answer_result.json'


def gen_data_feature(is_train=True):
    """生成 SVM Rank 格式的训练和测试数据
    train:4281 dev:1071
    """
    # 读取sent json文件
    with open(RAW_SENTENCE, 'r', encoding='utf-8') as fin:
        items = [json.loads(line.strip()) for line in fin.readlines()]
    # 读取特征文件
    with open(RAW_FEATURE, 'r', encoding='utf-8') as f:
        feature_mat = [line.strip().split(' ') for line in f.readlines()]
    print(len(items))
    print(len(feature_mat))
    data = []
    data_qid = []
    qid_set = set()
    train_index = int(0.8 * float(5352))
    flag = False
    for k in range(len(items)):
        item = items[k]
        qid_set.add(item['qid'])
        # 写入训练集文件
        if len(qid_set) >= train_index + 1 and flag is False:
            sort_index = np.argsort(data_qid)
            with open(SVM_RANK_TRAIN_DATA, 'w', encoding='utf-8') as f:
                for j in range(len(sort_index)):
                    f.write(data[sort_index[j]])
            flag = True
            data.clear()
            data_qid.clear()
        feature_array = feature_mat[k]
        index = [0, 1, 2, 3, 4, 5, 6, 7]
        feature = ["{}:{}".format(j + 1, feature_array[index[j]]) for j in range(len(index))]
        data.append("{} qid:{} {}\n".format(item['label'], item['qid'], ' '.join(feature)))
        data_qid.append(item['qid'])
    # 写入开发集
    sort_index = np.argsort(data_qid)
    with open(SVM_RANK_DEV_DATA, 'w', encoding='utf-8') as f:
        for j in range(len(sort_index)):
            f.write(data[sort_index[j]])


def build_feature(is_train=True):
    """
    建立特征文件
    q:5352, s:112055, avg(s/q):20.93703288490284, vocab:168611
    """
    print("Initializing Segmentor!")
    segmentor = Segmentor()
    segmentor.load(cws_model_path)
    # 读取train json文件
    if is_train:
        with open(TRAIN_DATA, 'r', encoding='utf-8') as fin:
            items = [json.loads(line.strip()) for line in fin.readlines()]
    else:
        with open(SEARCH_RESULT, 'r', encoding='utf-8') as fin:
            items = [json.loads(line.strip()) for line in fin.readlines()]
        items.sort(key=lambda item_: item_['qid'])  # 按qid升序排序
    # 读入passage json文件
    passage = {}
    with open(SEG_PASSAGE_DATA, encoding='utf-8') as fin:
        for line in fin.readlines():
            read = json.loads(line.strip())
            passage[read['pid']] = read['document']
    # 读入raw passage json文件
    passage_raw = {}
    with open(RAW_PASSAGE_DATA, encoding='utf-8') as fin:
        for line in fin.readlines():
            read = json.loads(line.strip())
            passage_raw[read['pid']] = read['document']

    # 建立特征矩阵
    feature = []
    sents_json = []
    # sents = []
    # for item in items:
    #     sents += passage[item['pid']]

    for k in range(len(items)):
        item = items[k]
        sents, corpus = [], []
        if is_train:
            # 建立词袋模型
            cv = CountVectorizer(token_pattern=r"(?u)\b\w+\b")
            cv.fit(passage[item['pid']])
            tv = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
            tv.fit(passage[item['pid']])
            for sent in passage[item['pid']]:
                corpus.append(sent.split())

        else:
            for pid in item['answer_pid']:
                sents += passage[pid]
                for sent in passage[pid]:
                    corpus.append(sent.split())
            if len(sents) == 0:  # 没有检索到文档
                print("no answer pid: {}".format(item['qid']))
                continue
            cv = CountVectorizer(token_pattern=r"(?u)\b\w+\b")
            cv.fit(sents)
            tv = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
            tv.fit(sents)

        # 提取 BM25 特征
        bm25_model = bm25.BM25(corpus)
        # average_idf = sum(map(lambda k: float(bm25_model.idf[k]), bm25_model.idf.keys())) / len(bm25_model.idf.keys())
        q = list(segmentor.segment(item['question']))
        scores = bm25_model.get_scores(q)

        if is_train:
            for i in range(len(passage[item['pid']])):
                ans_sent = passage[item['pid']][i]
                feature_array = extract_feature(q, ans_sent, cv, tv, scores[i])
                feature.append(' '.join([str(attr) for attr in feature_array]) + '\n')
                sen = {}
                if passage_raw[item['pid']][i] in item['answer_sentence']:
                    sen['label'] = 1
                else:
                    sen['label'] = 0
                sen['qid'] = item['qid']
                sen['question'] = item['question']
                sen['answer'] = passage[item['pid']][i]
                sents_json.append(sen)
        else:
            for i in range(len(sents)):
                feature_array = extract_feature(q, sents[i], cv, tv, scores[i])
                feature.append(' '.join([str(attr) for attr in feature_array]) + '\n')
                sen = {'label': 0, 'qid': item['qid'], 'question': item['question'], 'answer': sents[i]}
                sents_json.append(sen)
    # 特征写入文件
    feature_path = RAW_FEATURE if is_train else TEST_FEATURE
    with open(feature_path, 'w', encoding='utf-8') as f:
        f.writelines(feature)
    # 句子写入文件
    sentence_path = RAW_SENTENCE if is_train else TEST_SENTENCE
    with open(sentence_path, 'w', encoding='utf-8') as fout:
        for sample in sents_json:
            fout.write(json.dumps(sample, ensure_ascii=False) + '\n')
    segmentor.release()


def extract_feature(question, answer, cv, tv, bm25_score):
    """ 抽取句子各种特征"""
    feature = []
    question_words = question
    answer_words = answer.split(' ')
    # 特征1：答案句词数
    feature.append(len(answer_words))
    # 特征2：是否含冒号
    feature.append(1) if '：' in answer else feature.append(0)
    # 特征3：问句和答案句词数差异
    feature.append(abs(len(question_words) - len(answer_words)))
    # 特征4：编辑距离
    # feature.append(distance.levenshtein(question, ''.join(answer_words)))
    # 特征4：unigram 词共现比例：答案句和问句中出现的相同词占问句总词数的比例
    feature.append(len(set(question_words) & set(answer_words)) / float(len(set(question_words))))
    # 特征5：字符共现比例:答案句和问句中出现的相同字符占问句的比例
    feature.append(len(set(question) & set(''.join(answer_words))) / float(len(set(question))))
    # 特征6：one hot 余弦相似度
    vectors = cv.transform([' '.join(question_words), answer]).toarray()
    cosine_similar = np.dot(vectors[0], vectors[1]) / (norm(vectors[0]) * norm(vectors[1]))
    feature.append(cosine_similar if not np.isnan(cosine_similar) else 0.0)
    # 特征7：tf-idf 相似度
    vectors = tv.transform([' '.join(question_words), answer]).toarray()
    tf_sim = np.dot(vectors[0], vectors[1]) / (norm(vectors[0]) * norm(vectors[1]))
    feature.append(tf_sim if not np.isnan(tf_sim) else 0)
    # 特征8：bm25 评分
    feature.append(bm25_score)
    return feature


def calc_mrr():
    """ 计算排序结果的 MRR """
    # 读入排序结果
    with open(SVM_RANK_TRAIN_RESULT, 'r', encoding='utf-8') as f:
        predictions = np.array([float(line.strip()) for line in f.readlines()])
    # print(len(predictions))
    # 读入开发集文件
    dev = []
    with open(SVM_RANK_DEV_DATA, 'r', encoding='utf-8') as f:
        i = 0
        for line in f.readlines():
            dev.append((i, int(line[0]), int(line.split(' ')[1].split(':')[1])))
            i += 1
    # print(len(dev))
    # 统计并计算 MRR
    old_pid = dev[0][2]
    q_s = 0
    question_num = 0
    question_with_answer = 0
    prefect_correct = 0
    mrr = 0.0
    for i in range(len(dev)):
        if dev[i][2] != old_pid:
            # print(i)
            p = np.argsort(-predictions[q_s:i]) + q_s
            for k in range(len(p)):
                if dev[p[k]][1] == 1:
                    question_with_answer += 1
                    if k == 0:
                        prefect_correct += 1
                    mrr += 1.0 / float(k + 1)
                    break
            # print(p)
            q_s = i
            old_pid = dev[i][2]
            question_num += 1
    p = np.argsort(-predictions[q_s:]) + q_s
    for k in range(len(p)):
        if dev[p[k]][1] == 1:
            question_with_answer += 1
            if k == 0:
                prefect_correct += 1
            mrr += 1.0 / float(k + 1)
            break
    question_num += 1
    print(
        "question num:{}, question with answer{}, prefect_correct:{}, MRR:{}".format(question_num, question_with_answer,
                                                                                     prefect_correct,
                                                                                     mrr / question_num))


def gen_test_data():
    """生成测试数据
    test 1978
    """
    # 读取sent json文件
    with open(TEST_SENTENCE, 'r', encoding='utf-8') as fin:
        items = [json.loads(line.strip()) for line in fin.readlines()]
    # 读取特征文件
    with open(TEST_FEATURE, 'r', encoding='utf-8') as f:
        feature_mat = [line.strip().split(' ') for line in f.readlines()]
    data = []
    data_qid = []
    qid_set = set()
    for k in range(len(items)):
        item = items[k]
        qid_set.add(item['qid'])
        feature_array = feature_mat[k]
        index = [0, 1, 2, 3, 4, 5, 6, 7]
        feature = ["{}:{}".format(j + 1, feature_array[index[j]]) for j in range(len(index))]
        data.append("{} qid:{} {}\n".format(item['label'], item['qid'], ' '.join(feature)))
        data_qid.append(item['qid'])
    # 写入开发集
    with open(SVM_RANK_TEST_DATA, 'w', encoding='utf-8') as f:
        f.writelines(data)


def get_test_ans():
    """ 根据SVM Rank 对test.json 的预测文件得到候选答案句"""
    # 读入排序结果
    with open(SVM_RANK_TEST_RESULT, 'r', encoding='utf-8') as f:
        predictions = np.array([float(line.strip()) for line in f.readlines()])
    print(len(predictions))
    # 读取sent json文件
    with open(TEST_SENTENCE, 'r', encoding='utf-8') as fin:
        items = [json.loads(line.strip()) for line in fin.readlines()]
    print(len(items))
    sent_qid = []
    for item in items:
        sent_qid.append(item['qid'])
    # 读取test res json文件
    with open(SEARCH_RESULT, 'r', encoding='utf-8') as fin:
        test_res = [json.loads(line.strip()) for line in fin.readlines()]
    for res in test_res:
        if res['qid'] not in sent_qid:
            res['answer_sentence'] = []
            continue
        s = sent_qid.index(res['qid'])
        e = s
        while e < len(sent_qid) and sent_qid[e] == res['qid']:
            e += 1
        # print(s)
        # print(e)
        # print(predictions[s:e])
        p = np.argsort(-predictions[s:e])
        # print(p)
        # print(len(p))
        answer = []
        for i in p[0:3]:
            answer.append(items[i + s]['answer'])
        res['answer_sentence'] = answer
    # 写回文件
    with open(TEST_RESULT, 'w', encoding='utf-8') as fout:
        for sample in test_res:
            fout.write(json.dumps(sample, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    """
    TF: question num:1071, question with answer1064, prefect_correct:491, MRR:0.6200992728238546
    Bow: question num:1071, question with answer1064, prefect_correct:376, MRR:0.5296269550798569
    BM25 + SVMRank: question num:1071, question with answer1064, prefect_correct:550, MRR:0.6524602350657909
    BM25:  question num:5352, question with answer5319, prefect_correct:2783, MRR:0.6594948607696011
    -- [8, 10]: BM25 + TF: question num:1071, question with answer1064, prefect_correct:566, MRR:0.6626076288570559
    1  [0, 8, 10]: question num:1071, question with answer1064, prefect_correct:571, MRR:0.6672033276872427
    1  [1, 8, 10]: question num:1071, question with answer1064, prefect_correct:580, MRR:0.6760457345092623
    1  [2, 8, 10]: question num:1071, question with answer1064, prefect_correct:573, MRR:0.6696462509248209
    -  [6, 8, 10]: question num:1071, question with answer1064, prefect_correct:566, MRR:0.6623672110525073
    0  [9, 8, 10]: question num:1071, question with answer1064, prefect_correct:564, MRR:0.6606765867066795
    0  [7, 8, 10]: question num:1071, question with answer1064, prefect_correct:563, MRR:0.6607581853759598
    1  [5, 8, 10]: question num:1071, question with answer1064, prefect_correct:567, MRR:0.6702383345109053
    1  [4, 8, 10]: question num:1071, question with answer1064, prefect_correct:566, MRR:0.6630578089290848
    1  [3, 8, 10]: question num:1071, question with answer1064, prefect_correct:575, MRR:0.6726265375296776
    1  [3, 4, 5, 8, 10]: question num:1071, question with answer1064, prefect_correct:575, MRR:0.6774593108601733
    1  [3, 4, 5,6,8,10]: question num:1071, question with answer1064, prefect_correct:582, MRR:0.6815223646174593
       [3, 4, 5,6,7,8,10]: question num:1071, question with answer1064, prefect_correct:580, MRR:0.6808120259425803
    =  [0, 1, 2, 3, 4, 5, 6,8,10]: question num:1071, question with answer1064, prefect_correct:615, MRR:0.7118507347966923  1000
    ALL: question num:1071, question with answer1064, prefect_correct:608, MRR:0.7071495565046418  1000
    == ：[0, 1, 2, 3, 4, 5, 6,8,10]: question num:1071, question with answer1064, prefect_correct:618, MRR:0.7163615823193664 5000
    ------------not remove stopwords---------
    -- [8, 10]: BM25 + TF: question num:1071, question with answer1064, prefect_correct:500, MRR:0.622616494858178
    [0, 1, 2, 3, 4, 5, 6,8,10]: question num:1071, question with answer1064, prefect_correct:604, MRR:0.7060185519102772  1000
    ALL: question num:1071, question with answer1064, prefect_correct:605, MRR:0.7077294225814748
    """
    # gen_data()
    # test()

    # build_feature()
    # gen_data_feature()
    # build_feature(False)
    # gen_test_data()
    calc_mrr()
    get_test_ans()
