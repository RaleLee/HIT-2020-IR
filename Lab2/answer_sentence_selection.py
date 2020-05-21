from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from gensim.summarization import bm25
from scipy.linalg import norm
import json
from pyltp import Segmentor
from preprocessed import cws_model_path, TRAIN_DATA, SEARCH_RESULT
import numpy as np

RAW_PASSAGE_DATA = 'data/passages_multi_sentences.json'
SEG_PASSAGE_DATA = 'data/seg_passages.json'
RAW_FEATURE = 'data/feature.txt'
RAW_SENTENCE = 'data/all_sentence.txt'
TRAIN_BM25 = 'data/train_bm25.txt'
TEST_FEATURE = 'data/test_feature.txt'
TEST_SENTENCE = 'data/test_sentence.txt'
SVM_RANK_TRAIN_DATA = 'data/svm_train.txt'
SVM_RANK_DEV_DATA = 'data/svm_dev.txt'
SVM_RANK_TEST_DATA = 'data/svm_test.txt'
RANK_RESULT = 'data/svm_result.txt'
SVM_RANK_TRAIN_RESULT = 'data/dev_predictions'
SVM_RANK_TEST_RESULT = 'data/test_predictions'
TEST_RESULT = 'data/test_answer_result.json'


def generate_svm_rank_data(is_train=True):
    """
    为SVM Rank生成训练和测试数据
    :param is_train: 训练模式
    :return:
    """
    if is_train:
        sen_path, feature_path = RAW_SENTENCE, RAW_FEATURE
    else:
        sen_path, feature_path = TEST_SENTENCE, TEST_FEATURE
    sentences = []
    for line in open(sen_path, 'r', encoding='utf-8'):
        sentences.append(json.loads(line.strip()))
    # 读取特征文件
    features = []
    for line in open(feature_path, 'r', encoding='utf-8'):
        features.append(line.strip().split(' '))
    assert len(sentences) == len(features), 'Something wrong!'
    data, data_qid, qid_set = [], [], set()
    train_index = int(0.8 * float(5352))
    flag = False
    for k in range(len(sentences)):
        item = sentences[k]
        qid_set.add(item['qid'])
        # 写入训练集文件
        if len(qid_set) >= train_index + 1 and is_train and flag is False:
            sort_index = np.argsort(data_qid)
            with open(SVM_RANK_TRAIN_DATA, 'w', encoding='utf-8') as f:
                for j in range(len(sort_index)):
                    f.write(data[sort_index[j]])
            flag = True
            data.clear()
            data_qid.clear()
        feature_array = features[k]
        index = [0, 1, 2, 3, 4, 5, 6, 7]
        feature = ["{}:{}".format(j + 1, feature_array[index[j]]) for j in range(len(index))]
        data.append("{} qid:{} {}\n".format(item['label'], item['qid'], ' '.join(feature)))
        data_qid.append(item['qid'])
    # 写入开发集
    if is_train:
        sort_index = np.argsort(data_qid)
        with open(SVM_RANK_DEV_DATA, 'w', encoding='utf-8') as f:
            for j in range(len(sort_index)):
                f.write(data[sort_index[j]])
    else:
        with open(SVM_RANK_TEST_DATA, 'w', encoding='utf-8') as f:
            f.writelines(data)


# noinspection PyArgumentList
def build_feature(is_train=True):
    """

    :param is_train:
    :return:
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

    for k in range(len(items)):
        item = items[k]
        sents, corpus = [], []
        if is_train:
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
        question = list(segmentor.segment(item['question']))
        scores = bm25_model.get_scores(question)

        if is_train:
            for i in range(len(passage[item['pid']])):
                ans_sent = passage[item['pid']][i]
                feature_array = extract_feature(question, ans_sent, cv, tv)
                feature_array.append(scores[i])
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
                feature_array = extract_feature(question, sents[i], cv, tv)
                feature_array.append(scores[i])
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


def extract_feature(question, answer, cv, tv):
    """
    抽取句子的特征
    :param question: 问题
    :param answer: 答案
    :param cv: Count Vector
    :param tv: Tf-idf Vector
    :return: 特征列表
    """
    feature = []
    answer_words = answer.split(' ')
    len_answer, len_question = len(answer_words), len(question)
    # 特征1:答案句词数
    feature.append(len_answer)
    # 特征2:是否含冒号
    feature.append(1) if '：' in answer else feature.append(0)
    # 特征3:问句和答案句词数差异
    feature.append(abs(len_question - len_answer))
    # 特征4:uni-gram词共现比例：答案句和问句中出现的相同词占问句总词数的比例, 该特征反应了两个句子在文本上的一致程度。
    feature.append(len(set(question) & set(answer_words)) / float(len(set(question))))
    # 特征5:字符共现比例:答案句和问句中出现的相同字符占问句的比例
    feature.append(len(set(question) & set(''.join(answer_words))) / float(len(set(question))))
    # 特征6：one hot 余弦相似度
    vectors = cv.transform([' '.join(question), answer]).toarray()
    cosine_similar = np.dot(vectors[0], vectors[1]) / (norm(vectors[0]) * norm(vectors[1]))
    feature.append(cosine_similar if not np.isnan(cosine_similar) else 0.0)
    # 特征7：tf-idf 相似度
    vectors = tv.transform([' '.join(question), answer]).toarray()
    tf_sim = np.dot(vectors[0], vectors[1]) / (norm(vectors[0]) * norm(vectors[1]))
    feature.append(tf_sim if not np.isnan(tf_sim) else 0)
    return feature


def calc_mrr():
    """
    计算开发集上的完美匹配率 mrr
    :return:
    """
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
    print("question num:{}, question with answer{}, prefect_correct:{}, MRR:{}"
          .format(question_num, question_with_answer, prefect_correct, mrr / question_num))


def get_test_ans():
    """
    将SVM rank的结果转化为特征答案句
    :return:
    """
    # 读入排序结果
    with open(SVM_RANK_TEST_RESULT, 'r', encoding='utf-8') as f:
        predictions = np.array([float(line.strip()) for line in f.readlines()])
    print(len(predictions))
    # 读取sent json文件
    with open(TEST_SENTENCE, 'r', encoding='utf-8') as f:
        items = [json.loads(line.strip()) for line in f.readlines()]
    print(len(items))
    sent_qid = []
    for item in items:
        sent_qid.append(item['qid'])
    # 读取test res json文件
    with open(SEARCH_RESULT, 'r', encoding='utf-8') as f:
        test_res = [json.loads(line.strip()) for line in f.readlines()]
    for res in test_res:
        if res['qid'] not in sent_qid:
            res['answer_sentence'] = []
            continue
        s = sent_qid.index(res['qid'])
        e = s
        while e < len(sent_qid) and sent_qid[e] == res['qid']:
            e += 1
        p = np.argsort(-predictions[s:e])
        answer = []
        for i in p[0:3]:
            answer.append(items[i + s]['answer'])
        res['answer_sentence'] = answer
    # 写回文件
    with open(TEST_RESULT, 'w', encoding='utf-8') as f:
        for sample in test_res:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    """
    运行说明：
    前四行代码用于生成给SVM Rank来训练和预测的文件。
    后面两行是计算mmr和生成本阶段的答案。
    
    要运行时，首先运行前四行代码，将后面两行代码注释；
    然后在命令行中使用svm_rank_windows进行训练，
    训练使用的命令：
    ./svm_rank_learn -c 10.0  ../data/svm_train.txt model_10.dat
    使用训练好的模型预测dev集：
    ./svm_rank_classify ../data/svm_dev.txt model_10.dat ../data/dev_predictions
    使用训练好的模型预测test集
    ./svm_rank_classify ../data/svm_test.txt model_10.dat ../data/test_predictions
    
    得到test集上的结果之后，将前四行注释掉，运行后面两行代码。
    得到mmr结果和候选答案句的选取，结果保存在test_answer_result.json中
    
    [0, 1, 2, 3, 5]prefect_correct:470, MRR:0.591683226004102
    [0, 1, 2, 4, 5]prefect_correct:426, MRR:0.5643933064143644
    [0, 1, 2, 3, 4, 5]prefect_correct:466, MRR:0.5895169264670681
    [0, 1, 2, 3, 4, 5, 6, 7]prefect_correct:606, MRR:0.6986251509668727
    """
    # build_feature()
    # generate_svm_rank_data()
    # build_feature(False)
    # generate_svm_rank_data(False)

    calc_mrr()
    get_test_ans()
