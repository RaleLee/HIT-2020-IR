from sklearn.externals import joblib
from metric import *
from pyltp import Segmentor, Parser, Postagger
import json
import os
from preprocessed import LTP_DATA_DIR, TRAIN_DATA, cws_model_path
from answer_sentence_selection import TEST_RESULT
from question_classification import ROUGH_MODEL_PATH, TF_IDF_MODEL_PATH

TRAIN_ANSWER = 'data/train_answer.json'
TRAIN_DIFF = 'data/train_diff.json'
TEST_ANSWER = 'data/test_answer.json'
pos_model_path = os.path.join(LTP_DATA_DIR, 'pos.model')
par_model_path = os.path.join(LTP_DATA_DIR, 'parser.model')


def train_test():
    """对训练集进行答案提取 测试性能"""
    postagger = Postagger()  # 初始化实例
    postagger.load(pos_model_path)  # 加载模型

    segmentor = Segmentor()  # 初始化实例
    segmentor.load(cws_model_path)  # 加载模型

    parser = Parser()  # 初始化实例
    parser.load(par_model_path)  # 加载模型

    clf = joblib.load(ROUGH_MODEL_PATH)
    tv = joblib.load(TF_IDF_MODEL_PATH)
    # 读取json文件
    with open(TRAIN_DATA, 'r', encoding='utf-8') as fin:
        items = [json.loads(line.strip()) for line in fin.readlines()]
    res = []
    train_diff_res = []
    none_cnt = 0
    for item in items:
        ans = item['answer']
        # sent = ' '.join(jieba.cut())
        sent = ' '.join(list(segmentor.segment(item['question'])))
        test_data = tv.transform([sent])
        label = clf.predict(test_data)[0]
        q_words = [word for word in segmentor.segment(item['question'])]
        ans_sent = ' '.join(item['answer_sentence'])
        ans_words = [word for word in segmentor.segment(ans_sent)]
        words_pos = postagger.postag(ans_words)
        if '：' in ans_sent:
            # item['answer'] = ''.join(ans_sent.split('：')[1:])
            item['answer'] = ans_sent.split('：')[1]
        elif ':' in ans_sent:
            item['answer'] = ans_sent.split(':')[1]
        elif label == 'HUM':  # 人物
            item['answer'] = extract_by_pos(q_words, ans_words, words_pos, ['nh', 'ni'])
        elif label == 'LOC':
            item['answer'] = extract_by_pos(q_words, ans_words, words_pos, ['nl', 'ns'])
        elif label == 'NUM':
            item['answer'] = extract_by_pos(q_words, ans_words, words_pos, ['m'])
        elif label == 'TIME':
            item['answer'] = extract_by_pos(q_words, ans_words, words_pos, ['nt'])
        # elif label == 'OBJ':
        #     item['answer'] = extract_by_pos(q_words, ans_words, words_pos, ['n'])
        # elif label == 'DES':
        #     item['answer'] = extract_by_pos(q_words, ans_words, words_pos, ['j'])
        else:
            item['answer'] = ''.join(ans_words)
            # item['answer'] = ''
        res.append(item)
        if item['answer'] == '':
            none_cnt += 1
        tmp = {}
        tmp['l'] = label
        tmp['q'] = item['question']
        tmp['pre'] = item['answer']
        tmp['true'] = ans
        tmp['s'] = ' '.join(ans_words)
        tmp['pos'] = [pos for pos in words_pos]
        # tmp['arcs'] = " ".join("%d:%s" % (arc.head, arc.relation) for arc in arcs)
        train_diff_res.append((label, tmp))
        # print(tmp)
    print("none_cnt: {}".format(none_cnt))
    segmentor.release()
    postagger.release()  # 释放模型
    parser.release()  # 释放模型
    # 写回json文件
    with open(TRAIN_ANSWER, 'w', encoding='utf-8') as fout:
        for sample in res:
            fout.write(json.dumps(sample, ensure_ascii=False) + '\n')
    # 写中间结果文件
    train_diff_res.sort(key=lambda item: item[0])
    with open(TRAIN_DIFF, 'w', encoding='utf-8') as f:
        for i in range(len(train_diff_res)):
            f.write(json.dumps(train_diff_res[i][1], ensure_ascii=False) + '\n')


def extract_by_pos(q_words, words, words_pos, pos):
    """抽取某个词性的词"""
    res = []
    for i in range(len(words_pos)):
        if words_pos[i] in pos:
            res.append(words[i])
    if len(res):
        return ''.join(res)
    else:
        return ''.join(words)


def evaluate():
    """评估"""
    # 读取json文件
    with open(TRAIN_DATA, 'r', encoding='utf-8') as fin:
        train_data = [json.loads(line.strip()) for line in fin.readlines()]
    # 读取json文件
    with open(TRAIN_ANSWER, 'r', encoding='utf-8') as fin:
        train_ans = [json.loads(line.strip()) for line in fin.readlines()]
    cnt = len(train_data)
    # cnt = 10
    all_prediction = []
    all_ground_truth = []
    bleu = 0.0
    p = 0.0
    r = 0.0
    f1 = 0.0
    # for i in range(len(train_ans)):
    for i in range(cnt):
        bleu += bleu1(train_ans[i]['answer'], train_data[i]['answer'])
        p_1, r_1, f1_1 = precision_recall_f1(train_ans[i]['answer'], train_data[i]['answer'])
        p += p_1
        r += r_1
        f1 += f1_1
        all_prediction.append(train_ans[i]['answer'])
        all_ground_truth.append(train_data[i]['answer'])
    em = exact_match(all_prediction, all_ground_truth)
    print("bleu1:{}, em:{}, p:{}, r:{}, f1:{}".format(bleu / cnt, em, p / cnt, r / cnt, f1 / cnt))


def test_aps():
    """ 对测试文件进行答案抽取 """

    postagger = Postagger()  # 初始化实例
    postagger.load(pos_model_path)  # 加载模型

    segmentor = Segmentor()  # 初始化实例
    segmentor.load(cws_model_path)  # 加载模型

    parser = Parser()  # 初始化实例
    parser.load(par_model_path)  # 加载模型

    clf = joblib.load(ROUGH_MODEL_PATH)
    tv = joblib.load(TF_IDF_MODEL_PATH)
    # 读取sent json文件
    with open(TEST_RESULT, 'r', encoding='utf-8') as fin:
        items = [json.loads(line.strip()) for line in fin.readlines()]
    res = []
    none_cnt = 0
    for item in items:
        if len(item['answer_sentence']) == 0:
            del item['answer_pid']
            del item['answer_sentence']
            item['answer'] = ''
            res.append(item)
            continue
        # sent = ' '.join(jieba.cut(item['question']))
        sent = ' '.join(list(segmentor.segment(item['question'])))
        test_data = tv.transform([sent])
        label = clf.predict(test_data)[0]
        ans_sent = item['answer_sentence'][0]
        q_words = [word for word in segmentor.segment(item['question'])]
        ans_words = [word for word in segmentor.segment(ans_sent)]
        words_pos = postagger.postag(ans_words)
        if '：' in ans_sent:
            # item['answer'] = ''.join(ans_sent.split('：')[1:])
            item['answer'] = ans_sent.split('：')[1]
        elif ':' in ans_sent:
            item['answer'] = ans_sent.split(':')[1]
        elif label == 'HUM':  # 人物
            item['answer'] = extract_by_pos(q_words, ans_words, words_pos, ['nh', 'ni'])
        elif label == 'LOC':
            item['answer'] = extract_by_pos(q_words, ans_words, words_pos, ['nl', 'ns'])
        elif label == 'NUM':
            item['answer'] = extract_by_pos(q_words, ans_words, words_pos, ['m'])
        elif label == 'TIME':
            item['answer'] = extract_by_pos(q_words, ans_words, words_pos, ['nt'])
        # elif label == 'OBJ':
        #     item['answer'] = extract_by_pos(q_words, ans_words, words_pos, ['n'])
        # elif label == 'DES':
        #     item['answer'] = extract_by_pos(q_words, ans_words, words_pos, ['j'])
        else:
            item['answer'] = ''.join(ans_words)
            # item['answer'] = ''
        del item['answer_pid']
        del item['answer_sentence']
        res.append(item)
        if item['answer'] == '':
            none_cnt += 1
    # 写回文件
    with open(TEST_ANSWER, 'w', encoding='utf-8') as fout:
        for sample in res:
            fout.write(json.dumps(sample, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    train_test()
    evaluate()
    test_aps()
