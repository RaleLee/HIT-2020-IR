from sklearn.externals import joblib
from metric import *
from pyltp import Segmentor, Parser, Postagger
import json
import os
from preprocessed import LTP_DATA_DIR, TRAIN_DATA, cws_model_path
from answer_sentence_selection import TEST_RESULT
from question_classification import ROUGH_MODEL_PATH, TF_IDF_MODEL_PATH

TRAIN_ANSWER = 'data/train_answer.json'
TEST_ANSWER = 'data/test_answer.json'
pos_model_path = os.path.join(LTP_DATA_DIR, 'pos.model')
par_model_path = os.path.join(LTP_DATA_DIR, 'parser.model')


# noinspection PyArgumentList,PyArgumentList,PyArgumentList,PyArgumentList,PyArgumentList,PyArgumentList
def progress(is_train=False):
    """对训练集进行答案提取 测试性能"""
    pos_tagger = Postagger()  # 初始化实例
    pos_tagger.load(pos_model_path)  # 加载模型

    segmentor = Segmentor()  # 初始化实例
    segmentor.load(cws_model_path)  # 加载模型

    parser = Parser()  # 初始化实例
    parser.load(par_model_path)  # 加载模型

    clf = joblib.load(ROUGH_MODEL_PATH)
    tv = joblib.load(TF_IDF_MODEL_PATH)
    # 读取json文件
    path = TRAIN_DATA if is_train else TEST_RESULT
    with open(path, 'r', encoding='utf-8') as fin:
        items = [json.loads(line.strip()) for line in fin.readlines()]
    res = []
    none_cnt = 0
    for item in items:
        if is_train:
            ans_sent = ' '.join(item['answer_sentence'])
        else:
            if len(item['answer_sentence']) == 0:
                del item['answer_pid']
                del item['answer_sentence']
                item['answer'] = ''
                res.append(item)
                continue
            ans_sent = item['answer_sentence'][0]
        sent = ' '.join(list(segmentor.segment(item['question'])))
        test_data = tv.transform([sent])
        label = clf.predict(test_data)[0]
        ans_words = [word for word in segmentor.segment(ans_sent)]
        words_pos = pos_tagger.postag(ans_words)
        if '：' in ans_sent:
            # item['answer'] = ''.join(ans_sent.split('：')[1:])
            item['answer'] = ans_sent.split('：')[1]
        elif ':' in ans_sent:
            item['answer'] = ans_sent.split(':')[1]
        elif label == 'HUM':  # 人物
            item['answer'] = extract_by_pos(ans_words, words_pos, ['nh', 'ni'])
        elif label == 'LOC':
            item['answer'] = extract_by_pos(ans_words, words_pos, ['nl', 'ns'])
        elif label == 'NUM':
            item['answer'] = extract_by_pos(ans_words, words_pos, ['m'])
        elif label == 'TIME':
            item['answer'] = extract_by_pos(ans_words, words_pos, ['nt'])
        else:
            item['answer'] = ''.join(ans_words)
        if not is_train:
            del item['answer_pid']
            del item['answer_sentence']
        res.append(item)
        if item['answer'] == '':
            none_cnt += 1
    print("none_cnt: {}".format(none_cnt))
    segmentor.release()
    pos_tagger.release()  # 释放模型
    parser.release()  # 释放模型
    # 写回json文件
    answer_path = TRAIN_ANSWER if is_train else TEST_ANSWER
    with open(answer_path, 'w', encoding='utf-8') as fout:
        for sample in res:
            fout.write(json.dumps(sample, ensure_ascii=False) + '\n')


def extract_by_pos(words, words_pos, pos):
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
    # 读取json文件
    with open(TRAIN_DATA, 'r', encoding='utf-8') as fin:
        train_data = [json.loads(line.strip()) for line in fin.readlines()]
    # 读取json文件
    with open(TRAIN_ANSWER, 'r', encoding='utf-8') as fin:
        train_ans = [json.loads(line.strip()) for line in fin.readlines()]
    cnt = len(train_data)
    all_prediction = []
    all_ground_truth = []
    bleu = 0.0
    p = 0.0
    r = 0.0
    f1 = 0.0
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


if __name__ == '__main__':
    progress(True)
    evaluate()
    progress()
