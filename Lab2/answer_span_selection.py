from sklearn.externals import joblib
from metric import *
from pyltp import Segmentor, Postagger
import json
import os
from preprocessed import LTP_DATA_DIR, TRAIN_DATA, cws_model_path
from answer_sentence_selection import TEST_RESULT
from question_classification import ROUGH_MODEL_PATH, TF_IDF_MODEL_PATH

TRAIN_ANSWER = 'data/train_answer.json'
TEST_ANSWER = 'data/test_answer.json'
pos_model_path = os.path.join(LTP_DATA_DIR, 'pos.model')


# noinspection PyArgumentList,PyArgumentList,PyArgumentList,PyArgumentList,PyArgumentList,PyArgumentList
def progress(is_train=False):
    pos_tagger = Postagger()
    pos_tagger.load(pos_model_path)
    segmentor = Segmentor()
    segmentor.load(cws_model_path)

    clf = joblib.load(ROUGH_MODEL_PATH)
    tv = joblib.load(TF_IDF_MODEL_PATH)
    # 读取json文件
    path = TRAIN_DATA if is_train else TEST_RESULT
    with open(path, 'r', encoding='utf-8') as f:
        questions = [json.loads(line.strip()) for line in f.readlines()]
    res = []
    none_cnt = 0
    for question in questions:
        if is_train:
            ans_sent = ' '.join(question['answer_sentence'])
        else:
            if len(question['answer_sentence']) == 0:
                question.pop('answer_pid')
                question.pop('answer_sentence')
                question['answer'] = ''
                res.append(question)
                continue
            ans_sent = question['answer_sentence'][0]
        sent = ' '.join(list(segmentor.segment(question['question'])))
        test_data = tv.transform([sent])
        label = clf.predict(test_data)[0]
        ans_words = [word for word in segmentor.segment(ans_sent)]
        words_pos = pos_tagger.postag(ans_words)
        if '：' in ans_sent:
            question['answer'] = ans_sent.split('：')[1]
        elif ':' in ans_sent:
            question['answer'] = ans_sent.split(':')[1]
        elif label == 'HUM':
            question['answer'] = pos_answer(ans_words, words_pos, ['nh', 'ni'])
        elif label == 'LOC':
            question['answer'] = pos_answer(ans_words, words_pos, ['nl', 'ns'])
        elif label == 'NUM':
            question['answer'] = pos_answer(ans_words, words_pos, ['m'])
        elif label == 'TIME':
            question['answer'] = pos_answer(ans_words, words_pos, ['nt'])
        else:
            question['answer'] = ''.join(ans_words)
        if not is_train:
            question.pop('answer_pid')
            question.pop('answer_sentence')
        res.append(question)
        if question['answer'] == '':
            none_cnt += 1
    print("none_cnt: {}".format(none_cnt))
    # 写回json文件
    answer_path = TRAIN_ANSWER if is_train else TEST_ANSWER
    with open(answer_path, 'w', encoding='utf-8') as f:
        for sample in res:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')


def pos_answer(words, words_pos, pos):
    res = []
    for i in range(len(words_pos)):
        if words_pos[i] in pos:
            res.append(words[i])
    if len(res):
        return ''.join(res)
    else:
        return ''.join(words)


def evaluate():
    """
    使用老师给出的metrics来计算bleu得分
    :return: 评测结果
    """
    with open(TRAIN_DATA, 'r', encoding='utf-8') as f:
        train_data = [json.loads(line.strip()) for line in f.readlines()]
    with open(TRAIN_ANSWER, 'r', encoding='utf-8') as f:
        train_ans = [json.loads(line.strip()) for line in f.readlines()]
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
    print("bleu1:{}, exact_match:{},\np:{}, r:{}, f1:{}".format(bleu / cnt, em, p / cnt, r / cnt, f1 / cnt))


if __name__ == '__main__':
    progress(True)
    evaluate()
    progress()
