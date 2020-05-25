import json
import numpy as np

from answer_sentence_selection import SVM_RANK_TRAIN_RESULT, SVM_RANK_DEV_DATA
from answer_span_selection import TRAIN_ANSWER
from metric import precision_recall_f1, bleu1, exact_match
from preprocessed import TRAIN_DATA


def calculate_mrr():
    """
    计算开发集上的完美匹配率 mrr
    :return:
    """
    with open(SVM_RANK_TRAIN_RESULT, 'r', encoding='utf-8') as f:
        predictions = np.array([float(line.strip()) for line in f.readlines()])
    dev = []
    with open(SVM_RANK_DEV_DATA, 'r', encoding='utf-8') as f:
        i = 0
        for line in f.readlines():
            dev.append((i, int(line[0]), int(line.split(' ')[1].split(':')[1])))
            i += 1
    old_pid = dev[0][2]
    q_s = 0
    question_num = 0
    question_with_answer = 0
    prefect_correct = 0
    mrr = 0.0
    for i in range(len(dev)):
        if dev[i][2] != old_pid:
            p = np.argsort(-predictions[q_s:i]) + q_s
            for k in range(len(p)):
                if dev[p[k]][1] == 1:
                    question_with_answer += 1
                    if k == 0:
                        prefect_correct += 1
                    mrr += 1.0 / float(k + 1)
                    break
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
