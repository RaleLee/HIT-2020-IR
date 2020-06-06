import os
import math
import json
import numpy as np
from win32com import client as wc

from pyltp import Segmentor

LTP_DATA_DIR = 'D:/Course/IR/ltp_data_v3.4.0'
DATA_DIR = 'data/full_data.json'
PRE_PROCESSED_DATA_DIR = 'data/preprocessed.json'
STOP_WORDS = 'data/stopwords.txt'
FILE_DIR = 'files/doc'
BM25_PAGE_DIR = 'data/bm25_page.json'
BM25_FILE_DIR = 'data/bm25_file.json'
cws_model_path = os.path.join(LTP_DATA_DIR, 'cws.model')


class BM25:
    def __init__(self, docs, path=None):
        if path is not None and os.path.exists(path):
            self.load(path)
            return
        # list of document length in docs
        self.doc_len_list = [len(doc) for doc in docs]
        # average document length in docs
        self.avg_len_list = sum(self.doc_len_list) / len(self.doc_len_list)
        # frequency of words
        self.tf = []
        # IDF
        self.idf = {}
        # DF
        self.df = {}
        for doc in docs:
            freq = {}
            for word in doc:
                freq[word] = 1 if word not in freq else freq[word] + 1
            self.tf.append(freq)

            for word, f in freq.items():
                self.df[word] = 1 if word not in self.df else self.df[word] + 1

        for word, f in self.df.items():
            self.idf[word] = math.log(len(self.doc_len_list) - math.log(f))

    def save(self, path):
        bm25 = {'doc_len_list': self.doc_len_list, 'avg_len_list': self.avg_len_list,
                'doc_word_freq': self.tf, 'idf': self.idf}
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(bm25, f)

    def load(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            bm25 = json.load(f)
        self.doc_len_list = bm25['doc_len_list']
        self.avg_len_list = bm25['avg_len_list']
        self.tf = bm25['doc_word_freq']
        self.idf = bm25['idf']

    def score(self, query, specific_range=None):
        scores = []
        if specific_range is None:
            for idx in range(len(self.doc_len_list)):
                f = self.tf[idx]
                score = sum([self.idf[word] * f[word] * (2 + 1) /
                             (f[word] + 2 * (1 - 0.5 + 0.5 * self.doc_len_list[idx] / self.avg_len_list))
                             for word in query if word in f])
                scores.append(score)
        else:
            for idx in range(len(self.doc_len_list)):
                f = self.tf[idx]
                score = sum([self.idf[word] * f[word] * (2 + 1) /
                             (f[word] + 2 * (1 - 0.5 + 0.5 * self.doc_len_list[idx] / self.avg_len_list))
                             for word in query if word in f]) if idx in specific_range else 0
                scores.append(score)
        return scores


class InvertedIndex:
    def __init__(self, docs):
        self.dic = {}
        for idx, sentence in enumerate(docs):
            for word in sentence:
                if word not in self.dic:
                    self.dic[word] = set()
                self.dic[word].add(idx)

    def search(self, query):
        ret = set()
        for word in query:
            if word in self.dic:
                ret = ret | self.dic[word]
        return ret


def load():
    with open(PRE_PROCESSED_DATA_DIR, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f.readlines()]


class Search:
    def __init__(self):
        self.pages = load()
        self.files, docs_file = [], []
        for index, page in enumerate(self.pages):
            self.files += [(index, i) for i in range(len(page['segmented_file_name']))]
            docs_file += [page['segmented_title'] + file + file_content for file, file_content in
                          zip(page['segmented_file_name'], page['segmented_file_contents'])]

        docs_page = [page['segmented_title'] + page['segmented_paragraphs'] for page in self.pages]
        self.bm25_page = BM25(docs_page)
        self.bm25_page.save(BM25_PAGE_DIR)
        self.index_page = InvertedIndex(docs_page)

        self.bm25_file = BM25(docs_file)
        self.bm25_file.save(BM25_FILE_DIR)
        self.index_file = InvertedIndex(docs_file)

    def search(self, query, authority, mode='page'):
        segmentor = Segmentor()
        segmentor.load(cws_model_path)
        query = list(segmentor.segment(query))
        if mode == 'page':
            spec = self.index_page.search(query)
            scores = self.bm25_page.score(query, spec)
        else:
            spec = self.index_file.search(query)
            scores = self.bm25_file.score(query, spec)
        sorted_scores = np.argsort(-np.array(scores)).tolist()
        idx = sorted_scores.index(0)
        sorted_scores = sorted_scores[:idx]
        if mode == 'page':
            return [self.pages[index] for index in sorted_scores if self.pages[index]['authority'] <= authority]
        else:
            ret = [self.files[idx] for idx in sorted_scores]
            return [(self.pages[page_idx]['file_name'][file_idx], self.pages[page_idx]['title'],
                     self.pages[page_idx]['authority']) for page_idx, file_idx in ret
                    if self.pages[page_idx]['authority'] <= authority]


def convert_doc2docx():
    files = os.listdir('D:\\Course\\IR\\HIT-2020-IR\\Lab3\\files\\doc')
    word = wc.Dispatch('Word.Application')
    for item in files:
        if 'docx' in item:
            continue
        if 'doc' in item:
            doc = word.Documents.Open('D:\\Course\\IR\\HIT-2020-IR\\Lab3\\files\\doc\\' + item)
            doc.SaveAs('D:\\Course\\IR\\HIT-2020-IR\\Lab3\\files\\doc\\' + item + 'x', 12, False, "", True, "", False,
                       False, False, False)
            doc.Close()
    word.Quit()


if __name__ == '__main__':
    convert_doc2docx()
