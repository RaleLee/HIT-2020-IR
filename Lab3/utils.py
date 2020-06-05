import os
import math
import json


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


class Search:
    def __init__(self):
        pass
