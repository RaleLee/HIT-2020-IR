import os

from pyltp import Segmentor

LTP_DATA_DIR = 'D:/Course/IR/ltp_data_v3.4.0'
cws_model_path = os.path.join(LTP_DATA_DIR, 'cws.model')

segmentor = Segmentor()
segmentor.load(cws_model_path)
words = segmentor.segment("元芳你怎么看")
print('\t'.join(words))
segmentor.release()