import pandas as pd

from tsne import TSNEDisplayer
from pca import PCADisplayer
from tfidf import TfidfClz
from bert import Text2VecClz
from analysis import analysis
import numpy as np
import os

# 设置工作空间 & 读取数据
train = pd.read_csv('./train.csv', sep='\t')

filename = 'embeddings.npz'

# tfidf_matrix = None
# if os.path.exists(filename):
#     data = np.load(filename)
#     tfidf_matrix = data['embeddings']
# else:
#     print(f"File: {filename} does not exist!")
#     tfidf_matrix = Text2VecClz().text2vec(train['comment'])
#     np.savez_compressed('embeddings.npz', embeddings=tfidf_matrix)
# from sklearn.preprocessing import normalize
# tfidf_matrix = normalize(tfidf_matrix)

tfidf_matrix = TfidfClz().text2vec(train['comment']).toarray()

# 进行模型训练
from sklearn.cluster import DBSCAN

# clustering = DBSCAN(eps=0.4, min_samples=3).fit(tfidf_matrix)
clustering = DBSCAN(eps=0.95, min_samples=5).fit(tfidf_matrix)

pd.set_option('display.max_rows', 1000)
TSNEDisplayer().show(tfidf_matrix, clustering)
analysis(train, clustering)
