# 加载所需要的包
import jieba
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from tsne import TSNEDisplayer

# 设置工作空间 & 读取数据
train = pd.read_csv('./train.csv', sep='\t')

data = train['comment'].apply(lambda x: ' '.join(jieba.lcut(x)))
vectorizer_word = TfidfVectorizer(max_features=800000,
                                  token_pattern=r"(?u)\b\w+\b",
                                  min_df=5,
                                  # max_df=0.1,
                                  analyzer='word',
                                  ngram_range=(1, 2)
                                  )
vectorizer_word = vectorizer_word.fit(data)
tfidf_matrix = vectorizer_word.transform(data)

# 进行模型训练
from sklearn.cluster import DBSCAN

clustering = DBSCAN(eps=0.98, min_samples=5).fit(tfidf_matrix)

pd.set_option('display.max_rows', 1000)
TSNEDisplayer().show(tfidf_matrix, clustering)

# 我们看看分了所少个群，每个群的样本数是多少
len(pd.Series(clustering.labels_).value_counts())

pd.Series(clustering.labels_).value_counts()

# 分群标签打进原来的数据
train['labels_'] = clustering.labels_

# 抽取编号为5的群看看 可以看到，这个群都是吃拉肚子的反馈，聚类效果还是非常可以的
for i in train[train['labels_'] == 1]['comment']:
    print(i)

# 对每个标签分组，然后抽出几个样本查看
grouped = train.groupby('labels_')

for name, group in grouped:
    print("\nCluster:", name)
    print("Number of comments in this cluster:", len(group))
    print("Sample comments:")
    # 抽取5个样本查看
    if len(group) >= 5:
        # 抽取5个样本查看
        sample_comments = group['comment'].sample(5)
    else:
        # 如果样本数少于5，则全部输出
        sample_comments = group['comment']
    for comment in sample_comments:
        print(comment)
