# 加载所需要的包
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import os
import jieba

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
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

clustering = DBSCAN(eps=0.95, min_samples=5).fit(tfidf_matrix)

# # 使用 PCA 进行降维处理。选择门限95%，也就是降维后的总解释方差占比
# pca = PCA(n_components=0.95)
#
# # 注意：PCA的输入数据需要先进行归一化。我们之前对TF-IDF向量进行过归一化，所以这里不需要再次操作
# pca_tf_idf = pca.fit_transform(tfidf_matrix.toarray())
#
# # 按照聚类结果给数据点上色
# plt.scatter(pca_tf_idf[:, 0], pca_tf_idf[:, 1], c=clustering.labels_)
#
# # 标签为-1的数据点为噪声点，我们用黑色标出
# # plt.scatter(pca_tf_idf[clustering.labels_ == -1, 0], pca_tf_idf[clustering.labels_ == -1, 1], color='black')
#
# plt.show()

#  设置下最大可展示的行
# 使用 PCA 进行降维到3维
pca_3d = PCA(n_components=3)
pca_tf_idf_3d = pca_3d.fit_transform(tfidf_matrix.toarray())

# 准备创建3D图
fig = plt.figure()

# 创建3D画布
ax = plt.axes(projection='3d')

# 为每个聚类设置颜色
colors = clustering.labels_

# 创建3D散点图
p = ax.scatter3D(pca_tf_idf_3d[:, 0],
                 pca_tf_idf_3d[:, 1],
                 pca_tf_idf_3d[:, 2],
                 c=colors,
                 cmap='viridis')

# 添加颜色条
fig.colorbar(p)

# 显示图形
plt.show()


pd.set_option('display.max_rows', 1000)

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