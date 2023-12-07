from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class TSNEDisplayer:
    def show(self, tfidf_matrix, clustering):
        tfidf_matrix = tfidf_matrix.toarray()[clustering.labels_ != -1]
        labels_filtered = clustering.labels_[clustering.labels_ != -1]

        # 使用 t-SNE 进行降维到3维
        tsne = TSNE(n_components=3)
        tsne_data = tsne.fit_transform(tfidf_matrix)

        # 准备创建3D图
        fig = plt.figure(figsize=(5, 5))

        # 创建3D画布
        ax = fig.add_subplot(111, projection='3d')

        # 所有簇的颜色标签
        colors = labels_filtered

        # 创建3D散点图
        ax.scatter(tsne_data[:, 0], tsne_data[:, 1], tsne_data[:, 2], c=colors, cmap='viridis')

        # 显示图形
        plt.show()
