from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


class PCADisplayer:
    def show(self, tfidf_matrix, clustering):
        tfidf_matrix = tfidf_matrix.toarray()[clustering.labels_ != -1]
        labels_filtered = clustering.labels_[clustering.labels_ != -1]

        #  设置下最大可展示的行
        # 使用 PCA 进行降维到3维
        pca_3d = PCA(n_components=3)
        pca_tf_idf_3d = pca_3d.fit_transform(tfidf_matrix.toarray())

        # 准备创建3D图
        fig = plt.figure()

        # 创建3D画布
        ax = plt.axes(projection='3d')

        # 为每个聚类设置颜色
        colors = labels_filtered

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
