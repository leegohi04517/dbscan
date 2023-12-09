import pandas as pd


def analysis(train, clustering):
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
