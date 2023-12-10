import jieba
from sklearn.feature_extraction.text import TfidfVectorizer


class TfidfClz:
    def text2vec(self, comment):
        data = comment.apply(lambda x: ' '.join(jieba.lcut(x)))
        vectorizer_word = TfidfVectorizer(max_features=800000,
                                          token_pattern=r"(?u)\b\w+\b",
                                          min_df=5,
                                          # max_df=0.1,
                                          analyzer='word',
                                          ngram_range=(1, 2)
                                          )
        vectorizer_word = vectorizer_word.fit(data)
        tfidf_matrix = vectorizer_word.transform(data)
        return tfidf_matrix
