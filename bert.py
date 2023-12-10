from text2vec import SentenceModel
from tqdm.notebook import tqdm


class Text2VecClz:
    def __init__(self):
        self.model = SentenceModel('shibing624/text2vec-base-chinese')

    def text2vec(self, comments):
        embeddings = []
        for comment in tqdm(comments, desc="Processing texts", dynamic_ncols=True):
            embedding = self.model.encode(comment)
            embeddings.append(embedding)
        return embeddings
