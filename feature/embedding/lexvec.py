
from gensim.models import KeyedVectors
from sklearn.preprocessing import FunctionTransformer
import numpy as np

DEFAULT_LEXVEC_FILE = "../../data/pretrained/lexvec.enwiki+newscrawl.300d.W.pos.vectors"

class EmbeddinTransformer(FunctionTransformer):

    def __init__(self, emb_file=None, oov='zero'):
        if not emb_file:
            emb_file = DEFAULT_LEXVEC_FILE
        self.model: KeyedVectors = KeyedVectors.load_word2vec_format(emb_file, binary=False)

        super().__init__(func=self.tokens_list_to_embedding, validate=False)

    def token_to_embbeding(self, token):
        # TODO oov emb with random value
        return self.model.get_vector(token) \
            if token in self.model.vocab else np.zeros(self.model.vector_size)

    def tokens_to_embedding(self, tokens):
        return np.array([self.token_to_embbeding(token) for token in tokens])

    def tokens_list_to_embedding(self, toknes_list):
        return [self.tokens_to_embedding(tokens) for tokens in toknes_list]
