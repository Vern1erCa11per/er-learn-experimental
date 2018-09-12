# coding: utf-8

import fastText
import numpy as np

from sklearn.preprocessing import FunctionTransformer

module = fastText.load_model("data/wiki.en/wiki.en.bin")
#puncs = str.maketrans("", "", r"'!\"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n")

def to_subword_embbeding(tokenized_docs):
    return [np.array([module.get_word_vector(token) for token in doc]) for doc in tokenized_docs]

#
# def to_emb(tokens):
#
# def to_embs(sentence):
#     if pd.isnull(sentence):
#         return []
#     tokens = fastText.tokenize(sentence)
#     tokenized = [token.lower().translate(puncs) for token in tokens]
#     return np.array([modle.get_word_vector(word) for word in tokenized])

class FastTextTokenizer(FunctionTransformer):
    def __init__(self):
        super().__init__(lambda X: [fastText.tokenizer(x) if x else [] for x in X], validate=False)
