import logging
from collections import OrderedDict

import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.mixture import GaussianMixture

logger = logging.getLogger(__name__)


class Scdv(BaseEstimator, ClusterMixin, TransformerMixin):
    """ implementation of https://dheeraj7596.github.io/SDV/"""
    # TODO accept args for idfvectorizer
    def __init__(self, word_emb_func=None, stop_words=None, sparse_threshold_p=0.04,
                 n_components=50, covariance_type="full", tol=0.001, reg_covar=1e-06, max_iter=100, n_init=1,
                 init_params="kmeans", weights_init=None, means_init=None, precisions_init=None, random_state=None,
                 warm_start=False, verbose=0, verbose_interval=10):
        self.semantic_soft_clustering = GaussianMixture(n_components, covariance_type, tol, reg_covar, max_iter, n_init,
                                                        init_params, weights_init, means_init, precisions_init,
                                                        random_state,
                                                        warm_start, verbose, verbose_interval)
        super(Scdv, self).__init__()

        self._tfidf_vectorizer = IdfStoredCountVectorizer(stop_words=stop_words, use_idf=True, smooth_idf=True)

        self.sparse_threshold_p = sparse_threshold_p
        self._sparse_threshold_ratio = None
        self.sparse_threshold = None

        self._word_emb_func = word_emb_func if word_emb_func else lambda x: x
        self._word_embs = {}

        self._word_topic_vecs = None

    def __getstate__(self):
        state = self.__dict__.copy()
        # Remove the unpicklable entries.
        del state['_word_emb_func']
        return state

    def to_idf(self, index):
        word_index = index

        if isinstance(index, str):
            word_index = self._tfidf_vectorizer.vocabulary_[index]
        return self._tfidf_vectorizer.idf_[word_index]

    def fit(self, docs, y=None):
        logger.info("creating dictionary and computing idf...")
        self._tfidf_vectorizer.fit(docs)

        self._word_embs = OrderedDict({index: self._word_emb_func(word) for word, index in
                                       sorted(self._tfidf_vectorizer.vocabulary_.items(),
                                              key=lambda key_value: key_value[1])})

        word_vecs = np.vstack(list(self._word_embs.values()))

        logger.info("clustering in-vocabulary words (size: %d) ...", len(word_vecs))
        self.semantic_soft_clustering.fit(word_vecs)

        logger.info("getting word-topic_vectors...")
        self._word_topic_vecs = self._to_word_topic_vectors(word_vecs)

        logger.info("computing threshold to make sparse...")
        self._compute_sparse_threshold_ratio(self.transform_into_ncdv(docs))

        logger.info("fitting has finished!!")
        return self

    def transform(self, raw_documents, should_compress=True):
        return self._make_sparse(self.transform_into_ncdv(raw_documents), should_compress)

    def fit_transform(self, raw_documents, y=None, **kwargs):
        return self.fit(raw_documents).transform(raw_documents)

    def _to_word_topic_vectors(self, word_vecs):
        semantic_cluster_probs = self.semantic_soft_clustering.predict_proba(word_vecs)
        # TODO replace with faster way. Can I use mode product?
        topic_vector_dim = (word_vecs.shape[0], semantic_cluster_probs.shape[1] * word_vecs.shape[1])
        word_topic_vectors = np.zeros(topic_vector_dim)
        for i, word_index in zip(range(word_vecs.shape[0]), self._word_embs.keys()):
            word_vec = word_vecs[i]
            word_vec = word_vec.reshape((word_vec.shape[0], 1))
            word_cluster_prob = semantic_cluster_probs[i]
            word_cluster_prob = word_cluster_prob.reshape((1, word_cluster_prob.shape[0]))
            word_topic_vectors[i, :] = (np.dot(word_vec, word_cluster_prob)
                                       * self.to_idf(word_index)).reshape(word_topic_vectors.shape[1:])
        return word_topic_vectors

    def _compute_sparse_threshold_ratio(self, non_sparse_doc_vectors):

        feature_min_value = np.mean(np.min(non_sparse_doc_vectors, axis=0))
        feature_max_value = np.mean(np.max(non_sparse_doc_vectors, axis=0))

        self._sparse_threshold_ratio = (np.abs(feature_max_value) + np.abs(feature_min_value)) / 2
        self.sparse_threshold = self.sparse_threshold_p * self._sparse_threshold_ratio

    def transform_into_ncdv(self, raw_documents):
        bag_of_words_list = self._tfidf_vectorizer.transform(raw_documents)

        doc_vectors = bag_of_words_list.dot(self._word_topic_vecs)
        return doc_vectors

    def _make_sparse(self, matrix, should_compress=True):
        matrix[np.where(np.abs(should_compress) < self.sparse_threshold)] = 0
        if should_compress:
            return sparse.csr_matrix(matrix)
        return matrix


class IdfStoredCountVectorizer(TfidfVectorizer):

    def __init__(self, input='content', encoding='utf-8', decode_error='strict', strip_accents=None, lowercase=True,
                 preprocessor=None, tokenizer=None, analyzer='word', stop_words=None, token_pattern=r"(?u)\b\w\w+\b",
                 ngram_range=(1, 1), max_df=1.0, min_df=1, max_features=None, vocabulary=None, binary=False,
                 dtype=np.int64, norm='l2', use_idf=True, smooth_idf=True, sublinear_tf=False):
        super().__init__(input, encoding, decode_error, strip_accents, lowercase, preprocessor, tokenizer, analyzer,
                         stop_words, token_pattern, ngram_range, max_df, min_df, max_features, vocabulary, binary,
                         dtype, norm, use_idf, smooth_idf, sublinear_tf)

    def transform(self, raw_documents, copy=True):
        return super(TfidfVectorizer, self).transform(raw_documents)

    def fit_transform(self, raw_documents, y=None):
        return self.fit(raw_documents, y).transform(raw_documents)
