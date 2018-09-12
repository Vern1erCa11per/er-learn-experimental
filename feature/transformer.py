import itertools
import re

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer
from sklearn.preprocessing import binarize

DEFAULT_PUNCTUATIONS = r"'!\"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n"
DEFAULT_PUNCTUATIONS_PATTERN = re.compile(r"['!\"#$%&()*+,-./:;<=>?@\[\\\]^_`{|}~\t\n]")

def strip_puncs(rows):
    def f(row):
        return [DEFAULT_PUNCTUATIONS_PATTERN.sub("", row[0])]
    if len(rows) == 1:
        return DEFAULT_PUNCTUATIONS_PATTERN.sub("", rows[0])

    return np.apply_along_axis(func1d=f, arr=rows.reshape((rows.shape[0], 1)), axis=1)

def remove_from_tokens(token_rows):
    # TODO replace more efficient way
    cleaned = [[token.lower().translate(DEFAULT_PUNCTUATIONS) for token in tokens
                if token.translate(DEFAULT_PUNCTUATIONS) != ''] for tokens in token_rows]
    return cleaned

class StripPunctuations(FunctionTransformer):

    def __init__(self):
        super(StripPunctuations, self).__init__(strip_puncs, validate=False)

class StripPunctuationsFromToken(FunctionTransformer):

    def __init__(self):
        super().__init__(remove_from_tokens, validate=False)

#
# def extract_features_from_pair_columns(col1, col2, extractor, check_common_func):
#     amazon_size = amazon_df.shape[0]
#     google_size = google_df.shape[0]
#     concatted = np.vstack([amazon_df[col1].values.reshape((amazon_size, 1)),
#                            google_df[col2].values.reshape((google_size, 1))])
#     # print(concatted.shape)
#     concat_feature = extractor().fit_transform(concatted)
#     return check_common_func(concat_feature[:amazon_df.shape[0]], concat_feature[amazon_df.shape[0]:])


def common_elemnt(x):
    return binarize(np.dot(x[0], x[1].T))

class CommonElementFeatureBasedOnOneHotEncoding(FunctionTransformer):

    def __init__(self):
        super(CommonElementFeatureBasedOnOneHotEncoding, self).__init__(common_elemnt, validate=False)


def has_common_token(x, y):
    common_tokens = set(x) & set(y)
    return int(len(common_tokens) > 0)

class HasCommonElement(FunctionTransformer):

    def __init__(self):
        super(HasCommonElement, self).__init__(has_common_token, validate=False)


class TokenOneHotEncoder(Pipeline):

    def __init__(self):
        super(TokenOneHotEncoder, self).__init__([("strip", StripPunctuations()),
                                                  ("encode", OneHotEncoder())])

class UnigramBinaryBoW(CountVectorizer):

    def __int__(self):
        super(UnigramBinaryBoW, self).__init__(binary=True, preprocessor=strip_puncs)
    unigrm_countvectorizer = lambda: CountVectorizer()

class BigramBinaryBoW(CountVectorizer):

    def __init__(self):
        super(BigramBinaryBoW, self).__init__(binary=True, preprocessor=strip_puncs, ngram_range=(2, 2))

quadrigram_count_vectorizer = lambda: CountVectorizer(binary=True, preprocessor=strip_puncs, ngram_range=(4, 4))

sexagram_count_vectorizer = lambda: CountVectorizer(binary=True, preprocessor=strip_puncs, ngram_range=(6, 6))

#
# # only for description
# key1, key2 = key_features_columns[-2]
# common_token_features["quadrigram_value_{}_{}".format(key1, key2)] = extract_features_from_pair_columns(key1, key2,
#                                                                                                         quadrigram_count_vectorizer,
#                                                                                                         common_elemnt)

def extract_digit_token(rows):
    def f(row):
        if len(row) <= 0:
            return []
        return re.findall(r'\d+', row[0])

    result = [' '.join(f(row)) for row in rows]
    return result
#
# class DigitTokenizer(CountVectorizer):
#
#     def __init__(self):
#         super(DigitTokenizer, self).__init__(tokenizer=extract_digit_token, stop_words=None)
#
# class DigitTokenEncoder():
#
#     @staticmethod
#     def encode(X, , extractor):
#         concatted = preprocess_func(X)
#         concat_feature = extractor().fit_transform(X)

# # for multi label encoder
# def extract_features_with_preprocess(col1, col2, extractor, check_common_func, prerpcess_func):
#     amazon_size = amazon_df.shape[0]
#     google_size = google_df.shape[0]
#     concatted = np.vstack([amazon_df[col1].values.reshape((amazon_size, 1)),
#                            google_df[col2].values.reshape((google_size, 1))])
#     concatted = prerpcess_func(concatted)
#     concat_feature = extractor().fit_transform(concatted)
#     return check_common_func(concat_feature[:amazon_df.shape[0]], concat_feature[amazon_df.shape[0]:])

#
#
# # don't include numeric value
# for key1, key2 in key_features_columns:
#     feature_name = "common_digit_{}_{}".format(key1, key2)
#     common_token_features[feature_name] = extract_features_with_preprocess(key1, key2, common_digit_token,
#                                                                            common_elemnt,
#                                                                            prerpcess_func=extract_digit_token)

def off_by_ones(rows):
    def extract_digit(row):
        if len(row) <= 0:
            return []
        return re.findall(r'\d+', row[0])

    def f(element):
        digits = int(element)
        return [str(digits - 1), str(digits), str(digits + 1)]

    return [" ".join(itertools.chain.from_iterable([f(element) for element in extract_digit(row)])) for row in rows]


class OffByOneDigitTokenizer(FunctionTransformer):
    def __init__(self):
        super(OffByOneDigitTokenizer, self).__init__(func=off_by_ones, validate=False)

class PreprocessableMultiLabelBinarizer(MultiLabelBinarizer):

    def __init__(self, func, classes=None, sparse_output=False):
        self.preprocess_func = func
        super(PreprocessableMultiLabelBinarizer, self).__init__(classes, sparse_output)

    def fit(self, y):
        return super().fit(self.preprocess_func(y))

    def fit_transform(self, y):
        return super().fit_transform(self.preprocess_func(y))

    def transform(self, y):
        return super().transform(self.preprocess_func(y))

class DigitTokenEncoder(PreprocessableMultiLabelBinarizer):

    def __init__(self, classses=None, sparse_output=False):
        super(DigitTokenEncoder, self).__init__(extract_digit_token, classses, sparse_output)

class OffByOneDigitEncoder(PreprocessableMultiLabelBinarizer):

    def __init__(self, classes=None, sparse_output=False):
        super(OffByOneDigitEncoder, self).__init__(off_by_ones, classes, sparse_output)

#
# # don't include numeric value
# for key1, key2 in key_features_columns:
#     feature_name = "common_off_by_one_digit_{}_{}".format(key1, key2)
#     print(feature_name)
#     common_token_features[feature_name] = extract_features_with_preprocess(key1, key2, MultiLabelBinarizer,
#                                                                            common_elemnt, prerpcess_func=off_by_ones)

def extract_first_n_chars(rows, n):
    def f(text):
        if len(text) < n:
            return text
        return [text[:n]]

    return np.apply_along_axis(func1d=f, arr=rows, axis=1)

class FirstCharsTransformer(FunctionTransformer):

    def __init__(self, n):
        super().__init__(lambda rows: extract_first_n_chars(rows, n), validate=False)

class FirstCharsEncoder(OneHotEncoder):

    def __init__(self, n_chars):
        super(FirstCharsEncoder, self).__init__()
        self.n_chars = n_chars
        self.extract_n_chars = lambda x: extract_first_n_chars(x, self.n_chars)

    def fit(self, X, y=None):
        return super().fit(self.extract_n_chars(X), y)

    def fit_transform(self, X, y=None):
        return super().fit_transform(self.extract_n_chars(X), y)

    def transform(self, X):
        return super().transform(self.extract_n_chars(X))

# from sklearn.feature.text import TfidfVectorizer
#
# # In[ ]:
#
#
# unigram_tfidf_vectorizer = lambda: TfidfVectorizer(decode_error='ignore', preprocessor=strip_puncs)
#
# # In[ ]:
#
#
# similar_features.append(unigram_tfidf_vectorizer)
#
# # In[ ]:
#
#
# trigram_tfidf_vectorizer = lambda: TfidfVectorizer(decode_error='ignore', preprocessor=strip_puncs, ngram_range=(3, 3))
#
# # In[ ]:
#
#
# similar_features.append(trigram_tfidf_vectorizer)
#
# # In[ ]:
#
#
# quinquegram_tfidf_vectorizer = lambda: TfidfVectorizer(decode_error='ignore', preprocessor=strip_puncs,
#                                                        ngram_range=(5, 5))
#
# # In[ ]:
#
#
# similar_features.append(quinquegram_tfidf_vectorizer)
#
# # In[ ]:
#
#
# amazon_df.columns
#
# # In[ ]:
#
#
# google_df.columns
#
# # In[ ]:
#
#
# common_extracted_features = [extract_features_from_pair_columns(col1, col2, common_features) for col1, col2 in
#                              key_features_columns]
#
