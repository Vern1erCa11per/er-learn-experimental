import itertools
import logging
from collections import namedtuple, UserList
from pathlib import Path
from typing import Union, Callable

import numpy as np
from sklearn.externals import joblib
from sklearn.pipeline import Pipeline
from scipy import sparse

logger = logging.getLogger(__name__)

class ConcatPipeline(Pipeline):

    def __init__(self, steps, memory=None):
        super().__init__(steps, memory)

    def fit(self, X, y=None, **fit_params):
        """

        :param X:  should be list of numpy array or list
        :param y:
        :param fit_params:
        :return:
        """
        stacked_X, sizes = self._stack(X)

        stacked_y = None
        if y:
            stacked_y, y_sizes = self._stack(y)
            assert len(sizes) == len(y_sizes)

        return super().fit(X=stacked_X, y=stacked_y, **fit_params)

    def _transform(self, X):
        return self._apply_transform_with_stacking(X, func=super()._transform)

    def predict(self, X):
        return self._apply_transform_with_stacking(X, func=super().predict)

    def _apply_fit_with_stacking(self, X, func, y=None, **fit_params):
        stacked_X, sizes = self._stack(X)

        stacked_y = y
        if y:
            stacked_y, _ = self._stack(y)
        stacked_predicts = func(stacked_X, stacked_y, **fit_params)

        return self._split(stacked_predicts, sizes)

    def _apply_transform_with_stacking(self, X, func):
        stacked_X, sizes = self._stack(X)

        stacked_predicts = func(stacked_X)

        return self._split(stacked_predicts, sizes)

    def _stack(self, X):

        sizes = [len(x) if not sparse.issparse(x) else x.getnnz() for x in X]

        if all(sparse.issparse(x) for x in X):
            stacked_X = sparse.vstack(X)
        elif all(isinstance(x, np.ndarray) for x in X):
            stacked_X = np.vstack(X)
        else:
            stacked_X = list(itertools.chain.from_iterable(X))

        return stacked_X, sizes

    def _split(self, X, sizes):
        split_indices = [(sum((sizes[:i])), sum(sizes[:i + 1])) if i > 0 else (0, sizes[i]) for i, size in enumerate(sizes)]
        predicts = [X[start:end] for start, end in split_indices]

        return predicts

    def fit_predict(self, X, y=None, **fit_params):
        return self._apply_fit_with_stacking(X, super().fit_predict, y, **fit_params)

    def fit_transform(self, X, y=None, **fit_params):
        return self._apply_fit_with_stacking(X, super().fit_transform, y, **fit_params)

PipelineFuncMap = namedtuple('PipelineFuncMap', ('func', 'type'))

class KeyColumnFeatureExtractor(object):

    def __init__(self, key_column_correspondences, pipeline_func_map_list, pipeline_save_dir, low_memory=False):
        self.key_column_correspondences = key_column_correspondences
        self.n_dfs = len(key_column_correspondences[0])
        if not all(self.n_dfs == len(correspondence) for correspondence in self.key_column_correspondences):
            raise AssertionError("The number of the columns in each correspondence should be same.")

        self.n_keys = len(self.key_column_correspondences)

        self.pipeline_func_map_list = pipeline_func_map_list
        self._pipeline_model: Union[ConcatPipeline] = None

        self.pipeline_save_dir = Path(pipeline_save_dir)
        self.pipeline_save_paths = []
        self.low_memory = low_memory
        self.pipeline_models = []

    def fit(self, dfs):
        self.pipeline_save_paths = []
        self.pipeline_models = []

        if len(dfs) != self.n_dfs:
            raise AssertionError("The number of dfs should be same with that of key column correspondences")

        for i, key_column in enumerate(self.key_column_correspondences):
            logging.info("extracting features from {}".format(str(key_column)))
            pipeline_func = self.pipeline_func_map_list[i]
            Xs = [df[column].tolist() for df, column in zip(dfs, key_column)]
            pipeline = self._fit_each_key(Xs, pipeline_func)

            self.pipeline_save_dir.mkdir(parents=True, exist_ok=True)
            model_save_path = self.pipeline_save_dir.joinpath("{}_concat_model.pickle".format(
                "_".join(key_column)
            ))
            self.pipeline_save_paths.append(str(model_save_path))

            joblib.dump(pipeline, model_save_path)
            if not self.low_memory:
                self.pipeline_models.append(pipeline)

        return self

    def _fit_each_key(self, Xs, pipeline_func):
        #
        # if isinstance(pipeline, list):
            # pipeline_models = [func() for func in pipeline]
            # for model, X in zip(pipeline_models, Xs):
            #     model.fit(X)
            # return pipeline_models
        #
        # if self.pipeline_types == "func":
        #     return self
        #
        # elif self.pipeline_types == "concat":
        #     self._pipeline_model = self.pipeline_func_map_list()
        #     self._pipeline_model.fit(Xs)
        #     return self
        # else:
        #     self._pipeline_model =

        pipeline_model: ConcatPipeline = pipeline_func()
        pipeline_model.fit(Xs)
        return pipeline_model

    def transform(self, dfs, save_dir=None):
        if len(dfs) != self.n_dfs:
            raise AssertionError("The number of dfs should be same with that of key column correspondences")

        features_list = []
        for i, key_column in enumerate(self.key_column_correspondences):
            Xs = [df[column].tolist() for df, column in zip(dfs, key_column)]

            if len(self.pipeline_models) == self.n_keys:
                pipeline = self.pipeline_models[i]
            else:
                pipeline = joblib.load(self.pipeline_save_paths[i])
            features_list.append(pipeline.transform(Xs))
            if save_dir:
                save_dir_path = Path(save_dir)
                save_dir_path.mkdir(exist_ok=True, parents=True)
                for i, (column_name, X) in enumerate(zip(key_column, Xs)):
                    save_path = save_dir_path.joinpath("df{}_{}_feature.pickle".format(i, column_name))
                    joblib.dump(X, save_path)
                    
        return [[features[i] for features in features_list] for i in range(self.n_dfs)]
#
#     def _transform(self, Xs):
#
#         if self.pipeline_types == "func":
#             return self.pipeline_func_map_list(self)
#         elif self.
