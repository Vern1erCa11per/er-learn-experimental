import logging
from pathlib import Path

from sklearn.externals import joblib
from sklearn.pipeline import Pipeline

from feature.pipelines import ConcatPipeline
import scipy.sparse as sparse
import numpy as np

loggger = logging.getLogger(__name__)

class FeatureWriter(object):

    def __init__(self, pipeline:Pipeline):
        self.pipeline = pipeline
        self._parent_dir_path = None

    def transform(self, X, save_dir, feature_name, data_names):
        transformed_X = self.pipeline.transform(X)

        self._parent_dir_path = Path(save_dir)
        self._parent_dir_path.mkdir(parents=True, exist_ok=True)

        if isinstance(self.pipeline, ConcatPipeline):
            for x, data_name in zip(transformed_X, data_names):
                self.save_single_feature(feature_x=x, data_name=data_name, feature_name=feature_name)

        else:
            self.save_single_feature(feature_x=transformed_X, data_name=data_names, feature_name=feature_name)


    def save_single_feature(self, feature_x, data_name, feature_name):
        file_path = self._parent_dir_path.joinpath("{}_{}".format(data_name, feature_name))

        if sparse.isspmatrix(feature_x):
            sparse.save_npz(file_path.with_suffix(".sparse.npz"), feature_x)
        elif isinstance(feature_x, np.ndarray):
            np.savez_compressed(file_path.with_suffix(".npz"), feature_x)
        elif isinstance(feature_x, list):
            filename = file_path.with_suffix(".joblib.dump")
            loggger.info("writing feature into {}".format(filename))
            joblib.dump(feature_x, filename, compress=3)
        else:
            raise TypeError("unsupported feature type")
