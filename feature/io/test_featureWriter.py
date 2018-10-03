import tempfile
from pathlib import Path
from unittest import TestCase

from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline

from feature.io.io import FeatureWriter


class TestFeatureWriter(TestCase):

    def test_transform(self):
        tokenizer = CountVectorizer()
        pipeline = Pipeline([("tokenizer", tokenizer)])

        data_x = [
            "I have a pen",
            "I don't know anything",
            "You love a cat",
            "Don't kick off me"
        ]

        pipeline.fit(data_x)

        sut = FeatureWriter(pipeline=pipeline)

        with tempfile.TemporaryDirectory() as tmpdir:
            sut.transform(data_x, tmpdir, "tokens", data_names="data1")

            loaded_list = joblib.load(Path(tmpdir).joinpath("{}_{}.joblib.dump".format("data1", "tokens")))

