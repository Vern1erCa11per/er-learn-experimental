from pathlib import Path

import pandas as pd

from benchdata.bench_data import BenchData

AMAZON_PRODUCT_PATH = Path(__file__).parent.joinpath("../data/raw/Amazon-GoogleProducts/Amazon.csv")
GOOGLE_PRODUCT_PATH = Path(__file__).parent.joinpath("../data/raw/Amazon-GoogleProducts/GoogleProducts.csv")


class AmazonGoogleData(BenchData):
    ENCODING = "iso-8859-1"
    PERFECT_MATCH_PATH = '../../data/raw/Amazon-GoogleProducts/Amzon_GoogleProducts_perfectMapping.csv'
    KEY_COLUMN_PAIRS = [
        {"amazon": "title", "google": "name"},
        {"amazon": "description", "google": "description"},
        {"amazon": "manufacturer", "google": "manufacturer"},
        {"amazon": "price", "google": "price"}
    ]
    AMAZON_ID_COLUMN_IN_MATCH = "idAmazon"
    GOOGLE_ID_COLUMN_IN_MATCH = "idGoogleBase"

    def __init__(self):
        self.google_products_df: pd.DataFrame = None
        self.amazon_df: pd.DataFrame = None
        self.perfect_match: pd.DataFrame = None
        self._amazon_id_to_index_map = None
        self._google_id_to_index_map = None

    def amazon_id_to_index(self, id):
        return self._amazon_id_to_index_map[id]

    def google_id_to_index(self):
        return self._google_id_to_index_map[id]

    def get_perfect_match_index(self, matrix=False):
        index_df = self.perfect_match.apply(self._perfect_match_row_to_index, axis=1)
        if matrix:
            return index_df.as_matrix()
        else:
            return index_df

    def _perfect_match_row_to_index(self, row):
        return pd.Series({"amazon_index": self._amazon_id_to_index_map[row[self.AMAZON_ID_COLUMN_IN_MATCH]],
                "google_index": self._google_id_to_index_map[row[self.GOOGLE_ID_COLUMN_IN_MATCH]]})

    def load(self):
        # iso-8859-1
        self.google_products_df = pd.read_csv(GOOGLE_PRODUCT_PATH, encoding=AmazonGoogleData.ENCODING)
        self.amazon_df = pd.read_csv(AMAZON_PRODUCT_PATH, encoding=AmazonGoogleData.ENCODING)
        self.perfect_match = pd.read_csv(AmazonGoogleData.PERFECT_MATCH_PATH)
        self._amazon_id_to_index_map = {value: index for index, value in self.amazon_df.id.to_dict().items()}
        self._google_id_to_index_map = {value: index for index, value in self.google_products_df.id.to_dict().items()}

    def default_fillna(self):
        self.default_google_fillna()
        self.default_amazon_fillna()

    def default_google_fillna(self):
        self.google_products_df["name"].fillna("", inplace=True)
        self.google_products_df["description"].fillna("", inplace=True)
        self.google_products_df["manufacturer"].fillna("", inplace=True)

    def default_amazon_fillna(self):
        self.amazon_df["title"].fillna("", inplace=True)
        self.amazon_df["description"].fillna("", inplace=True)
        self.amazon_df["manufacturer"].fillna("", inplace=True)

    # TODO default extract int from price
