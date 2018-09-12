from pathlib import Path

import pandas as pd

from benchdata.bench_data import BenchData

class AmazonWalmartProductsData(BenchData):
    ENCODING = "utf-8"
    PERFECT_MATCH_PATH = Path(__file__).parent.joinpath('../data/raw/products/matches_walmart_amazon.csv')
    AMAZON_PRODUCTS_PATH = Path(__file__).parent.joinpath("../data/raw/products/amazon.csv")
    WALMART_PRODUCTS_PATH = Path(__file__).parent.joinpath("../data/raw/products/walmart.csv")

    KEY_COLUMN_PAIRS = [
        {"amazon": "brand", "walmart": "brand"},
        {"amaaon": "modelno", "walmart": "modelno"},
        {"amazon": "category1", "walmart": "groupname"},
        {"amazon": "title", "walmart": "title"},
        {"amazon": "price", "walmart": "price"},
        # FIXME is this a correct correspondence?
        {"amazon": "techdetails", "walmart": "shelfdescr"},
        {"amazon": "proddescrlong", "walmart": "longdescr"}
    ]

    AMAZON_TEXT_COLUMNS = ["brand", "modelno", "category1", "pcategory1", "category2", "pcategory2", "title",
                           "techdetails", "proddescrshort", "proddescrlong", "dimensions"]
    WALMART_TEXT_COLUMNS = ["brand", "groupname", "title", "shelfdescr", "shortdescr", "longdescr", "modelno"]

    AMAZON_ID_COLUMN = "custom_id"
    WALMART_ID_COLUMN = "custom_id"

    AMAZON_ID_COLUMN_IN_MATCH = "id2"
    WALMART_ID_COLUMN_IN_MATCH = "id1"

    def __init__(self):
        self.amazon_df: pd.DataFrame = None
        self.walmart_df: pd.DataFrame = None
        self.perfect_match: pd.DataFrame = None
        self._amazon_id_to_index_map = None
        self._walmart_id_to_index_map = None

    def amazon_id_to_index(self, id):
        return self._amazon_id_to_index_map[id]

    def walmart_id_to_index(self, id):
        return self._walmart_id_to_index_map[id]

    def get_perfect_match_index(self, matrix=False):
        index_df = self.perfect_match.apply(self._perfect_match_row_to_index, axis=1)
        if matrix:
            return index_df.as_matrix()
        else:
            return index_df

    def _perfect_match_row_to_index(self, row):
        return pd.Series({"amazon_index": self._amazon_id_to_index_map[row[self.AMAZON_ID_COLUMN_IN_MATCH]],
                "walmart_index": self._walmart_id_to_index_map[row[self.WALMART_ID_COLUMN_IN_MATCH]]})

    def load(self):
        self.walmart_df = pd.read_csv(self.WALMART_PRODUCTS_PATH,
                                      encoding=AmazonWalmartProductsData.ENCODING)
        self.amazon_df = pd.read_csv(self.AMAZON_PRODUCTS_PATH, encoding=AmazonWalmartProductsData.ENCODING)
        self.perfect_match = pd.read_csv(AmazonWalmartProductsData.PERFECT_MATCH_PATH)
        self._amazon_id_to_index_map = {value: index for index, value in self.amazon_df[self.AMAZON_ID_COLUMN].to_dict().items()}
        self._walmart_id_to_index_map = {value: index for index, value in self.walmart_df[self.WALMART_ID_COLUMN].to_dict().items()}

    def default_fillna(self):
        self.default_walmart_fillna()
        self.default_amazon_fillna()

    def default_walmart_fillna(self):
        for column_name in self.WALMART_TEXT_COLUMNS:
            self.walmart_df[column_name].fillna("", inplace=True)

        # TODO fill with average?
        self.walmart_df["price"].fillna(0, inplace=True)

    def default_amazon_fillna(self):
        for column_name in self.AMAZON_TEXT_COLUMNS:
            self.amazon_df[column_name].fillna("", inplace=True)

        # TODO fill with average?
        self.amazon_df["price"].fillna(0, inplace=True)

    def default_preprocess(self):
        self.default_fillna()
