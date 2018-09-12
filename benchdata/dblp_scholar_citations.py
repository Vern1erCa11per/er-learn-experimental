from pathlib import Path

import pandas as pd

from benchdata.bench_data import BenchData

class DblpGoogleScholarCitationsData(BenchData):
    ENCODING = "utf-8"
    PERFECT_MATCH_PATH = Path(__file__).parent.joinpath('../data/raw/citations/matches_dblp_scholar.csv')
    DBLP_CITATIONS_PATH = Path(__file__).parent.joinpath("../data/raw/citations/dblp.csv")
    GOOGLE_SCHOLAR_CITATIONS_PATH = Path(__file__).parent.joinpath("../data/raw/citations/google_scholar.csv")

    KEY_COLUMN_PAIRS = [
        {"dblp": "title", "scholar": "title"},
        {"dblp": "authors", "scholar": "authors"},
        {"dblp": "venue", "scholar": "venue"},
        {"dblp": "year", "scholar": "year"}
    ]

    DBLP_ID_COLUMN = "id"
    SCHOLAR_ID_COLUMN = "id"

    DBLP_ID_COLUMN_IN_MATCH = "dblp_id"
    SCHOLAR_ID_COLUMN_IN_MATCH = "google_scholar_id"

    def __init__(self):
        self.dblp_df: pd.DataFrame = None
        self.scholar_df: pd.DataFrame = None
        self.perfect_match: pd.DataFrame = None
        self._dblp_id_to_index_map = None
        self._scholar_id_to_index_map = None

    def dblp_id_to_index(self, id):
        return self._dblp_id_to_index_map[id]

    def scholar_id_to_index(self, id):
        return self._scholar_id_to_index_map[id]

    def get_perfect_match_index(self, matrix=False):
        index_df = self.perfect_match.apply(self._perfect_match_row_to_index, axis=1)
        if matrix:
            return index_df.as_matrix()
        else:
            return index_df

    def _perfect_match_row_to_index(self, row):
        return pd.Series({"dblp_index": self._dblp_id_to_index_map[row[self.DBLP_ID_COLUMN_IN_MATCH]],
                "scholar_index": self._scholar_id_to_index_map[row[self.SCHOLAR_ID_COLUMN_IN_MATCH]]})

    def load(self):
        self.scholar_df = pd.read_csv(self.GOOGLE_SCHOLAR_CITATIONS_PATH,
                                      encoding=DblpGoogleScholarCitationsData.ENCODING)
        self.dblp_df = pd.read_csv(self.DBLP_CITATIONS_PATH, encoding=DblpGoogleScholarCitationsData.ENCODING)
        self.perfect_match = pd.read_csv(DblpGoogleScholarCitationsData.PERFECT_MATCH_PATH)
        self._dblp_id_to_index_map = {value: index for index, value in self.dblp_df.id.to_dict().items()}
        self._scholar_id_to_index_map = {value: index for index, value in self.scholar_df.id.to_dict().items()}

    def default_fillna(self):
        self.default_scholar_fillna()
        self.default_dblp_fillna()

    def default_scholar_fillna(self):
        self.scholar_df["title"].fillna("", inplace=True)
        self.scholar_df["authors"].fillna("", inplace=True)
        self.scholar_df["venue"].fillna("", inplace=True)
        # TODO fill with average?
        self.scholar_df["year"].fillna(0, inplace=True)

    def default_dblp_fillna(self):
        self.dblp_df["title"].fillna("", inplace=True)
        self.dblp_df["authors"].fillna("", inplace=True)
        self.dblp_df["venue"].fillna("", inplace=True)
        # TODO fill with average?
        self.dblp_df["year"].fillna(0, inplace=True)

    def default_preprocess(self):
        self.default_fillna()
