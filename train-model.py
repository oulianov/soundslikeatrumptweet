import pandas as pd
import numpy as np
import re

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.linear_model import LogisticRegressionCV
from xgboost import XGBClassifier
from joblib import dump, load
from gensim.models import Word2Vec
from gensim.sklearn_api import D2VTransformer

import marshal
from types import FunctionType
from sklearn.base import BaseEstimator, TransformerMixin, MetaEstimatorMixin

DATASET_NAME = "dataset_5.csv"
MODEL_NAME = "models/tfidf_len_vocab_log_4"


# Used to circonvent limitations when saving custom transformers
class MyFunctionTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, f):
        self.func = f

    def __call__(self, X):
        return self.func(X)

    def __getstate__(self):
        self.func_name = self.func.__name__
        self.func_code = marshal.dumps(self.func.__code__)
        del self.func
        return self.__dict__

    def __setstate__(self, d):
        d["func"] = FunctionType(
            marshal.loads(d["func_code"]), globals(), d["func_name"]
        )
        del d["func_name"]
        del d["func_code"]
        self.__dict__ = d

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return self.func(X)


print("Reading data.")
dataset = pd.read_csv(DATASET_NAME).dropna()  # .sample(5000)
dataset = dataset[["content", "trump"]]
full_dataset = dataset  # .sample(1000)
dataset = dataset  # .sample(10000)

share_trump = dataset["trump"].sum() / dataset.shape[0]


@MyFunctionTransformer
def get_nb_unique_tokens(x):
    import numpy as np  # will raise exception if not imported

    return np.array([len(set(t.split())) for t in x]).reshape(-1, 1)


@MyFunctionTransformer
def get_avg_token_len(x):
    import numpy as np  # will raise exception if not imported

    def get_txt_len(x):
        return np.array([len(t) for t in x]).reshape(-1, 1)

    def get_nb_tokens(x):
        return np.array([len(t.split()) for t in x]).reshape(-1, 1)

    return get_txt_len(x) / np.maximum(1, get_nb_tokens(x))


count_characters = (
    "count_characters",
    TfidfVectorizer(
        lowercase=False,
        ngram_range=(1, 2),
        analyzer="char",
        min_df=20,
        max_df=0.95,
        max_features=2000,
        binary=True,
        dtype=np.float32,
        strip_accents="ascii",
    ),
)

tfidf_2grams = (
    "freq_2grams",
    TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        analyzer="word",
        min_df=4,
        dtype=np.float32,
        max_df=0.95,
        max_features=10000,
        use_idf=False,
    ),
)

logistic_regression = (
    "logistic",
    LogisticRegressionCV(
        Cs=[2000, 5000, 10000, 15000, 20000],
        scoring="f1",
        n_jobs=-1,
        class_weight="balanced",
        max_iter=20000,
    ),
)


log = Pipeline(
    [
        (
            "vectorize",
            FeatureUnion(
                [
                    ("nb_unique_tokens", get_nb_unique_tokens),
                    count_characters,
                    tfidf_2grams,
                ],
                n_jobs=4,
            ),
        ),
        logistic_regression,
    ],
    verbose=True,
)

print("Start training.")
log.fit(dataset["content"].to_list(), dataset["trump"])


print("Saving model.")
# Set nb of jobs to 1 for streamlit-compatibility
log.steps[0][1].n_jobs = 1
log.steps[1][1].n_jobs = 1
dump(log, MODEL_NAME)
print("Done.")
print(log.score)

# Delete everything and reload model (test)

del log
del get_nb_unique_tokens
del get_avg_token_len
log = load(MODEL_NAME)
