from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.linear_model import LogisticRegressionCV
from xgboost import XGBClassifier
from joblib import dump, load
import pandas as pd
import numpy as np

DATASET_NAME = "dataset_4.csv"
MODEL_NAME = "tfidf_len_vocab_log_2"

import marshal
from types import FunctionType
from sklearn.base import BaseEstimator, TransformerMixin


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
full_dataset = dataset
# dataset = dataset.sample(5000)

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


log = Pipeline(
    [
        (
            "vectorize",
            FeatureUnion(
                [
                    ("nb_unique_tokens", get_nb_unique_tokens),
                    # ("nb_tokens", get_nb_tokens),
                    # ("len_txt", get_txt_len),
                    ("avg_token_len", get_avg_token_len),
                    (
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
                    ),
                    (
                        "tfidf_words",
                        TfidfVectorizer(
                            lowercase=True,
                            ngram_range=(1, 2),
                            analyzer="word",
                            min_df=5,
                            dtype=np.float32,
                            max_df=0.9,
                            max_features=100000,
                        ),
                    ),
                ],
                n_jobs=4,
            ),
        ),
        # ('pca', TruncatedSVD(n_components=128)),
        # (
        #    "xgboost",
        #    XGBClassifier(
        #        max_depth=10, n_estimators=120, base_score=share_trump, n_jobs=4
        #    ),
        # ),
        (
            "logistic",  # penalty="l1", solver="saga",
            LogisticRegressionCV(
                Cs=[2000, 5000, 10000, 15000, 20000], scoring="f1", n_jobs=-1
            ),
        ),
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

# Delete everything and reload model to display the F1.

del log
del get_nb_unique_tokens
del get_avg_token_len
log = load(MODEL_NAME)

from sklearn.metrics import f1_score

print(
    f"F1: {f1_score(full_dataset['trump'].to_numpy(), log.predict(full_dataset['content'].to_list()))}"
)