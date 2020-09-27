import streamlit as st
from joblib import load
from sklearn.pipeline import Pipeline
from random import sample

import marshal
from types import FunctionType
from sklearn.base import BaseEstimator, TransformerMixin

PRODUCTION_MODE = True

st.beta_set_page_config(page_title="Sounds like a Trump tweet", page_icon="trump.jpeg")

# Hide Hamburger menu
hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>

"""
if PRODUCTION_MODE:
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)

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


@st.cache(hash_funcs={Pipeline: id})
def load_model():
    model = load("tfidf_len_vocab_log_3")
    # Set nb of jobs to 1 for streamlit-compatibility
    model.steps[0][1].n_jobs = 1
    model.steps[1][1].n_jobs = 1
    return model


model = load_model()


# @st.cache
def get_description(score):
    desc_list = [
        ["Not a Trump tweet. For sure.", "This is so not Trump it could be Biden."],
        ["Doesn't sound much like Trump.", "Not Trump-esque enough."],
        ["Not very Trump-like."],
        ["Doesn't really sound like Trump."],
        ["Trump vibes for sure.", "Feels somehow like Trump."],
        [
            "This sounds a bit like Trump.",
            "This could be Trump (I mean, you never know).",
        ],
        ["This sounds a lot like Trump.", "This sounds VERY MUCH like Trump."],
        [
            "Trump would totally tweet that.",
            "In a parallel universe Trump did tweet that.",
        ],
        [
            "Totally sounds like Trump (could be him).",
            "Yep. Definitely sounds like Trump!",
            "That's ABSOLUTELY how Trump sounds.",
        ],
    ]
    score_range = [i / len(desc_list) for i in range(len(desc_list) + 1)]
    for i, ref in enumerate(score_range):
        if ref > score:
            return sample(desc_list[i - 1], 1)[0]


st.markdown(
    """
# Sounds like a Trump tweet

Enter some text below. A machine learning algorithm tells you if it sounds\
     like a Donald Trump's tweet. Try to fool the algorithm!
"""
)

user_input = st.text_area(
    label="Does this text sounds like a Donald Trump's tweet?",
    max_chars=280,
    height=128,
).strip()

if user_input != "":
    score = model.predict_proba([user_input])[:, 1]
    st.markdown(
        f"""
    ## {get_description(score)}
    ```"Trumpiness" score: {round(score[0],4)}```
    """
    )

st.markdown(
    """
    --------------
    # About 

    **How does the algorithm work?** A machine learning algorithm was trained \
    over 280,000 tweets to automatically understand the syntax and semantics\
    of Donald Trump's tweets. Visit the \
    [github repository](https://github.com/oulianov/soundslikeatrumptweet) for more details.
    
    Made by [Nicolas Oulianov](https://github.com/oulianov/) as a silly joke. \
    I do not support Donald Trump. 
    """
)