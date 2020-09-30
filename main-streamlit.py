import streamlit as st
from random import sample
from load_bert_model import load_bert_model
from torch.nn import Module

MODEL_NAME = "models/bert_cased_6.pt"
PRODUCTION_MODE = False

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


@st.cache(hash_funcs={Module: id})
def load_model():
    return load_bert_model(MODEL_NAME)


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

Enter some text below. A Deep Learning model tells you if it sounds\
     like a Donald Trump's tweet. Try to fool the model!
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

    **How does the algorithm work?** A Deep Learning model was trained \
    over 120,000 tweets to automatically understand the syntax and semantics\
    of Donald Trump's tweets. Visit the \
    [github repository](https://github.com/oulianov/soundslikeatrumptweet) for more details.
    
    Made by [Nicolas Oulianov](https://github.com/oulianov/) as a silly joke. \
    I do not support Donald Trump. 
    """
)