import streamlit as st
from random import sample
from load_bert_model import load_bert_model
from torch.nn import Module

MODEL_NAME = "minibert_cased_3.pt"
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


@st.cache(hash_funcs={Module: id})
def load_model():
    return load_bert_model(MODEL_NAME)


model = load_model()


# @st.cache
def get_description(score):
    desc_list = [
        [
            "Not written like a Trump tweet. For sure.",
            "This is definitely not a Trump tweet.",
        ],
        [
            "Doesn't sound like a Trump tweet.",
            "Not Trump-esque at all.",
            "This has no Trump's tweeter vibes.",
        ],
        [
            "Not very Trump-like.",
            "Not very Trump-ish.",
            "Would be weird if Trump tweeted that, wouldn't it?",
        ],
        [
            "Doesn't really sound like Trump.",
            "Trump wouldn't write that on tweeter.",
            "Trump wouldn't tweet it like that.",
        ],
        [
            "Maybe some Trump vibes.",
            "Feels somehow like Trump.",
            "Kinda Trump and kinda not trump.",
        ],
        [
            "This is written a bit like Trump.",
            "Trump could maybe write this.",
        ],
        [
            "Trump really could have written this (while tired).",
            "This sounds a lot like Trump.",
            "Lots of Trump energy.",
        ],
        [
            "Classic Trump style.",
            "That's Trump's style for sure!",
            "Definitely Trump's tweeter mood!",
        ],
        [
            "Yep, this is written like a Trump tweet.",
            "Trump could have literally tweeted that.",
            "That's pretty much how Trump writes on twitter.",
            "Totally written like a Trump tweet.",
        ],
    ]
    score_range = [i / len(desc_list) for i in range(len(desc_list) + 1)]
    for i, ref in enumerate(score_range):
        if ref > score:
            return sample(desc_list[i - 1], 1)[0]


st.markdown(
    """
# Sounds like a Trump tweet

Enter some text below. A Deep Learning model tells you if it's written\
     like a Donald Trump's tweet. Try to fool the model!
"""
)

user_input = st.text_area(
    label="Is this written like a Donald Trump's tweet?",
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
    ### Tips to make your tweet Trump-like
    - Add exclamation points!!!
    - Write. Short. Sentences. 
    - Use Trump's linguo.

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
