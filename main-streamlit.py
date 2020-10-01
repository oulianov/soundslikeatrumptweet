import streamlit as st
from random import sample
from load_bert_model import load_bert_model
from torch.nn import Module

MODEL_NAME = "models/minibert_cased_3.pt"
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
        [
            "Not written like a Trump tweet. For sure.",
            "This is definitely not a Trump tweet.",
        ],
        ["Doesn't sound much like a Trump tweet.", "Not Trump-esque at all."],
        ["Not very Trump-like.", "Hmm, not very Trump-ish."],
        ["Doesn't really sound like Trump."],
        ["Trump vibes for sure.", "Feels somehow like Trump."],
        [
            "This is written a bit like Trump.",
            "Trump could maybe write this.",
        ],
        ["Trump pretty much writes like this.", "This sounds a lot like Trump."],
        [
            "Classic Trump style.",
            "That's Trump's style for sure!",
        ],
        [
            "This is how Trump write!",
            "Trump could easily write that.",
            "That's pretty much how Trump writes.",
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
    - WRITE IN ALL CAPS
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