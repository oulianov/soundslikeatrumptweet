import streamlit as st
from random import sample
from load_model import metaModel, MyFunctionTransformer

bert_model_name = "minibert_cased_3.pt"
log_model_name = "tfidf_len_vocab_log_4"
PRODUCTION_MODE = False

st.beta_set_page_config(page_title="Sounds like a Trump tweet", page_icon="trump.jpeg")

if PRODUCTION_MODE:
    # Hide Hamburger menu
    hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>

    """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)
else:
    bert_model_name = "models/" + bert_model_name
    log_model_name = "models/" + log_model_name


@st.cache(hash_funcs={metaModel: id})
def load_meta_model():
    model = metaModel(bert_model_name, log_model_name)
    return model


model = load_meta_model()


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
            "Not really Trump's tweeter vibes.",
        ],
        [
            "Not very Trump-like.",
            "Not very Trump-ish.",
            "Would be weird if Trump tweeted that, wouldn't it?",
        ],
        [
            "Doesn't really sound like Trump.",
            "Trump wouldn't write that on tweeter (hopefuly).",
            "Trump wouldn't exactly tweet it like that.",
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
    for char in "#-=@+%":  # These make the model fail
        user_input = user_input.replace(char, "")
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
    - YELL IN ALL CAPS
    - Add exclamation points!!!
    - Write short sentences. 
    - Use Trump's linguo.
    
    --------------
    # About 

    **How does the algorithm work?** A Deep Learning model was trained \
    over 280,000 tweets to automatically understand the syntax and semantics\
    of Donald Trump's tweets. Visit the \
    [github repository](https://github.com/oulianov/soundslikeatrumptweet) for more details.
    
    Made by [Nicolas Oulianov](https://github.com/oulianov/) as a silly joke. \
    I do not support Donald Trump. 
    """
)
