import pandas as pd
import numpy as np
import re

relevant_columns = ["content", "trump"]


def load_nontrump_data(filepath, txt_col):
    df = pd.read_csv(filepath, error_bad_lines=False)
    df["trump"] = 0
    df.rename({txt_col: "content"}, axis=1, inplace=True)
    df = df[relevant_columns]
    return df


# Tweets by trump
trump = pd.read_csv("data/trump.csv")
trump["trump"] = 1
trump = trump[relevant_columns]
trump.head(5)

# Other tweets. This a collection of various generic tweets dataset.
filepaths = [
    "elonmusk.csv",
    "companytweets.csv",
    "financetweets.csv",
    "populartweets.csv",
    "JoeBidenTweets.csv",
    "factcheck.csv",
    "AdamSavageTweets.csv",
    "BarackObama.csv",
    "HillaryClintonTweets.csv",
    "KimKardashianTweets.csv",
    "NeildeGrasseTysonTweets.csv",
    "RichardDawkins.csv",
    "ScottKelly.csv",
]
filepaths = ["data/" + i for i in filepaths]

txt_columns = [
    "text",
    "tweet ",
    "text",
    "content",
    "tweet",
    "tweet",
    "text",
    "text",
    "text",
    "text",
    "text",
    "text",
    "text",
]
other_df = []

for filepath, txt_col in zip(filepaths, txt_columns):
    other_df.append(load_nontrump_data(filepath, txt_col))

dataset = trump

for df in other_df:
    dataset = dataset.append(df)

# We are only interested in writing style, so we remove links.


def remove_links(txt):
    return re.sub(r"\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*", "", txt)


def remove_trump(txt):
    return re.sub(r"(@|#\w*\b|--|(d|D)onald|J\.|(t|T)rump)", " ", txt)


dataset["content"] = dataset["content"].map(remove_links)
dataset["content"] = dataset["content"].map(remove_trump)

# Save
dataset.to_csv("dataset_4.csv")
