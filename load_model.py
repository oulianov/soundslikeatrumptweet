from transformers import BertModel, BertConfig, AutoTokenizer
from torch import save, load, device, nn
import numpy as np

import marshal
from types import FunctionType
from sklearn.base import BaseEstimator, TransformerMixin, MetaEstimatorMixin
import joblib


class BERT_Arch(nn.Module):
    def __init__(self, bert, tokenizer):
        super(BERT_Arch, self).__init__()
        self.bert = bert
        self.tokenizer = tokenizer
        # dropout layer
        self.dropout = nn.Dropout(0.2)
        # relu activation function
        self.relu = nn.ReLU()
        # dense layer 1
        self.fc1 = nn.Linear(256, 32)
        # Batch normalization
        self.batchnorm_32 = nn.BatchNorm1d(32)
        # dense layer 2
        self.fc2 = nn.Linear(32, 8)
        self.batchnorm_8 = nn.BatchNorm1d(8)
        # Output layer
        self.fc3 = nn.Linear(8, 2)
        # softmax activation function
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input_ids, attention_mask, **args):
        cls_hs = self.bert(input_ids, attention_mask=attention_mask, **args)[0][:, 0, :]
        x = self.dropout(cls_hs)
        # First hidden layer
        x = self.fc1(x)
        x = self.relu(x)
        # x = self.batchnorm_32(x)
        x = self.dropout(x)
        # Second layer
        x = self.fc2(x)

        x = self.relu(x)
        x = self.batchnorm_8(x)
        x = self.dropout(x)
        # output layer
        x = self.fc3(x)
        # apply softmax activation
        x = self.softmax(x)
        return x

    def predict_proba(self, txt):
        tok = self.tokenizer(
            txt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=280,
        )
        return np.exp(self(**tok).detach().numpy())


def load_bert_model(MODEL_NAME):
    config = BertConfig(
        **{
            "attention_probs_dropout_prob": 0.1,
            "gradient_checkpointing": False,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "hidden_size": 256,
            "initializer_range": 0.02,
            "intermediate_size": 1024,
            "layer_norm_eps": 1e-12,
            "max_position_embeddings": 512,
            "model_type": "bert",
            "num_attention_heads": 4,
            "num_hidden_layers": 4,
            "pad_token_id": 0,
            "type_vocab_size": 2,
            "vocab_size": 30522,
        }
    )
    bert = BertModel(config)
    tokenizer = AutoTokenizer.from_pretrained(
        "prajjwal1/bert-mini",
        model_max_length=280,
        tokenize_chinese_chars=False,
    )
    model = BERT_Arch(bert, tokenizer)

    model.load_state_dict(load(MODEL_NAME, map_location=device("cpu")))
    model.eval()
    return model


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


def load_log_model(MODEL_NAME):
    model = joblib.load(MODEL_NAME)
    return model


class metaModel:
    def __init__(self, bert_model_name, log_model_name):
        self.bert_model = load_bert_model(bert_model_name)
        self.log_model = load_log_model(log_model_name)

    def predict_proba(self, txt):
        bert_proba = self.bert_model.predict_proba(txt)
        log_proba = self.log_model.predict_proba(txt)
        return bert_proba * log_proba
