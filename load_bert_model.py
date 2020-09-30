from transformers import DistilBertModel, DistilBertConfig, AutoTokenizer
from torch import load, device, nn
import numpy as np


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
        self.fc1 = nn.Linear(768, 32)
        # Batch normalization
        self.batchnorm_32 = nn.BatchNorm1d(32)
        # dense layer 2
        self.fc2 = nn.Linear(32, 8)
        self.batchnorm_8 = nn.BatchNorm1d(8)
        # Output layer
        self.fc3 = nn.Linear(8, 2)
        # softmax activation function
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input_ids, attention_mask):
        cls_hs = self.bert(input_ids, attention_mask=attention_mask)[0][:, 0, :]
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


def load_bert_model(MODEL_NAME, cased=True):

    if cased:
        config = DistilBertConfig(
            **{
                "activation": "gelu",
                "attention_dropout": 0.1,
                "dim": 768,
                "dropout": 0.1,
                "hidden_dim": 3072,
                "initializer_range": 0.02,
                "max_position_embeddings": 512,
                "model_type": "distilbert",
                "n_heads": 12,
                "n_layers": 6,
                "output_past": True,
                "pad_token_id": 0,
                "qa_dropout": 0.1,
                "seq_classif_dropout": 0.2,
                "sinusoidal_pos_embds": False,
                "tie_weights_": True,
                "vocab_size": 28996,
            }
        )
        bert = DistilBertModel(config)
        tokenizer = AutoTokenizer.from_pretrained(
            "distilbert-base-cased",
            model_max_length=280,
            tokenize_chinese_chars=False,
        )
        model = BERT_Arch(bert, tokenizer)
    else:
        pass

    model.load_state_dict(load(MODEL_NAME, map_location=device("cpu")))
    model.eval()
    return model