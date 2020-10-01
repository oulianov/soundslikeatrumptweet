from transformers import BertModel, BertConfig, AutoTokenizer
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


def load_bert_model(MODEL_NAME, cased=True):

    if cased:
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
    else:
        pass

    model.load_state_dict(load(MODEL_NAME, map_location=device("cpu")))
    model.eval()
    return model