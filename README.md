# Sounds like a Trump tweet

Deep Learning model that tells you if a text sounds like a Trump tweet. 

[Go to this website to test the model!](http://soundslikeatrumptweet.herokuapp.com/) You might have to wait a bit for the model to load.

It was trained on about 280,000 tweets assembled from various Kaggle tweet datasets. 7% of those tweets were labeled as from Donald Trump. 

## Model used

This model is actually composed of two submodels:

- **A semantic model,** based on a small **BERT model for classification**. It analyzes the lowercase content of the text to detect words and sentence structures commonly used in Donald Trump. It was trained with PyTorch and Transfer Learning on Google Colab's GPU.
- **A syntactic model,** based on a logistic regression. It analyzes the text on a **character level,** for example to favour UPPERCASE words, and to reject 😔 emojis. It was trained with scikit-learn. 

The predictions of the semantic and the syntactic submodels are multiplied together to yield the final prediction. This means that if a text is classified as a Trump tweet, it is because both submodels agree. 

## Constraints

1. In the dataset, there are only tweets. But on the website, the user may enter any text. This means that the model will have to work with text *out of distribution*, that doesn't look like the training data. 
2. Heroku, the service hosting the website, imposes severe limits on disk size (500 Mo), memory, and CPU. Thus, model can't be too large or computationaly expensive.
3. More than precise, model's predictions have to be fun. A user *should* be able to fool the model. 

The constraints were solved the following way:

1. Instead of just one, I used a combination of two submodels to improve the model's robustness. In practice, this lowers Recall and improves Sensitivity. 
2. Instead of having a giant cased BERT model (over 1GB in size), I used a smaller version of the architecture. Thanks to [user Prajjwall](https://huggingface.co/prajjwal1/bert-mini) for providing this model! To keep case-dection, I combined it with a syntactic model. 
3. To make sure that the model was foolable in a fun way, I asked friends to test the model, and adjusted parameters according to their feedback. I also added on the website hints on how to fool the model.

## Note

This side-project was made as a silly joke. I do not support Donald Trump.
